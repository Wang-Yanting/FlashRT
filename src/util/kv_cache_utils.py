import torch
from src.util.string_utils import *
from src.util.utils import *
from transformers.cache_utils import StaticCache
from transformers import StaticCache
import torch
def initialize_kv_cache(model,context_left,context_right,payload,query, adv, target_answer, override_payload_ids=None):

    kv_caches = []  # list to store kv_cache for each prompt
    suffix_manager = SuffixManager(model,context_left,context_right,query, adv, payload, target_answer)
    if override_payload_ids is not None:
        _override_suffix_manager_payload_ids(suffix_manager, override_payload_ids)
    input_ids = torch.tensor([suffix_manager.get_prompt_with_target_ids()]).to(model.model.device)
    with torch.no_grad():
        outputs = model.model(input_ids, use_cache=True)

    kv_caches = outputs.past_key_values  # extract kv_cache from the output
    return kv_caches,suffix_manager


def _override_suffix_manager_payload_ids(suffix_manager, override_payload_ids):
    suffix_manager.payload_ids = list(override_payload_ids)
    suffix_manager.prompt_ids = (
        suffix_manager.before_prefix_ids
        + suffix_manager.prefix_ids
        + suffix_manager.payload_ids
        + suffix_manager.suffix_ids
        + suffix_manager.context_ids_right
        + suffix_manager.after_context_ids
    )
    suffix_manager.prompt_with_target_ids = suffix_manager.prompt_ids + suffix_manager.target_ids
    suffix_manager.init_prefix_suffix_slice()
    suffix_manager.init_other_slices()
    suffix_manager.init_context_right_recompute_slice()
def slice_kv_cache(cache, k1,k2):
    # Create a new cache with the first k token positions
    new_cache = []
    for key, value in cache:
        # Slice the key and value tensors to keep only the first k positions
        new_key = key[:, :, k1:k2, :]
        new_value = value[:, :, k1:k2, :]
        new_cache.append((new_key, new_value))
    return new_cache

def concat_kv_cache(cache1, cache2):
    new_cache = []
    for (key1, value1), (key2, value2) in zip(cache1, cache2):
        # Concatenate along the sequence length dimension (dimension 2)
        new_key = torch.cat([key1, key2], dim=2)
        new_value = torch.cat([value1, value2], dim=2)
        new_cache.append((new_key, new_value))
    return new_cache

def extend_kv_cache(kv_cache, model,new_input_ids):
    outputs = model.model(input_ids=new_input_ids, past_key_values=tuple(kv_cache), use_cache=True)
    new_kv_caches = outputs.past_key_values 
    return tuple(new_kv_caches)

def random_kv_cache_eviction(cache, ratio=0.5):
    # Randomly evict a portion of the kv cache
    import random
    new_cache = []
    
    # Determine positions to keep once for all layers and heads
    first_key = cache[0][0]  # Get the first key tensor to determine sequence length
    seq_len = first_key.shape[2]  # sequence length dimension
    keep_len = int(seq_len * (1 - ratio))
    
    # Randomly select positions to keep (same for all layers and heads)
    positions = list(range(seq_len))
    keep_positions = sorted(random.sample(positions, keep_len))
    
    for key, value in cache:
        # Create new tensors with only the kept positions
        new_key = key[:, :, keep_positions, :]
        new_value = value[:, :, keep_positions, :]
        new_cache.append((new_key, new_value))
    return new_cache,keep_positions


def to_static(model, kv_cache):
    import torch
    if not isinstance(kv_cache, list) and not isinstance(kv_cache, tuple):
        
        kv_cache = kv_cache.to_legacy_cache()
        
        if torch.cuda.device_count() == 1:
            if "gemma" in model.__class__.__name__.lower():
                return legacy_to_static_gemma(model, kv_cache)
            else:
                return legacy_to_static(model, kv_cache)
        else:

            return legacy_to_static_multi_device(model, kv_cache)
    else:
        if torch.cuda.device_count() == 1:
            return to_static_single_device(model, kv_cache)
        else:
            return to_static_multi_device(model, kv_cache)
def to_static_single_device(model, kv_cache):
    # kv_cache: tuple of (k, v) per layer, each [B, Hkv, L, D]
    k0, v0 = kv_cache[0]
    if k0.dim() == 4 and k0.shape[1] != getattr(model.config, "num_key_value_heads", model.config.num_attention_heads):
        # handle BLHD -> BHLD
        kv_cache = tuple((k.transpose(1, 2).contiguous(),
                          v.transpose(1, 2).contiguous()) for (k, v) in kv_cache)
        k0, v0 = kv_cache[0]

    B, Hkv, L, D = k0.shape
    static = StaticCache(
        batch_size=B,
        max_cache_len=L,
        config=model.config,
        device=k0.device,
        dtype=k0.dtype,
    )

    for i, (k, v) in enumerate(kv_cache):
        static.key_cache[i][:, :, :L, :].copy_(k)
        static.value_cache[i][:, :, :L, :].copy_(v)

    # Mark how many tokens are valid
    if hasattr(static, "cache_lengths"):
        static.cache_lengths = torch.full((B,), L, device=k0.device, dtype=torch.long)

    # Full-visible mask for existing tokens
    attention_mask = torch.ones(B, L, dtype=torch.long, device=k0.device)
    return static, attention_mask, L
def legacy_to_static_gemma(
    model,
    legacy_cache,
    extend_cache: bool = False,
    additional_tokens: int = 0,
):
    """
    Convert a legacy (tuple of per-layer K,V) cache into a Gemma-compatible StaticCache.

    Args:
        model: HF Gemma model (e.g., GemmaForCausalLM)
        legacy_cache: tuple/list of (K, V) per layer.
                      Accepts either BHLD [B, Hkv, L, D] or BLHD [B, L, Hkv, D].
        extend_cache: if True, pre-allocate room for future tokens
        additional_tokens: number of future tokens to pre-allocate if extend_cache

    Returns:
        static_cache:  StaticCache object populated with the legacy KV states
        attention_mask: bool tensor of shape [B, L] (all ones for the prefilled tokens)
        cache_length:   int L, number of already-populated tokens
    """
    if not legacy_cache:
        raise ValueError("legacy_cache is empty")

    k0, v0 = legacy_cache[0]
    if k0.dim() != 4 or v0.dim() != 4:
        raise ValueError(f"KV tensors must be 4D, got K: {k0.dim()}D, V: {v0.dim()}D")

    B = k0.shape[0]
    device = k0.device
    dtype = k0.dtype

    # Determine expected #KV heads (Gemma uses MQA/GQA)
    exp_hkv = getattr(model.config, "num_key_value_heads",
                      getattr(model.config, "num_attention_heads", None))
    if exp_hkv is None:
        raise ValueError("Cannot infer num_key_value_heads from model.config")

    # Normalize layout to BHLD = [B, Hkv, L, D]
    # If input is BLHD [B, L, Hkv, D], transpose(1,2) -> [B, Hkv, L, D]
    def _to_bhld(k, v):
        if k.shape[1] == exp_hkv:
            # already BHLD
            return k.contiguous(), v.contiguous()
        if k.shape[2] == exp_hkv:
            # BLHD -> BHLD
            return k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()
        raise ValueError(f"Unexpected KV shape {k.shape}; cannot normalize to BHLD with Hkv={exp_hkv}.")

    legacy_bhld = []
    for (k, v) in legacy_cache:
        k_n, v_n = _to_bhld(k, v)
        legacy_bhld.append((k_n, v_n))

    k0, v0 = legacy_bhld[0]
    B, Hkv, L, D = k0.shape

    # Compute max cache len (optionally extended)
    max_cache_len = L + additional_tokens if extend_cache else L

    # Build StaticCache
    static_cache = StaticCache(
        config=model.config,
        max_batch_size=B,
        max_cache_len=max_cache_len,
        device=device,
        dtype=dtype,
    )

    # Create cache positions for the prefilled segment [0..L-1]
    cache_position = torch.arange(L, device=device)

    # Populate per layer via the official update() API
    for layer_idx, (k, v) in enumerate(legacy_bhld):
        if k.shape != (B, Hkv, L, D) or v.shape != (B, Hkv, L, D):
            raise ValueError(
                f"Layer {layer_idx} KV shapes must both be (B={B}, Hkv={Hkv}, L={L}, D={D}); "
                f"got K={tuple(k.shape)}, V={tuple(v.shape)}"
            )
        static_cache.update(
            key_states=k,
            value_states=v,
            layer_idx=layer_idx,
            cache_kwargs={"cache_position": cache_position},
        )

    # Record how many tokens are currently valid (support both fields if present)
    if hasattr(static_cache, "cache_lengths"):
        static_cache.cache_lengths = torch.full((B,), L, device=device, dtype=torch.long)
    if hasattr(static_cache, "past_seen_tokens"):  # Gemma frequently uses this name
        static_cache.past_seen_tokens = torch.full((B,), L, device=device, dtype=torch.long)

    # Gemma/SDPA prefer boolean attention masks with shape [B, L]
    attention_mask = torch.ones(B, L, dtype=torch.bool, device=device)

    return static_cache, attention_mask, L
def _find_layers_list(model):
    """
    Return an indexable list-like of Transformer layers for common model structures.
    """
    # Llama/Mistral/Gemma etc.
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # GPT2/OPT style
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    # Falcon/Baichuan variants
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError("Cannot locate transformer layers on this model.")

def _layer_device(layer):
    """
    Robustly get a layer's device. Falls back to CPU if no params/bufs.
    """
    for p in layer.parameters(recurse=True):
        return p.device
    for b in layer.buffers(recurse=True):
        return b.device
    return torch.device("cpu")

def _embed_device(model):
    # Try common embedding module paths
    candidates = [
        getattr(getattr(model, "model", None), "embed_tokens", None),
        getattr(getattr(model, "transformer", None), "wte", None),
        getattr(model, "embed_tokens", None),
    ]
    for mod in candidates:
        if mod is not None:
            # Find a tensor to read device from
            for p in mod.parameters(recurse=True):
                return p.device
            for b in mod.buffers(recurse=True):
                return b.device
    # Fallback to first layer's device
    layers = _find_layers_list(model)
    return _layer_device(layers[0])

def to_static_multi_device(model, kv_cache):
    """
    Convert a tuple[(k, v)] per layer (each [B, Hkv, L, D] or [B, L, Hkv, D]) into a StaticCache
    whose per-layer tensors are placed on the actual devices of those layers.
    Returns: static_cache, attention_mask, L
    """
    layers = _find_layers_list(model)

    # Normalize layout to [B, Hkv, L, D] if needed
    k0, v0 = kv_cache[0]
    # If shape is BLHD, transpose to BHLD
    if k0.dim() == 4 and k0.shape[1] != getattr(model.config, "num_key_value_heads", getattr(model.config, "num_attention_heads")):
        kv_cache = tuple(
            (k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous())
            for (k, v) in kv_cache
        )
        k0, v0 = kv_cache[0]

    B, Hkv, L, D = k0.shape

    # Create a temporary StaticCache on CPU (we'll replace its per-layer tensors)
    static = StaticCache(
        batch_size=B,
        max_cache_len=L,
        config=model.config,
        device=torch.device("cpu"),
        dtype=k0.dtype,
    )

    # Ensure key/value list lengths match number of layers
    num_layers = len(layers)
    if len(kv_cache) != num_layers:
        raise ValueError(f"len(kv_cache)={len(kv_cache)} does not match model layers={num_layers}")

    # Reallocate each layer's cache tensor on the correct device and copy data
    for i, (k, v) in enumerate(kv_cache):
        layer_dev = _layer_device(layers[i])

        # Allocate fresh tensors on the layer's device
        k_target = torch.empty((B, Hkv, L, D), device=layer_dev, dtype=k.dtype)
        v_target = torch.empty((B, Hkv, L, D), device=layer_dev, dtype=v.dtype)

        # Copy user-provided cache into the properly-placed tensors
        k_target.copy_(k.to(layer_dev, non_blocking=True))
        v_target.copy_(v.to(layer_dev, non_blocking=True))

        # Install into StaticCache
        static.key_cache[i] = k_target
        static.value_cache[i] = v_target

    # Mark how many tokens are valid
    first_dev = _embed_device(model)
    if hasattr(static, "cache_lengths"):
        static.cache_lengths = torch.full((B,), L, device=first_dev, dtype=torch.long)

    # Build a full-visible attention mask for the existing tokens on the first device
    attention_mask = torch.ones(B, L, dtype=torch.long, device=first_dev)

    return static, attention_mask, L


def legacy_to_static(
    model, 
    legacy_cache,
    extend_cache: bool = False,
    additional_tokens: int = 0
):
    """
    Convert legacy cache tuple to StaticCache using proper API methods.
    
    Args:
        model: The transformer model
        legacy_cache: Tuple of (key, value) pairs per layer
        extend_cache: Whether to allocate extra space for future tokens
        additional_tokens: Extra tokens to allocate if extend_cache is True
    
    Returns:
        static_cache: StaticCache object
        attention_mask: Attention mask tensor  
        cache_length: Current sequence length
    """
    # Validate input
    if not legacy_cache or len(legacy_cache) == 0:
        raise ValueError("Legacy cache is empty")
    
    # Get dimensions from first layer
    k0, v0 = legacy_cache[0]
    B, num_heads, seq_len, head_dim = k0.shape
    device = k0.device
    dtype = k0.dtype
    
    # Determine max cache length
    max_cache_len = seq_len + additional_tokens if extend_cache else seq_len
    
    # Create StaticCache with appropriate size
    static_cache = StaticCache(
        config=model.config,
        max_batch_size=B,
        max_cache_len=max_cache_len,
        device=device,
        dtype=dtype
    )
    
    # Create cache position indices for the current sequence
    cache_position = torch.arange(seq_len, device=device)
    
    # Update static cache using the proper API
    for layer_idx, (key_states, value_states) in enumerate(legacy_cache):
        # Ensure correct shape
        if key_states.shape[2] != seq_len:
            actual_len = key_states.shape[2]
            cache_position_layer = torch.arange(actual_len, device=device)
        else:
            cache_position_layer = cache_position
        
        # Use the update method to properly set cache values
        static_cache.update(
            key_states=key_states,
            value_states=value_states,
            layer_idx=layer_idx,
            cache_kwargs={'cache_position': cache_position_layer}
        )
    
    # Create attention mask for current sequence
    attention_mask = torch.ones(B, seq_len, dtype=torch.long, device=device)
    
    return static_cache,attention_mask,seq_len

def legacy_to_static_multi_device(
    model,
    legacy_cache,
    extend_cache: bool = False,
    additional_tokens: int = 0,
):
    """
    Convert a legacy past_key_values tuple into a StaticCache across multiple devices.
    Requires transformers >= 4.41 (StaticCache present). No DynamicCache fallback.

    Returns:
        static_cache: StaticCache
        attention_mask: (B, seq_len) torch.LongTensor (all ones)
        cache_length: int (seq_len)
    """
    # --- require StaticCache ---
    try:
        from transformers.cache_utils import StaticCache
    except Exception as e:
        raise RuntimeError(
            "StaticCache is not available in your transformers version. "
            "Please upgrade to transformers>=4.41."
        ) from e

    if not legacy_cache or len(legacy_cache) == 0:
        raise ValueError("legacy_cache is empty")

    # Shapes from first layer
    k0, v0 = legacy_cache[0]
    if k0.ndim != 4:
        raise ValueError(f"Expected key shape (B, num_kv_heads, seq_len, head_dim), got {k0.shape}")
    B, _, seq_len, _ = k0.shape
    dtype = k0.dtype

    # Decide max cache length (preallocation)
    max_cache_len = seq_len + (additional_tokens if extend_cache else 0)

    # --- derive per-layer device map ---
    num_layers = getattr(getattr(model, "config", object()), "num_hidden_layers", len(legacy_cache))
    device_map = getattr(model, "hf_device_map", None)
    per_layer_device = {}

    if device_map:
        # Try common prefix "model.layers.{i}" ; adjust if your model uses another prefix.
        for i in range(num_layers):
            key = f"model.layers.{i}"
            dev = device_map.get(key, None)
            per_layer_device[i] = torch.device(dev) if dev is not None else legacy_cache[i][0].device
    else:
        for i in range(num_layers):
            per_layer_device[i] = legacy_cache[i][0].device

    # Anchor device for the StaticCache object itself (buffers are placed per update() call)
    anchor_device ="cpu" #per_layer_device.get(0, k0.device)

    # Create the StaticCache
    static_cache = StaticCache(
        config=model.config,
        max_batch_size=B,
        max_cache_len=max_cache_len,
        device=anchor_device,
        dtype=dtype,
    )

    # Fill layers; support uneven cached lengths per layer (defensive)
    for layer_idx, (key_states, value_states) in enumerate(legacy_cache):
        L = key_states.shape[2]
        target_device = per_layer_device[layer_idx]

        if key_states.device != target_device:
            key_states = key_states.to(target_device, non_blocking=True)
            value_states = value_states.to(target_device, non_blocking=True)

        cache_position = torch.arange(L, device=target_device)
        static_cache.update(
            key_states=key_states,
            value_states=value_states,
            layer_idx=layer_idx,
            cache_kwargs={"cache_position": cache_position},
        )

    # Attention mask for the current cached sequence (extend yourself if you append tokens)
    attention_mask = torch.ones(B, seq_len, dtype=torch.long, device=anchor_device)
    return static_cache, attention_mask, seq_len

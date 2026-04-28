"""
This module incorporates code from the AT2 codebase.
"""

import math
from typing import Any, Optional
import torch as ch
import transformers.models


def infer_model_type(model):
    if "deepseek" in model.name_or_path.lower() or "mistral" in model.name_or_path.lower():
        return "llama"

    model_type_to_keyword = {
        "llama": "llama",
        "qwen2": "qwen",
    }
    for model_type, keyword in model_type_to_keyword.items():
        if keyword in model.name_or_path.lower():
            return model_type
    else:
        raise ValueError(f"Unknown model: {model.name_or_path}. Specify `model_type`.")


def get_helpers(model_type):
    if not hasattr(transformers.models, model_type):
        raise ValueError(f"Unknown model: {model_type}")
    model_module = getattr(transformers.models, model_type)
    modeling_module = getattr(model_module, f"modeling_{model_type}")
    return modeling_module.apply_rotary_pos_emb, modeling_module.repeat_kv


def get_position_ids_and_attention_mask(model, hidden_states):
    input_embeds = hidden_states[0]
    _, seq_len, _ = input_embeds.shape
    position_ids = ch.arange(0, seq_len, device=model.device).unsqueeze(0)
    attention_mask = ch.ones(
        seq_len, seq_len + 1, device=model.device, dtype=model.dtype
    )
    attention_mask = ch.triu(attention_mask, diagonal=1)
    attention_mask *= ch.finfo(model.dtype).min
    attention_mask = attention_mask[None, None]
    return position_ids, attention_mask


def get_attentions_shape(model):
    num_layers = len(model.model.layers)
    num_heads = model.model.config.num_attention_heads
    return num_layers, num_heads

def get_layer_attention_weights(
    model,
    hidden_states,
    layer_index,
    position_ids,
    attention_mask,
    attribution_start=None,
    attribution_end=None,
    model_type=None,
):
    model_type = model_type or infer_model_type(model)
    assert layer_index >= 0 and layer_index < len(model.model.layers)
    layer = model.model.layers[layer_index]
    self_attn = layer.self_attn
    hidden_states = hidden_states[layer_index]
    hidden_states = layer.input_layernorm(hidden_states)
    bsz, q_len, _ = hidden_states.size()

    num_attention_heads = model.model.config.num_attention_heads
    num_key_value_heads = model.model.config.num_key_value_heads
    head_dim = self_attn.head_dim
    if model_type in ("llama", "qwen2", "qwen1.5"):
        query_states = self_attn.q_proj(hidden_states)
        key_states = self_attn.k_proj(hidden_states)
    else:
        raise ValueError(f"Unknown model: {model.name_or_path}")


    query_states = query_states.view(bsz, q_len, num_attention_heads, head_dim)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim)
    key_states = key_states.transpose(1, 2)
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)

    cos, sin = position_embeddings

    apply_rotary_pos_emb, repeat_kv = get_helpers(model_type)
    target_device = query_states.device
    if (
        key_states.device != target_device
        or cos.device != target_device
        or sin.device != target_device
    ):
        key_states = key_states.to(target_device)
        cos = cos.to(target_device)
        sin = sin.to(target_device)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    key_states = repeat_kv(key_states, self_attn.num_key_value_groups)

    causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
    attribution_start = attribution_start if attribution_start is not None else 1
    attribution_end = attribution_end if attribution_end is not None else q_len + 1

    causal_mask = causal_mask[:, :, attribution_start - 1 : attribution_end - 1]
    if causal_mask.device != target_device:
        causal_mask = causal_mask.to(target_device)
    query_states = query_states[:, :, attribution_start - 1 : attribution_end - 1]

    attn_weights = ch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        head_dim
    )
    attn_weights = attn_weights + causal_mask
    dtype = attn_weights.dtype
    attn_weights = ch.softmax(attn_weights, dim=-1, dtype=ch.float32).to(dtype)
    return attn_weights


def get_attention_weights_one_layer(
    model: Any,
    hidden_states: Any,
    layer_index: int,
    attribution_start: Optional[int] = None,
    attribution_end: Optional[int] = None,
    model_type: Optional[str] = None,
    reverse: Optional[bool] = False,
) -> Any:

    with ch.no_grad():
        position_ids, attention_mask = get_position_ids_and_attention_mask(
            model, hidden_states
        )
        num_layers, num_heads = get_attentions_shape(model)
        num_tokens = hidden_states[0].shape[1] + 1
        attribution_start = attribution_start if attribution_start is not None else 1
        attribution_end = attribution_end if attribution_end is not None else num_tokens
        num_target_tokens = attribution_end - attribution_start
        weights = ch.zeros(
            num_layers,
            num_heads,
            num_target_tokens,
            num_tokens - 1,
            device=model.device,
            dtype=model.dtype,
        )
        if not reverse:
            weights = get_layer_attention_weights(
                model,
                hidden_states,
                layer_index,
                position_ids,
                attention_mask,
                attribution_start=attribution_start,
                attribution_end=attribution_end,
                model_type=model_type,
            )
        else:
            weights = get_layer_attention_weights_reverse(
                model,
                hidden_states,
                layer_index,
                position_ids,
                attention_mask,
                attribution_start=attribution_start,
                attribution_end=attribution_end,
                model_type=model_type,
            )

    return weights


def get_hidden_states_one_layer(
    model: Any,
    hidden_states: Any,
    layer_index: int,
    attribution_start: Optional[int] = None,
    attribution_end: Optional[int] = None,
    model_type: Optional[str] = None,
) -> Any:
    def get_hidden_states(
        model,
        hidden_states,
        layer_index,
        position_ids,
        attention_mask,
        attribution_start=None,
        attribution_end=None,
        model_type=None,
        ):
        model_type = model_type or infer_model_type(model)
        assert layer_index >= 0 and layer_index < len(model.model.layers)
        layer = model.model.layers[layer_index]
        self_attn = layer.self_attn
        hidden_states = hidden_states[layer_index]
        hidden_states = layer.input_layernorm(hidden_states)
        bsz, q_len, _ = hidden_states.size()

        num_attention_heads = model.model.config.num_attention_heads
        num_key_value_heads = model.model.config.num_key_value_heads
        head_dim = self_attn.head_dim

        if model_type in ("llama", "qwen2", "qwen1.5"):
            query_states = self_attn.q_proj(hidden_states)
            key_states = self_attn.k_proj(hidden_states)
        else:
            raise ValueError(f"Unknown model: {model.name_or_path}")

        query_states = query_states.view(bsz, q_len, num_attention_heads, head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).mean(dim=(0, 2))
        return key_states

    with ch.no_grad():
        position_ids, attention_mask = get_position_ids_and_attention_mask(
            model, hidden_states
        )
        num_layers, num_heads = get_attentions_shape(model)
        num_tokens = hidden_states[0].shape[1] + 1
        attribution_start = attribution_start if attribution_start is not None else 1
        attribution_end = attribution_end if attribution_end is not None else num_tokens
        num_target_tokens = attribution_end - attribution_start
        weights = ch.zeros(
            num_layers,
            num_heads,
            num_target_tokens,
            num_tokens - 1,
            device=model.device,
            dtype=model.dtype,
        )

        hidden_states = get_hidden_states(
            model,
            hidden_states,
            layer_index,
            position_ids,
            attention_mask,
            attribution_start=attribution_start,
            attribution_end=attribution_end,
            model_type=model_type,
        )
 

    return hidden_states
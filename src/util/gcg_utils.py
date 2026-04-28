import torch
from src.util.string_utils import *
from src.util.utils import *
import random

from src.util.kv_cache_utils import *
def init_candidate_set(model,adv):
    max_token_value = model.tokenizer.vocab_size
    prefix_tokens = model.tokenizer.encode(adv[0], add_special_tokens=False)
    suffix_tokens = model.tokenizer.encode(adv[1], add_special_tokens=False)
    return ([[i for i in range(0, max_token_value)] for _ in range(len(prefix_tokens))],[[i for i in range(0, max_token_value)] for _ in range(len(prefix_tokens))])



def get_candidate_set(model,context_left,context_right,payload,query, adv, target_answer, k=128,context_remove_ratio = 0.8):

    context_left_sentences = contexts_to_sentences([context_left])
    context_right_sentences = contexts_to_sentences([context_right])
    num_sentences_left = len(context_left_sentences)
    num_sentences_to_remove_left = int(num_sentences_left * context_remove_ratio)
    num_sentences_right = len(context_right_sentences)
    num_sentences_to_remove_right = int(num_sentences_right * context_remove_ratio)
    
    # Randomly select sentences to remove
    indices_to_remove = random.sample(range(num_sentences_left), min(num_sentences_to_remove_left, num_sentences_left))
    context_left_sentences = [sent for i, sent in enumerate(context_left_sentences) if i not in indices_to_remove]
    context_left = ''.join(context_left_sentences)
    indices_to_remove = random.sample(range(num_sentences_right), min(num_sentences_to_remove_right, num_sentences_right))
    context_right_sentences = [sent for i, sent in enumerate(context_right_sentences) if i not in indices_to_remove]
    context_right = ''.join(context_right_sentences)

    suffix_manager = SuffixManager(model,context_left,context_right,query, adv, payload, target_answer)
    
    prompt_with_target_ids = torch.tensor(suffix_manager.get_prompt_with_target_ids()).to(model.model.device)

    # Find the adversarial slice - tokens that appear in new_msg but not orig_msg
    prefix_slice = suffix_manager.prefix_slice
    suffix_slice = suffix_manager.suffix_slice

    target_token_id = suffix_manager.target_ids
    log_p_slice = suffix_manager.log_p_slice
    
    embed_weights = model.model.get_input_embeddings().weight
    prefix_one_hot = torch.zeros(
        prompt_with_target_ids[prefix_slice].shape[0],
        embed_weights.shape[0],
        device=model.model.device,
        dtype=embed_weights.dtype
    )
    suffix_one_hot = torch.zeros(
        prompt_with_target_ids[prefix_slice].shape[0],
        embed_weights.shape[0],
        device=model.model.device,
        dtype=embed_weights.dtype
    )
    prefix_one_hot.scatter_(
        1, 
        prompt_with_target_ids[prefix_slice].unsqueeze(1),
        torch.ones(prefix_one_hot.shape[0], 1, device=model.model.device, dtype=embed_weights.dtype)
    )

    suffix_one_hot.scatter_(
        1, 
        prompt_with_target_ids[suffix_slice].unsqueeze(1),
        torch.ones(suffix_one_hot.shape[0], 1, device=model.model.device, dtype=embed_weights.dtype)
    )
    prefix_one_hot.requires_grad_()
    suffix_one_hot.requires_grad_()

    prefix_input_embeds = (prefix_one_hot @ embed_weights).unsqueeze(0)
    suffix_input_embeds = (suffix_one_hot @ embed_weights).unsqueeze(0)
    
    # now stitch it together with the rest of the embeddings
    embeds = model.model.get_input_embeddings()(prompt_with_target_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:,:prefix_slice.start,:], 
            prefix_input_embeds, 
            embeds[:,prefix_slice.stop:suffix_slice.start,:], 
            suffix_input_embeds, 
            embeds[:,suffix_slice.stop:,:]
        ], 
        dim=1)

    logits = model.model(inputs_embeds=full_embeds).logits
    
    # Calculate loss based on target_token_ids and log_p_slice
    total_loss = 0
    for i in range(len(target_token_id)):
        position = log_p_slice.start + i
        next_token_logits = logits[0, position, :]
        log_probs = torch.log_softmax(next_token_logits, dim=-1)
        token_loss = log_probs[target_token_id[i]]
        total_loss += token_loss
    
    loss = total_loss
    prefix_grad, suffix_grad = torch.autograd.grad(loss, [prefix_one_hot, suffix_one_hot])
    
    prefix_grad = prefix_grad / prefix_grad.norm(dim=-1, keepdim=True)
    prefix_top_k_tokens = torch.topk(prefix_grad, k, dim=-1).indices.tolist()
    
    suffix_grad = suffix_grad / suffix_grad.norm(dim=-1, keepdim=True)
    suffix_top_k_tokens = torch.topk(suffix_grad, k, dim=-1).indices.tolist()

    return prefix_top_k_tokens, suffix_top_k_tokens

def get_candidate_set_segment(model,context_left,context_right,payload,query, adv, target_answer, k=128,context_remove_ratio = 0.8, segment_size = 50):

    # Remove 80% of text segments (each segment is 50 tokens)
    from transformers import AutoTokenizer

    def segment_text_by_tokens(text, tokenizer, segment_size=50):
        input_ids = tokenizer.encode(text, add_special_tokens=False)
        segments = []
        for i in range(0, len(input_ids), segment_size):
            segment_ids = input_ids[i:i+segment_size]
            segment_text = tokenizer.decode(segment_ids, clean_up_tokenization_spaces=True)
            segments.append(segment_text)
        return segments

    # You may want to make sure your model/model_name has tokenizer property
    # Otherwise, pass tokenizer as an argument
    tokenizer = model.tokenizer

    # Process left context
    context_left_segments = segment_text_by_tokens(context_left, tokenizer, segment_size=segment_size)
    num_segments_left = len(context_left_segments)
    num_segments_to_remove_left = int(num_segments_left * context_remove_ratio)
    indices_to_remove_left = random.sample(range(num_segments_left), min(num_segments_to_remove_left, num_segments_left))
    context_left_segments_kept = [seg for i, seg in enumerate(context_left_segments) if i not in indices_to_remove_left]
    context_left = ''.join(context_left_segments_kept)

    # Process right context
    context_right_segments = segment_text_by_tokens(context_right, tokenizer, segment_size=segment_size)
    num_segments_right = len(context_right_segments)
    num_segments_to_remove_right = int(num_segments_right * context_remove_ratio)
    indices_to_remove_right = random.sample(range(num_segments_right), min(num_segments_to_remove_right, num_segments_right))
    context_right_segments_kept = [seg for i, seg in enumerate(context_right_segments) if i not in indices_to_remove_right]
    context_right = ''.join(context_right_segments_kept)
    #print(f'context_right: {context_right}')

    #kv_cache,_ = initialize_kv_cache(model,context_left,context_right,payload,query, adv, target_answer)
    suffix_manager = SuffixManager(model,context_left,context_right,query, adv, payload, target_answer)
    
    prompt_with_target_ids = torch.tensor(suffix_manager.get_prompt_with_target_ids()).to(model.model.device)

    # Find the adversarial slice - tokens that appear in new_msg but not orig_msg
    prefix_slice = suffix_manager.prefix_slice
    suffix_slice = suffix_manager.suffix_slice

    target_token_id = suffix_manager.target_ids
    log_p_slice = suffix_manager.log_p_slice
    
    embed_weights = model.model.get_input_embeddings().weight
    prefix_one_hot = torch.zeros(
        prompt_with_target_ids[prefix_slice].shape[0],
        embed_weights.shape[0],
        device=model.model.device,
        dtype=embed_weights.dtype
    )
    suffix_one_hot = torch.zeros(
        prompt_with_target_ids[prefix_slice].shape[0],
        embed_weights.shape[0],
        device=model.model.device,
        dtype=embed_weights.dtype
    )
    prefix_one_hot.scatter_(
        1, 
        prompt_with_target_ids[prefix_slice].unsqueeze(1),
        torch.ones(prefix_one_hot.shape[0], 1, device=model.model.device, dtype=embed_weights.dtype)
    )

    suffix_one_hot.scatter_(
        1, 
        prompt_with_target_ids[suffix_slice].unsqueeze(1),
        torch.ones(suffix_one_hot.shape[0], 1, device=model.model.device, dtype=embed_weights.dtype)
    )
    prefix_one_hot.requires_grad_()
    suffix_one_hot.requires_grad_()

    prefix_input_embeds = (prefix_one_hot @ embed_weights).unsqueeze(0)
    suffix_input_embeds = (suffix_one_hot @ embed_weights).unsqueeze(0)
    
    # now stitch it together with the rest of the embeddings
    embeds = model.model.get_input_embeddings()(prompt_with_target_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:,:prefix_slice.start,:], 
            prefix_input_embeds, 
            embeds[:,prefix_slice.stop:suffix_slice.start,:], 
            suffix_input_embeds, 
            embeds[:,suffix_slice.stop:,:]
        ], 
        dim=1)

    logits = model.model(inputs_embeds=full_embeds).logits
    
    # Calculate loss based on target_token_ids and log_p_slice
    total_loss = 0
    for i in range(len(target_token_id)):
        position = log_p_slice.start + i
        next_token_logits = logits[0, position, :]
        log_probs = torch.log_softmax(next_token_logits, dim=-1)
        token_loss = log_probs[target_token_id[i]]
        total_loss += token_loss
    
    loss = total_loss
    prefix_grad, suffix_grad = torch.autograd.grad(loss, [prefix_one_hot, suffix_one_hot])
    
    prefix_grad = prefix_grad / prefix_grad.norm(dim=-1, keepdim=True)
    prefix_top_k_tokens = torch.topk(prefix_grad, k, dim=-1).indices.tolist()
    
    suffix_grad = suffix_grad / suffix_grad.norm(dim=-1, keepdim=True)
    suffix_top_k_tokens = torch.topk(suffix_grad, k, dim=-1).indices.tolist()

    return prefix_top_k_tokens, suffix_top_k_tokens

def get_candidate_set_segment_default(model,context_left,context_right,payload,query, adv, target_answer, k=128,context_remove_ratio = 0.8, segment_size = 50):

    from transformers import AutoTokenizer

    def segment_text_by_tokens(text, tokenizer, segment_size=50):
        input_ids = tokenizer.encode(text, add_special_tokens=False)
        segments = []
        for i in range(0, len(input_ids), segment_size):
            segment_ids = input_ids[i:i+segment_size]
            #segment_text = tokenizer.decode(segment_ids, clean_up_tokenization_spaces=True)
            segments.append(segment_ids)
        return segments

    # You may want to make sure your model/model_name has tokenizer property
    # Otherwise, pass tokenizer as an argument
    tokenizer = model.tokenizer

    # Process left context
    context_left_segments = segment_text_by_tokens(context_left, tokenizer, segment_size=segment_size)
    num_segments_left = len(context_left_segments)
    num_segments_to_remove_left = int(num_segments_left * context_remove_ratio)
    indices_to_remove_left = random.sample(range(num_segments_left), min(num_segments_to_remove_left, num_segments_left))
    context_left_segments_kept = [seg for i, seg in enumerate(context_left_segments) if i not in indices_to_remove_left]
    context_left = [token_id for segment in context_left_segments_kept for token_id in segment]
    context_left = tokenizer.decode(context_left, clean_up_tokenization_spaces=True)

    # Process right context
    context_right_segments = segment_text_by_tokens(context_right, tokenizer, segment_size=segment_size)
    num_segments_right = len(context_right_segments)
    num_segments_to_remove_right = int(num_segments_right * context_remove_ratio)
    indices_to_remove_right = random.sample(range(num_segments_right), min(num_segments_to_remove_right, num_segments_right))
    context_right_segments_kept = [seg for i, seg in enumerate(context_right_segments) if i not in indices_to_remove_right]
    context_right = [token_id for segment in context_right_segments_kept for token_id in segment]
    context_right = tokenizer.decode(context_right, clean_up_tokenization_spaces=True)

    suffix_manager = SuffixManager(model,context_left,context_right,query, adv, payload, target_answer)
    
    prompt_with_target_ids = torch.tensor(suffix_manager.get_prompt_with_target_ids()).to(model.model.device)

    # Find the adversarial slice - tokens that appear in new_msg but not orig_msg
    prefix_slice = suffix_manager.prefix_slice
    suffix_slice = suffix_manager.suffix_slice

    target_token_id = suffix_manager.target_ids
    log_p_slice = suffix_manager.log_p_slice
    
    embed_weights = model.model.get_input_embeddings().weight
    prefix_one_hot = torch.zeros(
        prompt_with_target_ids[prefix_slice].shape[0],
        embed_weights.shape[0],
        device=model.model.device,
        dtype=embed_weights.dtype
    )
    suffix_one_hot = torch.zeros(
        prompt_with_target_ids[prefix_slice].shape[0],
        embed_weights.shape[0],
        device=model.model.device,
        dtype=embed_weights.dtype
    )
    prefix_one_hot.scatter_(
        1, 
        prompt_with_target_ids[prefix_slice].unsqueeze(1),
        torch.ones(prefix_one_hot.shape[0], 1, device=model.model.device, dtype=embed_weights.dtype)
    )

    suffix_one_hot.scatter_(
        1, 
        prompt_with_target_ids[suffix_slice].unsqueeze(1),
        torch.ones(suffix_one_hot.shape[0], 1, device=model.model.device, dtype=embed_weights.dtype)
    )
    prefix_one_hot.requires_grad_()
    suffix_one_hot.requires_grad_()

    prefix_input_embeds = (prefix_one_hot @ embed_weights).unsqueeze(0)
    suffix_input_embeds = (suffix_one_hot @ embed_weights).unsqueeze(0)
    
    # now stitch it together with the rest of the embeddings
    embeds = model.model.get_input_embeddings()(prompt_with_target_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:,:prefix_slice.start,:], 
            prefix_input_embeds, 
            embeds[:,prefix_slice.stop:suffix_slice.start,:], 
            suffix_input_embeds, 
            embeds[:,suffix_slice.stop:,:]
        ], 
        dim=1)

    logits = model.model(inputs_embeds=full_embeds).logits
    
    # Calculate loss based on target_token_ids and log_p_slice
    total_loss = 0
    for i in range(len(target_token_id)):
        position = log_p_slice.start + i
        next_token_logits = logits[0, position, :]
        log_probs = torch.log_softmax(next_token_logits, dim=-1)
        token_loss = log_probs[target_token_id[i]]
        total_loss += token_loss
    
    loss = total_loss
    prefix_grad, suffix_grad = torch.autograd.grad(loss, [prefix_one_hot, suffix_one_hot])
    
    prefix_grad = prefix_grad / prefix_grad.norm(dim=-1, keepdim=True)
    prefix_top_k_tokens = torch.topk(prefix_grad, k, dim=-1).indices.tolist()
    
    suffix_grad = suffix_grad / suffix_grad.norm(dim=-1, keepdim=True)
    suffix_top_k_tokens = torch.topk(suffix_grad, k, dim=-1).indices.tolist()

    return prefix_top_k_tokens, suffix_top_k_tokens

def get_candidate_set_attention(model,clean_text,query, adv, target_answer,position, important_positions, k=128,context_remove_ratio = 0.8):
    
    _,context_left,context_right, payload = insert_malicious_instruction(clean_text,adv,target_answer,position)

    suffix_manager = SuffixManager(model,context_left,context_right,query, adv, payload, target_answer)
    context_left_filtered_positions = [i for i in important_positions if i >= suffix_manager.context_left_slice.stop and i < suffix_manager.context_left_slice.start]
    context_left_filtered_positions = sorted(context_left_filtered_positions)
    context_left_filtered_positions = [i for i in important_positions if i >= suffix_manager.context_right_slice.stop and i < suffix_manager.context_right_slice.start]
    context_right_filtered_positions = sorted(context_right_filtered_positions)
    new_context_left_ids = [suffix_manager.prompt_ids[i] for i in context_left_filtered_positions]
    new_context_right_ids = [suffix_manager.prompt_ids[i] for i in context_right_filtered_positions]
    suffix_manager.update_context_ids(new_context_left_ids,new_context_right_ids)
    prompt_with_target_ids = torch.tensor(suffix_manager.get_prompt_with_target_ids()).to(model.model.device)

    # Find the adversarial slice - tokens that appear in new_msg but not orig_msg
    prefix_slice = suffix_manager.prefix_slice
    suffix_slice = suffix_manager.suffix_slice

    target_token_id = suffix_manager.target_ids
    log_p_slice = suffix_manager.log_p_slice
    
    embed_weights = model.model.get_input_embeddings().weight
    prefix_one_hot = torch.zeros(
        prompt_with_target_ids[prefix_slice].shape[0],
        embed_weights.shape[0],
        device=model.model.device,
        dtype=embed_weights.dtype
    )
    suffix_one_hot = torch.zeros(
        prompt_with_target_ids[prefix_slice].shape[0],
        embed_weights.shape[0],
        device=model.model.device,
        dtype=embed_weights.dtype
    )
    prefix_one_hot.scatter_(
        1, 
        prompt_with_target_ids[prefix_slice].unsqueeze(1),
        torch.ones(prefix_one_hot.shape[0], 1, device=model.model.device, dtype=embed_weights.dtype)
    )

    suffix_one_hot.scatter_(
        1, 
        prompt_with_target_ids[suffix_slice].unsqueeze(1),
        torch.ones(suffix_one_hot.shape[0], 1, device=model.model.device, dtype=embed_weights.dtype)
    )
    prefix_one_hot.requires_grad_()
    suffix_one_hot.requires_grad_()

    prefix_input_embeds = (prefix_one_hot @ embed_weights).unsqueeze(0)
    suffix_input_embeds = (suffix_one_hot @ embed_weights).unsqueeze(0)
    
    # now stitch it together with the rest of the embeddings
    embeds = model.model.get_input_embeddings()(prompt_with_target_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:,:prefix_slice.start,:], 
            prefix_input_embeds, 
            embeds[:,prefix_slice.stop:suffix_slice.start,:], 
            suffix_input_embeds, 
            embeds[:,suffix_slice.stop:,:]
        ], 
        dim=1)

    logits = model.model(inputs_embeds=full_embeds).logits
    
    # Calculate loss based on target_token_ids and log_p_slice
    total_loss = 0
    for i in range(len(target_token_id)):
        position = log_p_slice.start + i
        next_token_logits = logits[0, position, :]
        log_probs = torch.log_softmax(next_token_logits, dim=-1)
        token_loss = log_probs[target_token_id[i]]
        total_loss += token_loss
    
    loss = total_loss

    prefix_grad, suffix_grad = torch.autograd.grad(loss, [prefix_one_hot, suffix_one_hot])
    
    prefix_grad = prefix_grad / prefix_grad.norm(dim=-1, keepdim=True)
    prefix_top_k_tokens = torch.topk(prefix_grad, k, dim=-1).indices.tolist()
    
    suffix_grad = suffix_grad / suffix_grad.norm(dim=-1, keepdim=True)
    suffix_top_k_tokens = torch.topk(suffix_grad, k, dim=-1).indices.tolist()

    return prefix_top_k_tokens, suffix_top_k_tokens

def get_candidate_set_counterfactual(model,clean_text,query, adv, target_answer,position, k=128,context_remove_ratio = 0.8, variant = "simple"):
    
    _,context_left,context_right, payload = insert_malicious_instruction(clean_text,adv,target_answer,position)

    # Store original unclipped contexts for counterfactual positional encodings
    original_context_left = context_left
    original_context_right = context_right

    context_left_sentences = contexts_to_sentences([context_left])
    context_right_sentences = contexts_to_sentences([context_right])
    num_sentences_left = len(context_left_sentences)
    num_sentences_to_remove_left = int(num_sentences_left * context_remove_ratio)
    num_sentences_right = len(context_right_sentences)
    num_sentences_to_remove_right = int(num_sentences_right * context_remove_ratio)
    
    # Randomly select sentences to remove
    indices_to_remove = random.sample(range(num_sentences_left), min(num_sentences_to_remove_left, num_sentences_left))
    context_left_sentences = [sent for i, sent in enumerate(context_left_sentences) if i not in indices_to_remove]
    context_left = ''.join(context_left_sentences)
    indices_to_remove = random.sample(range(num_sentences_right), min(num_sentences_to_remove_right, num_sentences_right))
    context_right_sentences = [sent for i, sent in enumerate(context_right_sentences) if i not in indices_to_remove]
    context_right = ''.join(context_right_sentences)

    # Create suffix managers for both clipped and unclipped contexts
    suffix_manager = SuffixManager(model,context_left,context_right,query, adv, payload, target_answer)
    counterfactual_suffix_manager = SuffixManager(model, original_context_left, original_context_right, query, adv, payload, target_answer)
    my_position_mapping = make_mapping(suffix_manager, counterfactual_suffix_manager, variant = variant)
    token = apply_positional_encoding_hook(model.model, my_position_mapping)

# ... run the code region you want to affect ...


    prompt_with_target_ids = torch.tensor(suffix_manager.get_prompt_with_target_ids()).to(model.model.device)
    counterfactual_prompt_with_target_ids = torch.tensor(counterfactual_suffix_manager.get_prompt_with_target_ids()).to(model.model.device)

    # Find the adversarial slice - tokens that appear in new_msg but not orig_msg
    prefix_slice = suffix_manager.prefix_slice
    suffix_slice = suffix_manager.suffix_slice

    target_token_id = suffix_manager.target_ids
    log_p_slice = suffix_manager.log_p_slice
    
    # Create position mapping for counterfactual positional encoding
    original_seq_len = len(prompt_with_target_ids)
    counterfactual_seq_len = len(counterfactual_prompt_with_target_ids)

    
    embed_weights = model.model.get_input_embeddings().weight
    prefix_one_hot = torch.zeros(
        prompt_with_target_ids[prefix_slice].shape[0],
        embed_weights.shape[0],
        device=model.model.device,
        dtype=embed_weights.dtype
    )
    suffix_one_hot = torch.zeros(
        prompt_with_target_ids[prefix_slice].shape[0],
        embed_weights.shape[0],
        device=model.model.device,
        dtype=embed_weights.dtype
    )
    prefix_one_hot.scatter_(
        1, 
        prompt_with_target_ids[prefix_slice].unsqueeze(1),
        torch.ones(prefix_one_hot.shape[0], 1, device=model.model.device, dtype=embed_weights.dtype)
    )

    suffix_one_hot.scatter_(
        1, 
        prompt_with_target_ids[suffix_slice].unsqueeze(1),
        torch.ones(suffix_one_hot.shape[0], 1, device=model.model.device, dtype=embed_weights.dtype)
    )
    prefix_one_hot.requires_grad_()
    suffix_one_hot.requires_grad_()

    prefix_input_embeds = (prefix_one_hot @ embed_weights).unsqueeze(0)
    suffix_input_embeds = (suffix_one_hot @ embed_weights).unsqueeze(0)
    
    # now stitch it together with the rest of the embeddings
    embeds = model.model.get_input_embeddings()(prompt_with_target_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:,:prefix_slice.start,:], 
            prefix_input_embeds, 
            embeds[:,prefix_slice.stop:suffix_slice.start,:], 
            suffix_input_embeds, 
            embeds[:,suffix_slice.stop:,:]
        ], 
        dim=1)

    logits = model.model(inputs_embeds=full_embeds).logits
    
    # Calculate loss based on target_token_ids and log_p_slice
    total_loss = 0
    for i in range(len(target_token_id)):
        position = log_p_slice.start + i
        next_token_logits = logits[0, position, :]
        log_probs = torch.log_softmax(next_token_logits, dim=-1)
        token_loss = log_probs[target_token_id[i]]
        total_loss += token_loss
    
    loss = total_loss

    prefix_grad, suffix_grad = torch.autograd.grad(loss, [prefix_one_hot, suffix_one_hot])
    
    prefix_grad = prefix_grad / prefix_grad.norm(dim=-1, keepdim=True)
    prefix_top_k_tokens = torch.topk(prefix_grad, k, dim=-1).indices.tolist()
    
    suffix_grad = suffix_grad / suffix_grad.norm(dim=-1, keepdim=True)
    suffix_top_k_tokens = torch.topk(suffix_grad, k, dim=-1).indices.tolist()

    restore_positional_encoding_hook(model.model, token)
    return prefix_top_k_tokens, suffix_top_k_tokens

def get_candidate_set_vanilla(model,context_left,context_right,payload,query, adv, target_answer, k=128):

    suffix_manager = SuffixManager(model,context_left,context_right,query, adv, payload, target_answer)
    
    prompt_with_target_ids = torch.tensor(suffix_manager.get_prompt_with_target_ids()).to(model.model.device)

    # Find the adversarial slice - tokens that appear in new_msg but not orig_msg
    prefix_slice = suffix_manager.prefix_slice
    suffix_slice = suffix_manager.suffix_slice

    target_token_id = suffix_manager.target_ids
    log_p_slice = suffix_manager.log_p_slice
    
    embed_weights = model.model.get_input_embeddings().weight
    prefix_one_hot = torch.zeros(
        prompt_with_target_ids[prefix_slice].shape[0],
        embed_weights.shape[0],
        device=model.model.device,
        dtype=embed_weights.dtype
    )
    suffix_one_hot = torch.zeros(
        prompt_with_target_ids[prefix_slice].shape[0],
        embed_weights.shape[0],
        device=model.model.device,
        dtype=embed_weights.dtype
    )
    prefix_one_hot.scatter_(
        1, 
        prompt_with_target_ids[prefix_slice].unsqueeze(1),
        torch.ones(prefix_one_hot.shape[0], 1, device=model.model.device, dtype=embed_weights.dtype)
    )

    suffix_one_hot.scatter_(
        1, 
        prompt_with_target_ids[suffix_slice].unsqueeze(1),
        torch.ones(suffix_one_hot.shape[0], 1, device=model.model.device, dtype=embed_weights.dtype)
    )
    prefix_one_hot.requires_grad_()
    suffix_one_hot.requires_grad_()

    prefix_input_embeds = (prefix_one_hot @ embed_weights).unsqueeze(0)
    suffix_input_embeds = (suffix_one_hot @ embed_weights).unsqueeze(0)
    
    # now stitch it together with the rest of the embeddings
    embeds = model.model.get_input_embeddings()(prompt_with_target_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:,:prefix_slice.start,:], 
            prefix_input_embeds, 
            embeds[:,prefix_slice.stop:suffix_slice.start,:], 
            suffix_input_embeds, 
            embeds[:,suffix_slice.stop:,:]
        ], 
        dim=1)

    logits = model.model(inputs_embeds=full_embeds).logits
    
    # Calculate loss based on target_token_ids and log_p_slice
    total_loss = 0
    for i in range(len(target_token_id)):
        position = log_p_slice.start + i
        next_token_logits = logits[0, position, :]
        log_probs = torch.log_softmax(next_token_logits, dim=-1)
        token_loss = log_probs[target_token_id[i]]
        total_loss += token_loss
    
    loss = total_loss
    
    # Get gradients directly using grad()
    prefix_grad, suffix_grad = torch.autograd.grad(loss, [prefix_one_hot, suffix_one_hot])
    
    prefix_grad = prefix_grad / prefix_grad.norm(dim=-1, keepdim=True)
    prefix_top_k_tokens = torch.topk(prefix_grad, k, dim=-1).indices.tolist()
    
    suffix_grad = suffix_grad / suffix_grad.norm(dim=-1, keepdim=True)
    suffix_top_k_tokens = torch.topk(suffix_grad, k, dim=-1).indices.tolist()

    return prefix_top_k_tokens, suffix_top_k_tokens

#This function takes in a kv_cache and returns the top k tokens for the prefix and suffix, it is more memory efficient.
def get_candidate_set_kv(kv_cache,model,context_left,context_right,payload,query, adv, target_answer, k=128):

    suffix_manager = SuffixManager(model,context_left,context_right,query, adv, payload, target_answer)
    
    prompt_with_target_ids = torch.tensor(suffix_manager.get_prompt_with_target_ids()).to(model.model.device)

    # Find the adversarial slice - tokens that appear in new_msg but not orig_msg
    prefix_slice = suffix_manager.prefix_slice
    suffix_slice = suffix_manager.suffix_slice

    target_token_id = suffix_manager.target_ids
    log_p_slice = suffix_manager.log_p_slice
    
    embed_weights = model.model.get_input_embeddings().weight
    prefix_one_hot = torch.zeros(
        prompt_with_target_ids[prefix_slice].shape[0],
        embed_weights.shape[0],
        device=model.model.device,
        dtype=embed_weights.dtype
    )
    suffix_one_hot = torch.zeros(
        prompt_with_target_ids[prefix_slice].shape[0],
        embed_weights.shape[0],
        device=model.model.device,
        dtype=embed_weights.dtype
    )
    prefix_one_hot.scatter_(
        1, 
        prompt_with_target_ids[prefix_slice].unsqueeze(1),
        torch.ones(prefix_one_hot.shape[0], 1, device=model.model.device, dtype=embed_weights.dtype)
    )

    suffix_one_hot.scatter_(
        1, 
        prompt_with_target_ids[suffix_slice].unsqueeze(1),
        torch.ones(suffix_one_hot.shape[0], 1, device=model.model.device, dtype=embed_weights.dtype)
    )
    prefix_one_hot.requires_grad_()
    suffix_one_hot.requires_grad_()

    prefix_input_embeds = (prefix_one_hot @ embed_weights).unsqueeze(0)
    suffix_input_embeds = (suffix_one_hot @ embed_weights).unsqueeze(0)
    
    # now stitch it together with the rest of the embeddings
    embeds = model.model.get_input_embeddings()(prompt_with_target_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            prefix_input_embeds, 
            embeds[:,prefix_slice.stop:suffix_slice.start,:], 
            suffix_input_embeds, 
            embeds[:,suffix_slice.stop:,:]
        ], 
        dim=1)

    kv_cache = slice_kv_cache(kv_cache, 0,prefix_slice.start)
    logits = model.model(inputs_embeds=full_embeds,past_key_values=kv_cache, use_cache=True).logits
    
    # Calculate loss based on target_token_ids and log_p_slice
    total_loss = 0
    for i in range(len(target_token_id)):
        position = logits.shape[1] - len(target_token_id) + i-1
        next_token_logits = logits[0, position, :]
        log_probs = torch.log_softmax(next_token_logits, dim=-1)
        token_loss = log_probs[target_token_id[i]]
        total_loss += token_loss
    
    loss = total_loss
    
    # Get gradients directly using grad()
    prefix_grad, suffix_grad = torch.autograd.grad(loss, [prefix_one_hot, suffix_one_hot])
    
    prefix_grad = prefix_grad / prefix_grad.norm(dim=-1, keepdim=True)
    prefix_top_k_tokens = torch.topk(prefix_grad, k, dim=-1).indices.tolist()
    
    suffix_grad = suffix_grad / suffix_grad.norm(dim=-1, keepdim=True)
    suffix_top_k_tokens = torch.topk(suffix_grad, k, dim=-1).indices.tolist()

    return prefix_top_k_tokens, suffix_top_k_tokens

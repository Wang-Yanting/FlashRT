import os
import torch
import random
import json
from typing import Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from inspect import signature
import numpy as np
import time
from typing import Tuple, Optional
import torch.nn as nn
from src.util.kv_cache_utils import *
from src.util.string_utils import *
from src.attribution.avg_attention import *
import random
def remove_bos_token(tokenizer, token_ids):
    if token_ids[0] == tokenizer.bos_token_id:
        token_ids = token_ids[1:]
    return token_ids

def schedule_n_to_change_prob(max_n_to_change, best_logprobs):
    """ Piece-wise constant schedule for `n_to_change` based on the best prob """
    if len(best_logprobs) <100:
        return max_n_to_change
    else:
        prob_best, prob_history = best_logprobs[-1], best_logprobs[-50]
        prob = np.exp(prob_best)
        prob_change =(prob_best) - (prob_history)

    if 0 <= prob <= 0.01:
        n_to_change = max_n_to_change //random.choice([1,2])
    elif 0.01< prob <= 0.1:
        n_to_change = max_n_to_change // random.choice([2,4])
    elif 0.1 < prob <= 1.0:
        n_to_change = max_n_to_change // 4
    else:
        raise ValueError(f'Wrong prob {prob}')
    n_to_change = max(n_to_change, 1)

    return n_to_change




def early_stopping_condition(best_logprobs, logprob_first_token, model,target_answer,no_improvement_history=2000, 
                             prob_start=0.02, no_improvement_threshold_prob=0.0001):

    if len(best_logprobs) == 0:
        return False
    else:
        best_logprob = best_logprobs[-1]
    if no_improvement_history < len(best_logprobs):
        prob_best, prob_history = np.exp(best_logprobs[-1]), np.exp(best_logprobs[-no_improvement_history])
        no_sufficient_improvement_condition = prob_best - prob_history <= no_improvement_threshold_prob and prob_history-prob_best <= no_improvement_threshold_prob
    else: 
        no_sufficient_improvement_condition = False
    if no_sufficient_improvement_condition:
        print("no sufficient improvement condition")
        return True 
    target_length = len(model.tokenizer.encode(target_answer, add_special_tokens=False))
    
    threshold = 0.4
    # for all other models
    if np.exp(best_logprob) > threshold and np.exp(logprob_first_token) > 0.5:  
        print(f"early stop!, best_logprob: {best_logprob}, logprob_first_token: {logprob_first_token}")
        return True
    return False

def no_improvement_condition(best_logprobs, no_improvement_history=100, no_improvement_threshold_prob=0.0001):

    if len(best_logprobs) == 0:
        return False
    else:
        best_logprob = best_logprobs[-1]
    if no_improvement_history < len(best_logprobs):
        prob_best, prob_history = np.exp(best_logprobs[-1]), np.exp(best_logprobs[-no_improvement_history])
        no_sufficient_improvement_condition = prob_best - prob_history <= no_improvement_threshold_prob and prob_history-prob_best <= no_improvement_threshold_prob
    else: 
        no_sufficient_improvement_condition = False
    return no_sufficient_improvement_condition

def get_logprob(model,msg,target_answer):
    messages = model.messages
    if len(messages) == 1:
        messages[0]["content"] = msg
    else:
        messages[1]["content"] = msg
    if model.tokenizer.chat_template is None:
        input_ids = torch.tensor([model.tokenizer.encode(msg, add_special_tokens=True)]).to(model.model.device)
    else:
        input_ids = model.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.model.device)
    
    # Get target token ids
    target_ids = model.tokenizer.encode(target_answer, add_special_tokens=False, return_tensors="pt").to(model.model.device)
    
    # Concatenate input and target ids
    combined_ids = torch.cat([input_ids, target_ids], dim=1)
    
    with torch.no_grad():
        outputs = model.model(combined_ids)
        logits = outputs.logits
        
    # Calculate log probabilities for target tokens
    total_log_prob = 0
    first_token_log_prob = None
    for i in range(len(target_ids[0])):
        position = input_ids.shape[1] + i - 1  # Position to calculate prob for
        next_token_logits = logits[0, position, :]
        next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=0)
        target_token_prob = next_token_probs[target_ids[0,i]]
        log_prob = torch.log(target_token_prob).item()
        if i == 0:
            first_token_log_prob = log_prob
        total_log_prob += log_prob
        
    return total_log_prob, first_token_log_prob




def get_logprob_suffix_manager(model,context_left,context_right,payload,query,new_adv,target_answer):
    
    suffix_manager = SuffixManager(model,context_left,context_right,query, new_adv, payload, target_answer)
   
    combined_ids = torch.tensor([suffix_manager.get_prompt_with_target_ids()]).to(model.model.device)
    # Get model outputs
    with torch.no_grad():
        outputs = model.model(combined_ids)
        logits = outputs.logits
        
    # Calculate log probabilities for target tokens
    total_log_prob = 0
    first_token_log_prob = None
    logits_slice = suffix_manager.log_p_slice

    target_ids = suffix_manager.target_ids
    for i in range(len(target_ids)):
        position = logits_slice.start + i  # Position to calculate prob for
        next_token_logits = logits[0, position, :]
        next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=0)
        target_token_prob = next_token_probs[target_ids[i]]
        log_prob = torch.log(target_token_prob).item()
        #print(f"logprob {i}-th token: {log_prob}")
        if i == 0:
            first_token_log_prob = log_prob
        total_log_prob += log_prob
        
    return total_log_prob, first_token_log_prob


def get_logprob_cache(kv_cache,model,context_left,context_right,payload,query, new_adv, target_answer):
    
    suffix_manager = SuffixManager(model,context_left,context_right,query, new_adv, payload, target_answer)
    print("input ids1: ",suffix_manager.prompt_with_target_ids)
    assert len(suffix_manager.get_prompt_with_target_ids()) == kv_cache[0][0].shape[2]
    before_context_cache = slice_kv_cache(kv_cache, 0, suffix_manager.before_context_slice.stop)
    context_left_cache = slice_kv_cache(kv_cache, suffix_manager.context_left_slice.start, suffix_manager.context_left_slice.stop)
    
    prefix_left_cache = concat_kv_cache(context_left_cache, before_context_cache)
    malicious_instruction_ids = suffix_manager.get_prompt_with_target_ids()[suffix_manager.malicious_instruction_slice]
    malicious_instruction_ids = torch.tensor([malicious_instruction_ids]).to(model.model.device)

    with torch.no_grad():
        outputs = model.model(input_ids=malicious_instruction_ids, past_key_values=tuple(prefix_left_cache), use_cache=True)

    before_context_right_cache = outputs.past_key_values
    context_right_cache = slice_kv_cache(kv_cache, suffix_manager.context_right_slice.start, suffix_manager.context_right_slice.stop)
    before_query_cache = concat_kv_cache(before_context_right_cache, context_right_cache)
    after_context_ids = torch.tensor([suffix_manager.after_context_ids_with_target]).to(model.model.device)

    with torch.no_grad():
        outputs = model.model(input_ids=after_context_ids, past_key_values=tuple(before_query_cache), use_cache=True)
    logits = outputs.logits

    target_ids = suffix_manager.target_ids
    total_log_prob = 0
    first_token_log_prob = None
    target_ids_slice = suffix_manager.log_p_slice
    for i in range(len(target_ids)):
        position = logits.shape[1] - len(target_ids) + i-1

        next_token_logits = logits[0, position, :]
        next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=0)
        target_token_prob = next_token_probs[target_ids[i]]
        log_prob = torch.log(target_token_prob).item()
        print(f"cache logprob {i}-th token: {log_prob}")
        if i == 0:
            first_token_log_prob = log_prob
        total_log_prob += log_prob
    end_time = time.time()
    
    return total_log_prob, first_token_log_prob

def get_important_tokens(model,context_left,context_right,payload,query, new_adv, target_answer, context_right_recompute_ratio=1.0, segment_size = 50, override_payload_ids=None):
    suffix_manager = SuffixManager(model,context_left,context_right,query, new_adv, payload, target_answer, 0)
    if override_payload_ids is not None:
        from src.util.kv_cache_utils import _override_suffix_manager_payload_ids
        _override_suffix_manager_payload_ids(suffix_manager, override_payload_ids)
    important_positions, importance_values,_, _ = AvgAttentionAttribution(model, ratio=context_right_recompute_ratio).attribute_segment(suffix_manager.prompt_ids,suffix_manager.target_ids,suffix_manager.malicious_instruction_slice,segment_size = segment_size)
    return important_positions, importance_values

def get_logprob_cache_attention(kv_cache,model,context_left,context_right,payload,query, new_adv, target_answer, important_tokens, override_payload_ids=None):

    suffix_manager = SuffixManager(model,context_left,context_right,query, new_adv, payload, target_answer, 0)
    if override_payload_ids is not None:
        from src.util.kv_cache_utils import _override_suffix_manager_payload_ids
        _override_suffix_manager_payload_ids(suffix_manager, override_payload_ids)
    total_len = len(suffix_manager.get_prompt_with_target_ids()) 
    
    malicious_instruction_ids = suffix_manager.get_prompt_with_target_ids()[suffix_manager.malicious_instruction_slice]
    recompute_ids1 = torch.tensor([malicious_instruction_ids]).to(model.model.device)
    recompute_slice1 = slice(suffix_manager.prefix_slice.start, suffix_manager.context_right_recompute_slice1.stop)
    recompute_ids2 = torch.tensor([suffix_manager.after_context_ids_with_target]).to(model.model.device)
    recompute_slice2 = slice(suffix_manager.context_right_slice.stop, suffix_manager.target_answer_slice.stop)
    
    important_positions = important_tokens
    filtered_positions = [i for i in important_positions if i >= recompute_slice1.stop and i < recompute_slice2.start]
    filtered_positions = sorted(filtered_positions)

    if len(filtered_positions) == 0:
        filtered_positions =[recompute_slice1.stop+1]

    recompute_ids = torch.cat([recompute_ids1, torch.tensor([suffix_manager.prompt_ids[i] for i in filtered_positions], device=model.model.device).unsqueeze(0), recompute_ids2], dim=1)
    abs_pos = torch.cat([
    torch.arange(recompute_slice1.start, recompute_slice1.stop, device=model.model.device),
    torch.tensor(filtered_positions, device=model.model.device),
    torch.arange(recompute_slice2.start, recompute_slice2.stop, device=model.model.device),
    ], dim=0)

    attention_mask = torch.ones(1, total_len, dtype=torch.long, device=model.model.device)
    start_time = time.time()

    with torch.no_grad():
        outputs = model.model(input_ids=recompute_ids, past_key_values=kv_cache,cache_position=abs_pos, position_ids=abs_pos.unsqueeze(0),attention_mask=attention_mask,use_cache=True)
    logits = outputs.logits

    target_ids = suffix_manager.target_ids
    total_log_prob = 0
    first_token_log_prob = None
    target_ids_slice = suffix_manager.log_p_slice
    for i in range(len(target_ids)):
        position = logits.shape[1] - len(target_ids) + i-1

        next_token_logits = logits[0, position, :]
        next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=0)
        target_token_prob = next_token_probs[target_ids[i]]
        log_prob = torch.log(target_token_prob).item()
        if i == 0:
            first_token_log_prob = log_prob
        total_log_prob += log_prob
    end_time = time.time()

    return total_log_prob, first_token_log_prob
def get_logprob_cache_test(kv_cache,old_suffix_manager,model,context_left,context_right,payload,query, new_adv, target_answer):

    suffix_manager = SuffixManager(model,context_left,context_right,query, new_adv, payload, target_answer)
    new_input_ids = suffix_manager.get_prompt_with_target_ids()
    old_input_ids = old_suffix_manager.get_prompt_with_target_ids()
    start_position = 0

    for i in range(len(old_input_ids)):
        
        if old_input_ids[i] == new_input_ids[i]:
            start_position +=1
        else:
            break
    
    start_position = min(start_position, len(suffix_manager.prompt_ids))-30
    kv_cache_slice = slice_kv_cache(kv_cache, 0,start_position)
    new_input_ids = torch.tensor([new_input_ids[start_position:]]).to(model.model.device)
    with torch.no_grad():
        outputs = model.model(input_ids=new_input_ids, past_key_values=tuple(kv_cache_slice), use_cache=True)
    logits = outputs.logits
    target_ids = suffix_manager.target_ids
    total_log_prob = 0
    first_token_log_prob = None
    for i in range(len(target_ids)):
        position = logits.shape[1] - len(target_ids) + i-1
        next_token_logits = logits[0, position, :]
        next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=0)
        target_token_prob = next_token_probs[target_ids[i]]
        log_prob = torch.log(target_token_prob).item()
        if i == 0:
            first_token_log_prob = log_prob
        total_log_prob += log_prob
        
    return total_log_prob, first_token_log_prob





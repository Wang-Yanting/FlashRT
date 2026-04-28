import time
import random
import numpy as np
from src.prompts import wrap_prompt
from src.util.opt_utils import *
from .Attacker import Attacker
from src.util.utils import clean_str
from src.util.string_utils import *
from src.util.nano_gcg_utils import *
import time
import random
import numpy as np
import time

import copy
import gc
import logging
import queue
import threading

from dataclasses import dataclass
from tqdm import tqdm
from typing import List, Optional, Tuple, Union
from src.util.gpu_memory_util import get_all_gpu_memory
import torch
import transformers
from torch import Tensor
from transformers import set_seed
from scipy.stats import spearmanr
from .Attacker import *
from src.util.nano_gcg_utils import (
    INIT_CHARS,
    configure_pad_token,
    find_executable_batch_size,
    get_nonascii_toks,
    mellowmax,
)


@dataclass
class ProbeSamplingConfig:
    draft_model: transformers.PreTrainedModel
    draft_tokenizer: transformers.PreTrainedTokenizer
    r: int = 8
    sampling_factor: int = 16


@dataclass
class GCGConfig:
    num_steps: int = 10000
    optim_str_init = None
    
    search_width: int = 1
    batch_size: int = None
    topk: int = 128
    n_replace: int = 1
    buffer_size: int = 1
    use_mellowmax: bool = False
    mellowmax_alpha: float = 1.0
    early_stop: bool = True
    use_kv_cache: bool = True
    allow_non_ascii: bool = False
    add_space_before_target: bool = False
    seed: int = None
    verbosity: str = "INFO"


@dataclass
class GCGResult:
    best_loss: float
    best_string: str
    losses: List[float]
    strings: List[str]


class AttackBuffer:
    def __init__(self, size: int):
        self.buffer = []  # elements are (loss: float, optim_strings: Tuple[tensor, tensor])
        self.size = size

    def add(self, loss: float, optim_ids: List[Tensor]) -> None:
        optim_prefix_ids = optim_ids[0]
        optim_suffix_ids = optim_ids[1]
        if self.size == 0:
            self.buffer = [(loss, optim_prefix_ids, optim_suffix_ids)]
            return

        if len(self.buffer) < self.size:
            self.buffer.append((loss, optim_prefix_ids, optim_suffix_ids))
        else:
            self.buffer[-1] = (loss, optim_prefix_ids, optim_suffix_ids)

        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self) -> Tuple[Tensor,Tensor]:
        return [self.buffer[0][1], self.buffer[0][2]]

    def get_lowest_loss(self) -> float:
        if isinstance(self.buffer[0][0], float):
            return self.buffer[0][0]
        else:
            return self.buffer[0][0].cpu().float()

    def get_highest_loss(self) -> float:
        return self.buffer[-1][0]


def sample_ids_from_grad(
    ids: Tensor,
    grad: Tensor,
    search_width = 1,
    topk: int = 256,
    n_replace: int = 1,
    not_allowed_ids: Tensor = False,
):
    """Returns `search_width` combinations of token ids based on the token gradient.

    Args:
        ids : Tensor, shape = (n_optim_ids)
            the sequence of token ids that are being optimized
        grad : Tensor, shape = (n_optim_ids, vocab_size)
            the gradient of the GCG loss computed with respect to the one-hot token embeddings
        search_width : int
            the number of candidate sequences to return
        topk : int
            the topk to be used when sampling from the gradient
        n_replace : int
            the number of token positions to update per sequence
        not_allowed_ids : Tensor, shape = (n_ids)
            the token ids that should not be used in optimization

    Returns:
        sampled_ids : Tensor, shape = (search_width, n_optim_ids)
            sampled token ids
    """
    n_optim_tokens = len(ids)
    original_ids = ids.repeat(search_width, 1)

    if not_allowed_ids is not None:
        grad[:, not_allowed_ids.to(grad.device)] = float("inf")

    topk_ids = (-grad).topk(topk, dim=1).indices

    sampled_ids_pos = torch.argsort(torch.rand((search_width, n_optim_tokens), device=grad.device))[..., :n_replace]
    sampled_ids_val = torch.gather(
        topk_ids[sampled_ids_pos],
        2,
        torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device),
    ).squeeze(2)

    new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)

    return new_ids


def filter_ids(ids: Tensor, tokenizer: transformers.PreTrainedTokenizer, search_width: int):
    """Filters out sequeneces of token ids that change after retokenization.

    Args:
        ids : Tensor, shape = (search_width, n_optim_ids)
            token ids
        tokenizer : ~transformers.PreTrainedTokenizer
            the model's tokenizer

    Returns:
        filtered_ids : Tensor, shape = (new_search_width, n_optim_ids)
            all token ids that are the same after retokenization
    """
    ids_decoded = tokenizer.batch_decode(ids)
    filtered_ids = []

    for i in range(len(ids_decoded)):
        # Retokenize the decoded token ids
        ids_encoded = tokenizer(ids_decoded[i], return_tensors="pt", add_special_tokens=False).to(ids.device)["input_ids"][0]

        if torch.equal(ids[i], ids_encoded):
            filtered_ids.append(ids[i])
            if len(filtered_ids) >= search_width:
                break


    if not filtered_ids:
        # This occurs in some cases, e.g. using the Llama-3 tokenizer with a bad initialization
        raise RuntimeError(
            "No token sequences are the same after decoding and re-encoding. "
            "Consider setting `filter_ids=False` or trying a different `optim_str_init`"
        )

    return torch.stack(filtered_ids)


class NanoGCGAttacker(OptimizationAttacker):
    def __init__(
        self,
        args,
        model
    ):  
        super().__init__(args)
        self.model = model
        self.config = GCGConfig()
        config = self.config
        self.config.optim_str_init = get_adv_init(model_name = args.model_name)
        self.embedding_layer = model.model.get_input_embeddings()
        tokenizer = model.tokenizer
        self.not_allowed_ids = None if config.allow_non_ascii else get_nonascii_toks(tokenizer, device=self.model.model.device)
        self.kv_cache = None
        

        if model.model.dtype in (torch.float32, torch.float64):
            print(f"Model is in {model.model.dtype}. Use a lower precision data type, if possible, for much faster optimization.")

    def inject(self, args,clean_data, query = None,payload = None,position = "mid",target_answer=None):
        self.args = args
        self.dataset_name = self.args.dataset_name
        
        optimization_result = self.run(clean_data, query,payload,position = position, target_answer=target_answer, config = self.config)
        best_adv = optimization_result['best_advs'][-1]
        target_answer = optimization_result['target_answer']
        context_after_injection,_,_,payload =self.insert_malicious_instruction(clean_data,best_adv,query,payload,target_answer,position)
        self.inject_data = best_adv[0]+payload+best_adv[1]
        self.optimization_result = optimization_result
        orig_msg = wrap_prompt(query, [clean_data],dataset_name=self.dataset_name)
        best_msg = wrap_prompt(query, [context_after_injection],dataset_name=self.dataset_name)
        orig_response_text = self.model.query(orig_msg)
        final_response_text = self.model.query(best_msg)
        optimization_result['orig_response_text'] = orig_response_text
        optimization_result['final_response_text'] = final_response_text
        optimization_result['orig_msg'] = orig_msg
        optimization_result['best_msg'] = best_msg
        if clean_str(optimization_result['target_answer']) in clean_str(final_response_text):
            print("success!!!")
        else:
            print("failed!!!")
        #print("context after injection: ", context_after_injection)
        return context_after_injection

    def run(
        self,
        clean_text,
        query,
        payload,
        position = "mid",
        target_answer = None,
        config = None,
    ):
        model = self.model.model
        tokenizer = self.model.tokenizer
        config = self.config

        if config.seed is not None:
            set_seed(config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

        self.target_answer = target_answer
        _,context_left,context_right, payload = self.insert_malicious_instruction(clean_text,(config.optim_str_init,config.optim_str_init),query,payload,target_answer,position)

        init_ids_prefix = tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
        init_ids_suffix = tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
        #print(init_ids_prefix.tolist(), init_ids_suffix.tolist())
        suffix_manager = SuffixManager(self.model,context_left,context_right,query, (init_ids_prefix.tolist()[0], init_ids_suffix.tolist()[0]), payload, target_answer)
     
        # Tokenize everything that doesn't get optimized
        before_prefix_ids = torch.tensor(suffix_manager.before_prefix_ids).to(model.device).unsqueeze(0)
        payload_ids = torch.tensor(suffix_manager.payload_ids).to(model.device).unsqueeze(0)
        after_suffix_ids = torch.tensor(suffix_manager.after_suffix_ids).to(model.device).unsqueeze(0)
        target_ids = torch.tensor(suffix_manager.target_ids).to(model.device).unsqueeze(0)

        self.before_prefix_ids = before_prefix_ids
        self.payload_ids = payload_ids
        self.after_suffix_ids = after_suffix_ids
        self.target_ids = target_ids

        # Embed everything that doesn't get optimized
        embedding_layer = self.embedding_layer
        before_prefix_embeds, after_suffix_embeds,payload_embeds, target_embeds = [embedding_layer(ids) for ids in (before_prefix_ids, after_suffix_ids, payload_ids, target_ids)]
        self.before_prefix_embeds = before_prefix_embeds
        self.after_suffix_embeds = after_suffix_embeds
        self.payload_embeds = payload_embeds
        self.target_embeds = target_embeds
        #print(self.before_prefix_embeds.shape)
        # Compute the KV Cache for tokens that appear before the optimized tokens
        if config.use_kv_cache:
            with torch.no_grad():
                output = model(inputs_embeds=self.before_prefix_embeds, use_cache=True)
                self.kv_cache = output.past_key_values

        total_n_queries = 0
        total_n_grad_calls = 1
        gpu_peak_memory = 0
        gradient_time = []
        logprob_time = []
        start_time = time.time()
        optim_strings_history = []
        self.best_logprobs_history = []
        self.stop_flag = False

        # Initialize the attack buffer
        buffer = self.init_buffer()
        optim_ids = buffer.get_best_ids()
        optim_embeds_prefix, optim_embeds_suffix = [embedding_layer(ids) for ids in buffer.get_best_ids()]
        


        optim_ids_onehot_grad = self.compute_token_gradient(optim_ids)

        for it in range(config.num_steps):
            # Compute the token gradient
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                sample_prefix_or_suffix = random.choice([0,1])
                # Sample candidate token sequences based on the token gradient
  
                sampled_ids = sample_ids_from_grad(
                    optim_ids[sample_prefix_or_suffix].squeeze(0),
                    optim_ids_onehot_grad[sample_prefix_or_suffix].squeeze(0),
                    64,
                    config.topk,
                    config.n_replace,
                    not_allowed_ids=self.not_allowed_ids,
                )
                #retain search_width samples
                search_width = config.search_width
                sampled_ids = filter_ids(sampled_ids, tokenizer,search_width)

                # Compute loss on all candidate sequences
                batch_size = search_width if config.batch_size is None else config.batch_size
                if self.kv_cache:
                    if sample_prefix_or_suffix == 0:
                        input_embeds = torch.cat([
                            embedding_layer(sampled_ids),
                            payload_embeds,
                            optim_embeds_suffix.repeat(search_width, 1, 1),
                            after_suffix_embeds.repeat(search_width, 1, 1),
                            target_embeds.repeat(search_width, 1, 1),
                        ], dim=1)
                    else:
                        input_embeds = torch.cat([
                            optim_embeds_prefix.repeat(search_width, 1, 1),
                            payload_embeds,
                            embedding_layer(sampled_ids),
                            after_suffix_embeds.repeat(search_width, 1, 1),
                            target_embeds.repeat(search_width, 1, 1),
                        ], dim=1)
                else:
                    if sample_prefix_or_suffix == 0:
                        input_embeds = torch.cat([
                            before_prefix_embeds.repeat(search_width, 1, 1),
                            embedding_layer(sampled_ids),
                            payload_embeds,
                            optim_embeds_suffix.repeat(search_width, 1, 1),
                            after_suffix_embeds.repeat(search_width, 1, 1),
                            target_embeds.repeat(search_width, 1, 1),
                        ], dim=1)
                    else:
                        input_embeds = torch.cat([
                            before_prefix_embeds.repeat(search_width, 1, 1),
                            optim_embeds_prefix.repeat(search_width, 1, 1),
                            payload_embeds,
                            embedding_layer(sampled_ids),
                            after_suffix_embeds.repeat(search_width, 1, 1),
                            target_embeds.repeat(search_width, 1, 1),
                        ], dim=1)

                logprobs_time_start = time.time()
                loss = self._compute_candidates_loss_original( batch_size,input_embeds)
                
                logprobs_time_end = time.time()
                logprob_time.append(logprobs_time_end - logprobs_time_start)
                current_loss = loss.min().item()
                
                optim_ids[sample_prefix_or_suffix] = sampled_ids[loss.argmin()].unsqueeze(0)
                if hasattr(self.model, 'guard_model'):
                    prefix_string =tokenizer.decode(optim_ids[0][0],add_special_tokens=True)
                    suffix_string =tokenizer.decode(optim_ids[1][0],add_special_tokens=True)
                    adv = [prefix_string,suffix_string]
                    msg = wrap_prompt(query, [self.insert_malicious_instruction(clean_text,adv,query,payload,target_answer,position)[0]],dataset_name=self.dataset_name)
                    guard_logprob = self.model.get_guard_logprob(msg,payload)
                    if guard_logprob<-0.5:
                        current_loss = current_loss-guard_logprob
                #print("optim_ids[0]: ", optim_ids[0])
                #print("optim_ids[1]: ", optim_ids[1])
                if it%200 == 0:
                    prefix_string =tokenizer.decode(optim_ids[0][0],add_special_tokens=True)
                    suffix_string =tokenizer.decode(optim_ids[1][0],add_special_tokens=True)
                    target_length = len(tokenizer.encode(target_answer, add_special_tokens=False))
                    exact_logprob,exact_logprob_first_token = get_logprob_suffix_manager(self.model,context_left,context_right,payload,query,(prefix_string,suffix_string),target_answer)
                    print("exact_loss: ",exact_logprob*target_length)
                    print("current_loss: ",current_loss)
                total_n_queries += 1

            
                # Update the buffer based on the loss
            #print("buffer.size: ", buffer.size)
            #print("buffer.get_highest_loss(): ", buffer.get_highest_loss())
            if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                buffer.add(current_loss, optim_ids)
                grad_time_start = time.time()
                optim_ids_onehot_grad = self.compute_token_gradient(optim_ids)
                total_n_grad_calls += 1
                grad_time_end = time.time()
                print("grad_time: ", grad_time_end - grad_time_start)
                gradient_time.append(grad_time_end - grad_time_start)
            optim_ids = buffer.get_best_ids()
            optim_strings = (tokenizer.batch_decode(optim_ids[0])[0],tokenizer.batch_decode(optim_ids[1])[0])
            optim_strings_history.append(optim_strings)
            #print("optim_strings: ", optim_strings)
            
            self.best_logprobs_history.append(-buffer.get_lowest_loss())
            gpu_memory = get_all_gpu_memory()
            if gpu_memory > gpu_peak_memory:
                gpu_peak_memory = gpu_memory
            if it%20 == 0:
                print(f"[iter {it}] logprobs_time: {logprobs_time_end - logprobs_time_start}, best_logprobs: {self.best_logprobs_history[-1]},best_prob: {np.exp(self.best_logprobs_history[-1])}" )
            if self.stop_flag:
                print("Early stopping due to finding a perfect match.")
                break
            
        result_dict = {
        'target_answer': target_answer,
        'n_queries': total_n_queries,
        'n_grad_calls': total_n_grad_calls,
        'gpu_peak_memory': gpu_peak_memory,
        'avg_gradient_time': sum(gradient_time)/len(gradient_time) if len(gradient_time) > 0 else 0.0,
        'avg_logprob_time': sum(logprob_time)/len(logprob_time) if len(logprob_time) > 0 else 0.0,
        'time': time.time() - start_time,
        'best_logprobs': self.best_logprobs_history,
        'best_advs': optim_strings_history,
        'restart_number': 0  # Save the restart number when finished
        }
        return result_dict

    def init_buffer(self) -> AttackBuffer:
        model = self.model.model
        tokenizer = self.model.tokenizer
        config = self.config

        # Create the attack buffer and initialize the buffer strings
        buffer = AttackBuffer(config.buffer_size)

        init_optim_ids_prefix = tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
        init_optim_ids_suffix = tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
        init_buffer_ids_prefix = init_optim_ids_prefix
        init_buffer_ids_suffix = init_optim_ids_suffix

        true_buffer_size = max(1, config.buffer_size)

        # Compute the loss on the initial buffer entries
        if self.kv_cache:
            init_buffer_embeds = torch.cat([
                self.embedding_layer(init_buffer_ids_prefix),
                self.payload_embeds.repeat(true_buffer_size, 1, 1),
                self.embedding_layer(init_buffer_ids_suffix),
                self.after_suffix_embeds.repeat(true_buffer_size, 1, 1),
                self.target_embeds.repeat(true_buffer_size, 1, 1),
            ], dim=1)
        else:
            init_buffer_embeds = torch.cat([
                self.before_prefix_embeds.repeat(true_buffer_size, 1, 1),
                self.embedding_layer(init_buffer_ids_prefix),
                self.payload_embeds.repeat(true_buffer_size, 1, 1),
                self.embedding_layer(init_buffer_ids_suffix),
                self.after_suffix_embeds.repeat(true_buffer_size, 1, 1),
                self.target_embeds.repeat(true_buffer_size, 1, 1),
            ], dim=1)

        init_buffer_losses = self._compute_candidates_loss_original( true_buffer_size,init_buffer_embeds)

        # Populate the buffer
        for i in range(true_buffer_size):
            buffer.add(init_buffer_losses[i], [init_buffer_ids_prefix[[i]], init_buffer_ids_suffix[[i]]])

        print("Initialized attack buffer.")

        return buffer

    def compute_token_gradient(
        self,
        optim_ids :Tuple[Tensor,Tensor],
    ) -> Tensor:
        """Computes the gradient of the GCG loss w.r.t the one-hot token matrix.

        Args:
            optim_ids : Tensor, shape = (1, n_optim_ids)
                the sequence of token ids that are being optimized
        """
        optim_ids_prefix = optim_ids[0]
        optim_ids_suffix = optim_ids[1]
        model = self.model.model
        embedding_layer = self.embedding_layer

        # Create the one-hot encoding matrix of our optimized token ids
        optim_ids_prefix_onehot = torch.nn.functional.one_hot(optim_ids_prefix, num_classes=embedding_layer.num_embeddings)
        optim_ids_prefix_onehot = optim_ids_prefix_onehot.to(model.device, model.dtype)

        optim_ids_suffix_onehot = torch.nn.functional.one_hot(optim_ids_suffix, num_classes=embedding_layer.num_embeddings)
        optim_ids_suffix_onehot = optim_ids_suffix_onehot.to(model.device, model.dtype)

        optim_ids_prefix_onehot.requires_grad_()
        optim_ids_suffix_onehot.requires_grad_()

        # (1, num_optim_tokens, vocab_size) @ (vocab_size, embed_dim) -> (1, num_optim_tokens, embed_dim)
        optim_embeds_prefix = optim_ids_prefix_onehot @ embedding_layer.weight
        optim_embeds_suffix = optim_ids_suffix_onehot @ embedding_layer.weight

        if self.kv_cache:
            input_embeds = torch.cat(
                [optim_embeds_prefix, 
                self.payload_embeds, 
                optim_embeds_suffix, 
                self.after_suffix_embeds, 
                self.target_embeds], dim=1)

            output = model(
                inputs_embeds=input_embeds,
                past_key_values=self.kv_cache,
                use_cache=True,
            )
        else:
            input_embeds = torch.cat(
                [
                    self.before_prefix_embeds,
                    optim_embeds_prefix,
                    self.payload_embeds,
                    optim_embeds_suffix,
                    self.after_suffix_embeds,
                    self.target_embeds,
                ],
                dim=1,
            )
            output = model(inputs_embeds=input_embeds)

        logits = output.logits

        # Shift logits so token n-1 predicts token n
        shift = input_embeds.shape[1] - self.target_ids.shape[1]
        shift_logits = logits[..., shift - 1 : -1, :].contiguous()  # (1, num_target_ids, vocab_size)
        shift_labels = self.target_ids

        if self.config.use_mellowmax:
            label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
            loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
        else:
            loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        optim_ids_prefix_grad, optim_ids_suffix_grad = torch.autograd.grad(outputs=[loss], inputs=[optim_ids_prefix_onehot, optim_ids_suffix_onehot])

        return [optim_ids_prefix_grad, optim_ids_suffix_grad]

    def _compute_candidates_loss_original(
        self,
        search_batch_size: int,
        input_embeds: Tensor,
    ) -> Tensor:
        """Computes the GCG loss on all candidate token id sequences.

        Args:
            search_batch_size : int
                the number of candidate sequences to evaluate in a given batch
            input_embeds : Tensor, shape = (search_width, seq_len, embd_dim)
                the embeddings of the `search_width` candidate sequences to evaluate
        """
        all_loss = []
        prefix_cache_batch = []
        num_target_tokens = self.target_ids.shape[1]
        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i:i + search_batch_size]
                current_batch_size = input_embeds_batch.shape[0]

                if self.kv_cache:
                    
                    if isinstance(self.kv_cache, list) or isinstance(self.kv_cache, tuple):
                        #print(f"Type of kv_cache: {type(self.kv_cache)}")
                        if not prefix_cache_batch or current_batch_size != search_batch_size:
                            prefix_cache_batch = [[x.expand(current_batch_size, -1, -1, -1) for x in self.kv_cache[i]] for i in range(len(self.kv_cache))]

                        outputs = self.model.model(inputs_embeds=input_embeds_batch, past_key_values=prefix_cache_batch, use_cache=True)
                    else:
                        outputs = self.model.model(inputs_embeds=input_embeds_batch, past_key_values=self.kv_cache, use_cache=True)
                else:
                    outputs = self.model.model(inputs_embeds=input_embeds_batch)

                logits = outputs.logits

                tmp = input_embeds.shape[1] - self.target_ids.shape[1]
                shift_logits = logits[..., tmp-1:-1, :].contiguous()
                shift_labels = self.target_ids.repeat(current_batch_size, 1)

                if self.config.use_mellowmax:
                    label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
                    loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
                    logprob_first_token = -torch.nn.functional.cross_entropy(shift_logits[0, 0:1], shift_labels[0, 0:1], reduction="none")
                else:
                    loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none")
                    logprob_first_token = -torch.nn.functional.cross_entropy(shift_logits[0, 0:1], shift_labels[0, 0:1], reduction="none")
                loss = loss.view(current_batch_size, -1).mean(dim=-1)
                all_loss.append(loss*num_target_tokens)
                logprob_first_token = logprob_first_token.cpu().float()
                if self.config.early_stop:
                    if "qwen" in self.model_name or "deepseek" in self.model_name or "prompt-guard" in self.model_name or "secalign" in self.model_name:
                        no_improvement_threshold_prob =0
                    else:
                        no_improvement_threshold_prob = 0.0001
                    if early_stopping_condition(self.best_logprobs_history, logprob_first_token, self.model,self.target_answer,no_improvement_threshold_prob=no_improvement_threshold_prob):
                        self.stop_flag = True

                del outputs
                gc.collect()
                torch.cuda.empty_cache()

        return torch.cat(all_loss, dim=0)

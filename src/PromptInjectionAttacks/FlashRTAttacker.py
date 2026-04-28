import time
import random
import numpy as np
from src.prompts import wrap_prompt
from src.util.opt_utils import *

from .Attacker import *
from src.util.utils import clean_str
from src.util.gcg_utils import *
from src.util.kv_cache_utils import *
import time
import random
import numpy as np
from src.util.gpu_memory_util import get_all_gpu_memory
class FlashRTAttacker(OptimizationAttacker):
    def __init__(self,args,model):
        super().__init__(args)
        self.model = model
        self.gradient_subsample_ratio = args.gradient_subsample_ratio
        self.gradient_subsample_interval = args.gradient_subsample_interval
        self.context_right_recompute_ratio = args.context_right_recompute_ratio
        self.segment_size = args.segment_size
        self.n_iterations = args.n_iterations
        self.n_restarts = args.n_restarts
        self.n_tokens_change_max = args.n_tokens_change_max
        self.heuristic = args.heuristic

    def inject(self, args,clean_data, query,payload,position = "mid",target_answer=None):
        self.args = args
        self.dataset_name = self.args.dataset_name
        optimization_result = self.gcg_search(args, clean_data, query,payload,position = position, target_answer=target_answer)
        best_adv = optimization_result['best_advs'][-1]
        target_answer = optimization_result['target_answer']
        context_after_injection,_,_,payload =self.insert_malicious_instruction(clean_data,best_adv,query,payload,target_answer,position)
        self.inject_data = best_adv[0]+payload+best_adv[1]
        self.optimization_result = optimization_result
        return context_after_injection




        
    def gcg_search(self,args, clean_data, query,payload,position = "mid",target_answer=None):
        
        model = self.model
        self.target_answer = target_answer
        n_iterations_target_not_in_topk, n_retry_final = 25, 5
        
        tokenizer, max_token_value = model.tokenizer, model.tokenizer.vocab_size
        
        start_time = time.time()
        total_n_queries = 0
        total_n_grad_calls = 0
        attribution_time = []
        gradient_time = []
        logprob_time = []
        gpu_peak_memory = 0
        error = 0
        error_count = 0
        logprobs = []
        exact_logprobs = []
        if "qwen" in self.model_name or "deepseek" in self.model_name or "secalign" in self.model_name or "prompt-guard" in self.model_name:
            no_improvement_threshold_prob =0
        else:
            no_improvement_threshold_prob = 0.0001
        for i_restart in range(self.n_restarts):    
            adv_init = get_adv_init(model_name = self.model_name,dataset_name =self.dataset_name)
            if isinstance(adv_init, list):
                adv = adv_init
            else:
                adv = [adv_init, adv_init]
            
            best_adv = adv.copy()

            msg = wrap_prompt(query, [self.insert_malicious_instruction(clean_data,adv,query,payload,target_answer,position)[0]],dataset_name=self.dataset_name)
            _,context_left,context_right, payload = self.insert_malicious_instruction(clean_data,best_adv,query,payload,target_answer,position)
            orig_response_text = model.query(msg)

            orig_msg = msg
            best_msg = msg

            best_logprob = -np.inf 
            logprob_first_token = -np.inf
            print(f'Original message: {orig_msg}')
            print(f'Original response text: {orig_response_text}')

            success = False
            n_tokens_change = self.n_tokens_change_max

            best_logprobs, best_advs = [], []
            
            judge_n_calls = 0
            response_list = []
            candidate_set = init_candidate_set(model,adv)
            kv_cache,cache_suffix_manager = initialize_kv_cache(model,context_left,context_right,payload,query, best_adv, target_answer)
            
            kv_cache, _,_ = to_static(model.model,kv_cache)
            
            important_tokens, importance_values = get_important_tokens(model,context_left,context_right,payload,query, adv, target_answer, context_right_recompute_ratio=self.context_right_recompute_ratio,segment_size = self.segment_size)
  
            n_iterations = self.n_iterations
            for it in range(1, n_iterations + 1):
                logprob_time_start = time.time()
                torch.cuda.reset_peak_memory_stats()

                if not early_stopping_condition(best_logprobs, logprob_first_token, model,target_answer,no_improvement_threshold_prob=no_improvement_threshold_prob) and not it == n_iterations:  
                    total_n_queries += 1
                    
                    logprob,logprob_first_token = get_logprob_cache_attention(kv_cache,model,context_left,context_right,payload,query, adv, target_answer,important_tokens)
                    if hasattr(model, 'guard_model'):
                        guard_logprob = model.get_guard_logprob(msg,payload)
                        if guard_logprob<-0.5:
                            logprob = (guard_logprob+logprob)
                    if it%200 ==0:
                        exact_logprob,exact_logprob_first_token = get_logprob(model,msg,target_answer)

                        print("exact prob:", np.exp(exact_logprob))
                        print("kv cache logprob", np.exp(logprob))
                else:  # early stopping criterion (important for query/token efficiency)
                    
                    logprob,logprob_first_token = get_logprob_cache_attention(kv_cache,model,context_left,context_right,payload,query, best_adv, target_answer, important_tokens)
                    if hasattr(model, 'guard_model'):
                        guard_logprob = model.get_guard_logprob(msg,payload)
                        if guard_logprob<-0.5:
                            logprob = (guard_logprob+logprob)
                    exact_logprob,exact_logprob_first_token = get_logprob(model,best_msg,target_answer)
                    print("==============early stop==========")
                    print("exact prob:", np.exp(exact_logprob))
                    print("kv cache logprob", np.exp(logprob))
                    msg_early_stop = best_msg 
                    output = model.query(best_msg)
                    print(f'output: {output}')
                    print(f'target_answer: {target_answer}')
                    if clean_str(target_answer) in clean_str(output):
                        print("success!!!")
                        success = True
                        break
                    else:
                        print(f'restart')
                        break
                logprob_time_end = time.time()
                logprob_time.append(logprob_time_end-logprob_time_start)
                if it%20 == 0:
                    print(f'it={it} [time: {logprob_time_end-logprob_time_start}] [best] logprob={best_logprob:.3f} prob={np.exp(best_logprob):.5f}  [curr] logprob={logprob:.3f} prob={np.exp(logprob):.5f} prob_first_token={np.exp(logprob_first_token):.5f}, target answer={target_answer}')
                    print(f'adv: {adv}')
                if no_improvement_condition(best_logprobs,no_improvement_history=self.gradient_subsample_interval, no_improvement_threshold_prob=no_improvement_threshold_prob) and it%self.gradient_subsample_interval == 0:
                    get_candidate_time_start = time.time()
                    total_n_grad_calls += 1

                    candidate_set = get_candidate_set_segment_default(model,context_left,context_right,payload,query, best_adv, target_answer, context_remove_ratio=1-self.gradient_subsample_ratio, segment_size = self.segment_size)
                    get_candidate_time_end= time.time()
                    print(f'get_candidate_time: {get_candidate_time_end-get_candidate_time_start}')
                if logprob > best_logprob:
                    #print(f'logprob > best_logprob')
                    #print(f'logprob: {logprob}, best_logprob: {best_logprob}')
                    best_adv = adv.copy()  # Create a copy instead of reference
                    get_candidate_time_start = time.time()
                    total_n_grad_calls += 1
     
                    candidate_set = get_candidate_set_segment_default(model,context_left,context_right,payload,query, best_adv, target_answer, context_remove_ratio=1-self.gradient_subsample_ratio,segment_size = self.segment_size)
                    get_candidate_time_end= time.time()
                    print(f'get_candidate_time: {get_candidate_time_end-get_candidate_time_start}')
                    gradient_time.append(get_candidate_time_end-get_candidate_time_start)
                    best_logprob = logprob
                    best_msg = msg  
             
                    kv_cache,_ = initialize_kv_cache(model,context_left,context_right,payload,query, best_adv, target_answer)
                    cache_conversion_time_start = time.time()
                    
                    kv_cache, _,_ = to_static(model.model,kv_cache)
                    cache_conversion_time_end = time.time()
                    #print(f"cache conversion time: {cache_conversion_time_end-cache_conversion_time_start}")
                    attribution_time_start = time.time()
                    important_tokens, importance_values = get_important_tokens(model,context_left,context_right,payload,query, best_adv, target_answer, context_right_recompute_ratio=self.context_right_recompute_ratio,segment_size = self.segment_size)
                    attribution_time_end = time.time()
                    attribution_time.append(attribution_time_end-attribution_time_start)
                best_logprobs.append(best_logprob)
                best_advs.append(best_adv)
                
                change_suffix_or_prefix = random.choice([0,1])
                n_tokens_change = schedule_n_to_change_prob(self.n_tokens_change_max, best_logprobs)

                best_adv_tokens = tokenizer.encode(best_adv[change_suffix_or_prefix], add_special_tokens=False)
                
                count = 0

                max_count = 100
                if "yi" in self.model_name or "codellama" in self.model_name or "mistral" in self.model_name:
                    max_count = 100
                while True:
                    #print(count)
                    if count > max_count:
                        print("Can not reach consistent tokenization!")
                        break

                    count += 1
                    adv_tokens = tokenizer.encode(best_adv[change_suffix_or_prefix], add_special_tokens=False)
                    substitute_pos_start = random.choice(range(len(adv_tokens)))
            
                    substitution_tokens = [random.choice(candidate_set[change_suffix_or_prefix][pos]) for pos in range(substitute_pos_start, min(substitute_pos_start + n_tokens_change, len(best_adv_tokens)))]

                    adv_tokens = adv_tokens[:substitute_pos_start] + substitution_tokens + adv_tokens[substitute_pos_start+n_tokens_change:]

                    if len(tokenizer.encode(tokenizer.decode(adv_tokens), add_special_tokens=False)) == len(best_adv_tokens) and tokenization_filter(tokenizer.decode(adv_tokens),change_suffix_or_prefix,model,context_left,context_right,payload,query, best_adv, target_answer):
                        break
                adv = best_adv.copy()
                adv[change_suffix_or_prefix] = tokenizer.decode(adv_tokens)
                
                # apply the new adversarial suffix
                msg = wrap_prompt(query, [self.insert_malicious_instruction(clean_data,adv,query,payload,target_answer,position)[0]],dataset_name=self.dataset_name)
                memory_it = get_all_gpu_memory()
                if memory_it > gpu_peak_memory:
                    gpu_peak_memory = memory_it
            print(f'orig_response_text: {orig_response_text}\n\n')
            print(f'final_response_text: {output}\n\n')
            print(f'max_prob={np.exp(best_logprob)},success = {success}, adv={best_adv}')
            print('\n\n\n')

            if success:  # exit the random restart loop
                break

        
        result_dict = {
            'target_answer': target_answer,
            'orig_response_text': orig_response_text,
            'final_response_text': output,
            'n_queries': total_n_queries,
            'n_grad_calls': total_n_grad_calls,
            'time': time.time() - start_time,
            'avg_attribution_time': sum(attribution_time)/len(attribution_time),
            'avg_gradient_time': sum(gradient_time)/len(gradient_time),
            'avg_logprob_time': sum(logprob_time)/len(logprob_time),
            'gpu_peak_memory': gpu_peak_memory,
            'orig_msg': orig_msg,
            'best_msg': best_msg,
            'best_logprobs': best_logprobs,
            'best_advs': best_advs,
            'restart_number': i_restart  # Save the restart number when finished
        }
        return result_dict

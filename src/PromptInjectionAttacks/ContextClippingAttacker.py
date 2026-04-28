import time
import random
import numpy as np
from src.prompts import wrap_prompt
from src.util.opt_utils import *
from src.util.string_utils import random_context_clipping
from .Attacker import *
from src.util.utils import clean_str
from src.util.gcg_utils import *
import time
import random
import numpy as np
from src.prompts import wrap_prompt
import time
from src.util.gpu_memory_util import get_all_gpu_memory
class ContextClippingAttacker(OptimizationAttacker):
    def __init__(self,args,model):
        super().__init__(args)
        self.model = model
        self.context_right_recompute_ratio = args.context_right_recompute_ratio
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

    def gcg_search(self,args, clean_data, query,payload,position = "mid", target_answer=None):
        
        model = self.model

        self.target_answer = target_answer
        n_iterations_target_not_in_topk, n_retry_final = 25, 5
        
        tokenizer, max_token_value = model.tokenizer, model.tokenizer.vocab_size
        
        gradient_time = []
        logprob_time = []
        gpu_peak_memory = 0
        start_time = time.time()
        total_n_queries = 0
        total_n_grad_calls = 0

        for i_restart in range(self.n_restarts):    
            adv_init = get_adv_init(dataset_name = self.dataset_name)
            adv = [adv_init, adv_init]
            
            best_adv = adv.copy()

            msg = wrap_prompt(query, [self.insert_malicious_instruction(clean_data,adv,query,payload,target_answer,position)[0]])
            _,context_left,context_right, payload = self.insert_malicious_instruction(clean_data,best_adv,query,payload,target_answer,position)
            clipped_context_left,clipped_context_right = random_context_clipping(context_left,context_right)
            orig_response_text = model.query(msg)
            orig_msg = msg
            best_msg = msg

            best_logprob = -np.inf 
            logprob_first_token = -np.inf
            print(f'Original message: {orig_msg}')

            success = False
            n_tokens_change = self.n_tokens_change_max


            best_logprobs, best_advs = [], []
            judge_n_calls = 0
            response_list = []
            candidate_set = init_candidate_set(model,adv)
            kv_cache,old_suffix_manager = initialize_kv_cache(model,clipped_context_left,clipped_context_right,payload,query, best_adv, target_answer)
            n_iterations = self.n_iterations
            for it in range(1, n_iterations + 1):
                logprob_time_start = time.time()
                torch.cuda.reset_peak_memory_stats()
                # note: to avoid an extra call to get_response(), for args.determinstic_jailbreak==True, the logprob_dict from the previous iteration is used 
                if not early_stopping_condition(best_logprobs, logprob_first_token, model,target_answer) and not it == n_iterations:  
                    total_n_queries += 1
                    
                    logprob,logprob_first_token = get_logprob_cache_test(kv_cache,old_suffix_manager,model,clipped_context_left,clipped_context_right,payload,query, adv, target_answer)
                    
                  
                    #print(f'iter time: {it_end-it_start}')
                    if it%500 == 0:
                        exact_logprob,exact_logprob_first_token = get_logprob(model,msg,target_answer)
                        print("exact logprob:", exact_logprob)
                        print("kv cache logprob:", logprob)
                        print("error:", logprob - exact_logprob)

                else:  # early stopping criterion (important for query/token efficiency)
                    
                    logprob,logprob_first_token = get_logprob_cache_test(kv_cache,old_suffix_manager,model,clipped_context_left,clipped_context_right,payload,query, best_adv, target_answer)
                    exact_logprob,exact_logprob_first_token = get_logprob_suffix_manager(model,context_left,context_right,payload,query,best_adv,target_answer)
                    print("==============early stop==========")
                    print("exact prob:", np.exp(exact_logprob))
                    print("kv cache prob", np.exp(logprob))
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

                if logprob > best_logprob:
                    #print(f'logprob > best_logprob')
                    #print(f'logprob: {logprob}, best_logprob: {best_logprob}')
                    best_adv = adv.copy()  # Create a copy instead of reference
                    get_candidate_time_start = time.time()
                    total_n_grad_calls += 1
                    torch.cuda.reset_peak_memory_stats()
                    
                    candidate_set = get_candidate_set_vanilla(model,clipped_context_left,clipped_context_right,payload,query, best_adv, target_answer)

                    get_candidate_time_end= time.time()
                    gradient_time.append(get_candidate_time_end-get_candidate_time_start)
                    print(f'get_candidate_time: {get_candidate_time_end-get_candidate_time_start}')
                    best_logprob = logprob
                    best_msg = msg  
             
                    kv_cache,_ = initialize_kv_cache(model,clipped_context_left,clipped_context_right,payload,query, best_adv, target_answer)
                best_logprobs.append(best_logprob)
                best_advs.append(best_adv)
                
                change_suffix_or_prefix = random.choice([0,1])
                n_tokens_change = schedule_n_to_change_prob(4, best_logprobs)

                best_adv_tokens = tokenizer.encode(best_adv[change_suffix_or_prefix], add_special_tokens=False)
                
                count = 0
                while True:

                    if count > 1000:
                        print(f'Failed to find a valid adversarial suffix after 100 attempts')
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
                
                msg = wrap_prompt(query, [self.insert_malicious_instruction(clean_data,adv,query,payload,target_answer,position)[0]])
                gpu_memory = get_all_gpu_memory()
                if gpu_memory > gpu_peak_memory:
                    gpu_peak_memory = gpu_memory

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

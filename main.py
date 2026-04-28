import argparse
from src.models import create_model
from src.load_dataset import _load_dataset, generate_incorrect_answer
from src.util.utils import *
from src.util.utils import _save_results
from src.prompts import wrap_prompt, get_payload
import gc
import torch
import src.PromptInjectionAttacks as PI

def parse_args():
    parser = argparse.ArgumentParser(prog='RAGdebugging', description="test")


    # General args
    parser.add_argument('--model_name', type=str, default='llama3.1-70b',
                        help="Name of the model to be used.")
    parser.add_argument("--dataset_name", type=str, default='nq-poison',
                        choices=['sst2', 'sms_spam', 'nq-poison', 'hotpotqa-poison', 'msmarco-poison',  # RAG with knowledge corruption attack
                                 "narrativeqa", "musique", "gov_report"],  # Prompt injection attack to LongBench datasets
                        help="Name of the dataset to be used.")
    # Optimization args
    parser.add_argument('--n_iterations', type=int, default=10000, 
                        help="Number of iterations.")
    parser.add_argument('--batch_size', type=int, default=1, 
                        help="Batch size.")
    parser.add_argument('--n_restarts', type=int, default=5, 
                        help="Number of restarts.")
    parser.add_argument('--n_tokens_change_max', type=int, default=4, 
                        help="Number of tokens to change.")
    parser.add_argument('--context_right_recompute_ratio', type=float, default=0.2)
    parser.add_argument('--gradient_subsample_ratio', type=float, default=0.2)
    parser.add_argument('--gradient_subsample_interval', type=int, default=100)
    parser.add_argument('--segment_size', type=int, default=50, 
                        help="Segment size for the context.")

  
    # prompt injection attack to LongBench
    parser.add_argument('--prompt_injection_attack', default='default', type=str, 
                        help="Type of prompt injection attack to perform.")
    parser.add_argument('--inject_times', type=int, default=1,
                        help="Number of times to inject the prompt.")
    parser.add_argument('--max_length', default=32000, type=int,
                        help="Control the maximum length of the context.")
    parser.add_argument('--position', default='end', type=str, 
                        help="Position to inject the prompt.")
    parser.add_argument('--heuristic', default='simple', type=str, 
                        help="Heuristic to use for the prompt injection attack.")

    #PoisonedRAG
    parser.add_argument('--retrieval_k', type=int, default=100, 
                        help="Number of top contexts to retrieve.")
    parser.add_argument("--retriever", type=str, default='contriever', 
                        help="Retriever model to be used.") # BEIR
    # other settings
    parser.add_argument('--gpu_id', type=str, default='-1', 
                        help="ID of the GPU to be used.")
    parser.add_argument('--seed', type=int, default=12, 
                        help="Random seed for reproducibility.")
    parser.add_argument('--data_start', type=int, default=0, 
        help="Number of evaluation data points.")           
    parser.add_argument('--data_num', type=int, default=50, 
                        help="Number of evaluation data points.")
    parser.add_argument("--results_path", type=str, default="main", 
                        help="Path to save the results.")
    parser.add_argument('--evaluate_saved', action='store_true', 
                        help="Evaluate the saved results.")
    parser.add_argument('--verbose', type=int, default=1, 
                        help="Enable verbose mode for detailed logging.")

    args = parser.parse_args()
    
    
    print(args)
    return args



def main(args):
    if args.dataset_name in ['nq-poison', 'hotpotqa-poison', 'msmarco-poison']: 
        benchmark = 'PoisonedRAG'
        args.benchmark = 'PoisonedRAG'
    elif args.dataset_name in ["narrativeqa",  "musique",  "gov_report"]:
        benchmark = 'LongBench'
        args.benchmark = 'LongBench'
    results_path = args.results_path
    
    # Load dataset and random select
    
    # Load LLM and init Attribution
    print("Loading LLM!")
    if args.gpu_id =="-1":
        llm = create_model(config_path = f'model_configs/{args.model_name}_config.json', device = f"auto")
    else:
        llm = create_model(config_path = f'model_configs/{args.model_name}_config.json', device = f"cuda:{args.gpu_id}")
    
    attr_results = []

    print("Loading Dataset!")
    dataset = _load_dataset(args.dataset_name, args.retriever, args.retrieval_k, 
    model_name=args.model_name, shot=1, seed=args.seed,num_poison = args.inject_times)
    attacker = PI.create_attacker(args,llm)

    data_num = 0 #initialize a counter for data number
    avg_optimization_time =0
    avg_attribution_time = 0
    avg_logprob_time = 0
    avg_gradient_time = 0
    avg_num_queries = 0
    avg_num_grad_calls = 0
    gpu_memory_use = 0
    ASV_counter = 0
    clean_ASV_counter = 0
    # Convert to list if it's a HuggingFace Dataset to avoid slicing issues
    if hasattr(dataset, 'select'):
        # It's a HuggingFace Dataset, use select method
        dataset = dataset.select(range(args.data_start, args.data_start + args.data_num))
        dataset = list(dataset)
    else:
        # It's already a list, slice normally
        dataset = dataset[args.data_start:args.data_start + args.data_num]
    for idx, dp in enumerate(dataset):
        print(f"\n------------------Start question {idx} -------------------")
        
        # Save results every 10 questions
        if idx > 1:
            _save_results(args, attr_results, results_path)


        # Extract context and question for LongBench
        if benchmark == 'LongBench':
            contexts = dp['context']
            question = dp["input"]
            gt_answer = dp["answers"]
            target_response = generate_incorrect_answer(question,args.dataset_name)
            payload =  get_payload(question,target_response, args.heuristic)
        elif benchmark == 'PoisonedRAG':

            question = dp['question']
            contexts = dp['topk_contents']
            target_response = dp["incorrect_answer"]

            injected_adv = dp["injected_adv"]
            
            payload = dp["payload"][0]
            if "deepseek" in args.model_name:
                target_response = f"{payload}</think>"
            contexts = "\n\n".join(contexts)
        # Get the length of the context, if it is longer than max_length, truncate it
        context_length = llm.get_prompt_length(contexts)
        if context_length > args.max_length:
            contexts = llm.cut_context(contexts,args.max_length)
        print("Question:", question)
        print("Context length:", context_length)

        # Generate a clean prompt and query the LLM. Used to calculate attack success rate without attack
        clean_prompt = wrap_prompt(question, [contexts],dataset_name=args.dataset_name)
        clean_answer = llm.query(clean_prompt)

        # Inject adversarial content
        contexts= attacker.inject(args, contexts, query=question,payload = payload, position = args.position,target_answer=target_response)
        if hasattr(attacker, 'optimization_result'):
            attr_results.append([attacker.optimization_result['best_advs'][-1], attacker.optimization_result['target_answer']])
        print("contexts after injection: ", contexts)
        
        if hasattr(attacker, 'optimization_result'):
            orig_response_text = attacker.optimization_result['orig_response_text']
            if 'time' in attacker.optimization_result:
                avg_optimization_time += attacker.optimization_result['time']
            if 'avg_attribution_time' in attacker.optimization_result:
                avg_attribution_time += attacker.optimization_result['avg_attribution_time']
            if 'avg_logprob_time' in attacker.optimization_result:
                avg_logprob_time += attacker.optimization_result['avg_logprob_time']
            if 'avg_gradient_time' in attacker.optimization_result:
                avg_gradient_time += attacker.optimization_result['avg_gradient_time']
            if 'n_queries' in attacker.optimization_result:
                avg_num_queries += attacker.optimization_result['n_queries']
            if 'n_grad_calls' in attacker.optimization_result:
                avg_num_grad_calls += attacker.optimization_result['n_grad_calls']
            if 'gpu_peak_memory' in attacker.optimization_result:
                gpu_memory_use += attacker.optimization_result['gpu_peak_memory']
        if benchmark == 'LongBench':
            target_response = attacker.target_answer

        # Query the LLM with the injected context
        prompt = wrap_prompt(question, [contexts],dataset_name=args.dataset_name)

        answer = llm.query(prompt)
        print("LLM's answer: [", answer, "]")
        if hasattr(attacker, 'optimization_result'):
            orig_response_text = attacker.optimization_result['orig_response_text']
            print("LLM's answer before optimization: [", orig_response_text, "]")
        else:
            print("LLM's answer before attack: [", clean_answer, "]")
        print("Target answer: [", target_response, "]")

        # Check if the target response is in the LLM's answer
        ASV = clean_str(target_response) in clean_str(answer)

        ASV_clean = clean_str(target_response) in clean_str(clean_answer)
        if ASV_clean:
            clean_ASV_counter += 1
        
        if data_num >= args.data_num:
            break
        if not ASV and args.prompt_injection_attack != 'none':
            data_num += 1
            print(f"Attack fails, continue")
            print("Current avg optimization time: ", avg_optimization_time / data_num)
            print("Current avg gpu memory use: ", gpu_memory_use / data_num)
            continue
        else:
            data_num += 1
            ASV_counter += 1

        print("Current ASV: ", ASV_counter / data_num)
        print("Current ASV clean: ", clean_ASV_counter / data_num)
        print("Current avg optimization time: ", avg_optimization_time / data_num)
        print("Current avg attribution time: ", avg_attribution_time / data_num)
        print("Current avg logprob time: ", avg_logprob_time / data_num)
        print("Current avg gradient time: ", avg_gradient_time / data_num)
        print("Current avg num queries: ", avg_num_queries / data_num)
        print("Current avg num grad calls: ", avg_num_grad_calls / data_num)
        print("Current avg gpu memory use: ", gpu_memory_use / data_num)
        # Perform attribution and append results
        
        print('done!')

    print(" ASV: ", ASV_counter / data_num)
    print(" ASV clean: ", clean_ASV_counter / data_num)
    print(" avg optimization time: ", avg_optimization_time / data_num)
    print(" avg attribution time: ", avg_attribution_time / data_num)
    print(" avg logprob time: ", avg_logprob_time / data_num)
    print(" avg gradient time: ", avg_gradient_time / data_num)
    print(" avg num queries: ", avg_num_queries / data_num)
    print(" avg num grad calls: ", avg_num_grad_calls / data_num)
    print(" avg gpu memory use: ", gpu_memory_use / data_num)
    # Delete the model, tokenizer, and any other large objects to free memory
    del llm
    
    # Force deletion of any CUDA tensors
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                del obj
        except:
            pass
            
    # Run garbage collection multiple times to ensure cleanup
    gc.collect()
    gc.collect()
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()

if __name__ == '__main__':
    args = parse_args()
    setup_seeds(args.seed)

    if args.evaluate_saved == False:
        main(args)



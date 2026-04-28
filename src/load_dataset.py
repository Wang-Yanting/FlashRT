'''
    Easily process & load LongBench and PoisonedRAG datasets.
'''
from src.util.utils import load_json
from datasets import load_dataset
import os
from src.models import create_model
def load_poison(dataset_name='nq-poison',retriever = 'contriever',top_k =5, num_poison = 1):
    result_path = f"datasets/PoisonedRAG/{dataset_name}-{retriever}.json"
    results_list = load_json(result_path)
    processed_results = []
    for iter,iteration_result in enumerate(results_list):
        processed_results.extend(iteration_result[f'iter_{iter}'])
    for result in processed_results:
        result['topk_results']=result['topk_results'][:top_k+4]
        result['topk_contents']=[result['topk_results'][i]['context'] for i in range(top_k)]
        target = result['incorrect_answer']
        question = result['question']
        # First, capture payloads (contexts containing the question) BEFORE removing them
        result['payload'] = [context for context in result['topk_contents'] if question in context]
        # Then remove all poisoned contexts that contain the question
        result['topk_contents'] = [context for context in result['topk_contents'] if question not in context]
    print("Processed result size: ",len(processed_results))
    return processed_results


def _load_dataset(dataset_name='nq-poison', retriever='contriever', retrieval_k=5, **kwargs):
    num_poison = kwargs.get('num_poison', 5)
    print("Load dataset: ",dataset_name)
    if dataset_name in ["narrativeqa","musique","gov_report"]:
        print("datset_name: ",dataset_name)
        dataset = load_dataset('THUDM/LongBench', dataset_name, split='test', trust_remote_code=True)
        if dataset_name == "gov_report":
            def modify_gov_report(example):
                #print("query is: ", example["input"])
                example["input"] = "Write a one-page summary of the government report."
                return example
            
            dataset = dataset.map(modify_gov_report)

    elif dataset_name in ['nq-poison', 'hotpotqa-poison', 'msmarco-poison']:
        dataset = load_poison(dataset_name, retriever, retrieval_k,num_poison = num_poison)
    else:
        raise NotImplementedError
    return dataset

def generate_incorrect_answer(question,dataset_name):
    if os.path.exists(f"datasets/prompt_injection/{dataset_name}/incorrect_answer_{dataset_name}.json"):
        dataset = load_json(f"datasets/prompt_injection/{dataset_name}/incorrect_answer_{dataset_name}.json")
        if question in dataset:
            return dataset[question]
        else:
            gpt = create_model(f'model_configs/gpt3.5_config.json')
            template =f"Please generate an incorrect answer for this question {question}. Only output the incorrect answer."
            incorrect_answer = gpt.query(template)
            dataset[question] = incorrect_answer
            save_json(dataset, f"datasets/prompt_injection/{dataset_name}/incorrect_answer_{dataset_name}.json")
            return incorrect_answer
    
    if not os.path.exists(f'datasets/prompt_injection/{dataset_name}'):
        os.makedirs(f'datasets/prompt_injection/{dataset_name}')
        gpt = create_model(f'model_configs/gpt4.1-mini_config.json')
        template =f"Please generate an incorrect answer for this question {question}. The incorrect answer should be short and concese. Only output the incorrect answer."
        incorrect_answer = gpt.query(template)
        dataset = {}
        dataset[question] = incorrect_answer
        save_json(dataset, f"datasets/prompt_injection/{dataset_name}/incorrect_answer_{dataset_name}.json")
        return incorrect_answer
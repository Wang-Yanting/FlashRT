import os
import json
import numpy as np
import random
import torch
import re
import torch
from pynvml import *
import time
from contextlib import contextmanager
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
        
def load_results(file_name):
    with open(os.path.join('results', file_name)) as file:
        results = json.load(file)
    return results
def save_json(results, file_path="debug.json"):
    json_dict = json.dumps(results, cls=NpEncoder)
    dict_from_str = json.loads(json_dict)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dict_from_str, f)

def load_json(file_path):
    with open(file_path) as file:
        results = json.load(file)
    return results


def save_results(results, dir, file_name="debug"):
    json_dict = json.dumps(results, cls=NpEncoder)
    dict_from_str = json.loads(json_dict)
    if not os.path.exists(f'results/{dir}'):
        os.makedirs(f'results/{dir}', exist_ok=True)
    with open(os.path.join(f'results/{dir}', f'{file_name}.json'), 'w', encoding='utf-8') as f:
        json.dump(dict_from_str, f)
def read_results(dir, file_name="debug"):
    file_path = os.path.join(f'results/{dir}', f'{file_name}.json')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: '{file_path}'")
    with open(file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results
def _save_results(args,attr_results, pred_results_path):
    if args.dataset_name in ['musique', 'narrativeqa', 'gov_report','nq-poison','hotpotqa-poison','msmarco-poison','nq-poison-combinatorial','nq-poison-insufficient','nq-poison-correctness','nq-poison-hotflip', 'nq-poison-safety','paper','code-completion','ehragent']:
        name = f"{args.prompt_injection_attack}"
    else:
        raise ValueError("Unsupported dataset_name.")

    save_results(attr_results, pred_results_path, name+f"_{args.dataset_name}_{args.model_name}_{args.prompt_injection_attack}_{args.context_right_recompute_ratio}")


def _read_results(args, pred_results_path):
    if args.dataset_name in ['musique', 'narrativeqa', 'gov_report']:
        name = f"{args.prompt_injection_attack}"
    elif args.dataset_name in ['nq-poison','hotpotqa-poison','msmarco-poison','nq-poison-combinatorial','nq-poison-insufficient','nq-poison-correctness','nq-poison-hotflip', 'nq-poison-safety']:
        name = "PoisonedRag"
    elif args.dataset_name in ['srt','mrt']:
        name = "needle_in_haystack"
    else:
        raise ValueError("Unsupported dataset_name.")
    if args.attr_type in ["vanilla_perturb","tracllm"]:
        return read_results( pred_results_path, name+f"_{args.dataset_name}_{args.inject_times}_{args.model_name}_{args.attr_type}_{'_'.join(args.score_funcs)}_{args.avg_k}_{args.K}")
    elif args.attr_type == "attntrace":
        return read_results( pred_results_path, name+f'_{args.dataset_name}_{args.inject_times}_{args.model_name}_{args.attr_type}_{args.avg_k}_{args.q}_{args.B}_{args.K}')
    elif args.attr_type == "self_citation" or "attention" in args.attr_type:
        return read_results( pred_results_path, name+f'_{args.dataset_name}_{args.inject_times}_{args.model_name}_{args.attr_type}_{args.K}')
    else:
        raise ValueError("Unsupported attr_type.")


def setup_seeds(seed):
    # seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def clean_str(s):
    try:
        s=str(s)
    except:
        print('Error: the output cannot be converted to a string')
    s=s.strip()
    if len(s)>1 and s[-1] == ".":
        s=s[:-1]
    return s.lower()
def newline_pad_contexts(contexts):
    return [contexts[0]] + ['\n\n'+context for context in contexts[1:]]
def f1_score(precision, recall):
    """
    Calculate the F1 score given precision and recall arrays.
    
    Args:
    precision (np.array): A 2D array of precision values.
    recall (np.array): A 2D array of recall values.
    
    Returns:
    np.array: A 2D array of F1 scores.
    """
    f1_scores = np.divide(2 * precision * recall, precision + recall, where=(precision + recall) != 0)
    
    return f1_scores

def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")


def find_indices(list1: list, list2: list):
    # 存储结果的列表
    indices = []
    # 遍历list1中的每个元素
    for element in list1:
        # 尝试找到element在list2中的索引
        try:
            index = list2.index(element)
            # 如果找到，将索引添加到结果列表中
            indices.append(index)
        except ValueError:
            # 如果元素不在list2中，跳过
            continue
    return indices

def contexts_to_segments(contexts):
    segment_size = 100
    context = contexts[0]
    words = context.split(' ')

    # Create a list to hold segments
    segments = []
    
    # Iterate over the words and group them into segments
    for i in range(0, len(words), segment_size):
        # Join a segment of 100 words and add to segments list
        segment = ' '.join(words[i:i + segment_size])+' '
        segments.append(segment)
    
    return segments

    
def contexts_to_sentences(contexts):

    all_sentences = split_into_sentences(contexts[0])
    return all_sentences

import re
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'
def split_into_phrases(text: str) -> list[str]:
    sentences = split_into_sentences(text)
    phrases = []
    for sent in sentences:
        phrases+=sent.split(',')
    return phrases
def split_into_sentences(text: str) -> list[str]:
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead 
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    text = " " + text + "  "
    text = text.replace("\n","<newline>")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    sentences = [s.replace("<newline>", "\n") for s in sentences]
    return sentences


def plot_sentence_importance(question, sentences_list, important_ids, importance_values, answer, explained_answer = "", width = 200):
    from rich.console import Console
    from rich.text import Text

    assert len(important_ids) == len(importance_values), "Mismatch between number of words and importance values."
    all_importance_values =np.zeros(len(sentences_list))
    all_importance_values[important_ids] = importance_values
    #print("sentences list: ", sentences_list)
    console = Console(width =width)
    text = Text()
    #print("MIN:",np.min(all_importance_values))
    #print(all_importance_values)
    #all_importance_values = (all_importance_values - np.min(all_importance_values)) / (np.max(all_importance_values) - np.min(all_importance_values)+0.0001)
    all_importance_values = (all_importance_values ) / (np.max(all_importance_values) +0.0001)
    
    text.append("Context:\n", style=f"black bold")
    for i,(sentence, imp) in enumerate(zip(sentences_list, all_importance_values)):

        #sentence = sentence.capitalize()
        red_intensity = 255
        blue_intensity=0
        #print(imp)
        if imp < 0 or imp ==0:
            green_intensity=255
            blue_intensity=255
        else:
            green_intensity = int(255* (1 - imp))

        bg_color = f"{red_intensity:02x}{green_intensity:02x}{blue_intensity:02x}"

        text.append(sentence, style=f"on #{bg_color} black")
    text.append("\nQuery: \n", style=f"black bold")
    red_intensity = 255
    green_intensity=255
    blue_intensity=255

    bg_color = f"{red_intensity:02x}{green_intensity:02x}{blue_intensity:02x}"
    text.append(question, style=f"on #{bg_color} black")
    text.append("\nLLM_response:\n", style=f"black bold")

    answer = answer.capitalize()
    red_intensity = 255
    green_intensity=255
    blue_intensity=255

    bg_color = f"{red_intensity:02x}{green_intensity:02x}{blue_intensity:02x}"
    text.append(answer, style=f"on #{bg_color} black")
    if explained_answer!="":
        text.append("\nExplained part:", style=f"black bold")

        red_intensity = 255
        green_intensity=255
        blue_intensity=255

        bg_color = f"{red_intensity:02x}{green_intensity:02x}{blue_intensity:02x}"
        text.append(explained_answer, style=f"on #{bg_color} black")
    console.print(text)

def unzip_tuples(tuple_list):
    list1 = [t[0] for t in tuple_list]
    list2 = [t[1] for t in tuple_list]
    return list1, list2
def manual_zip(list1, list2):
    # Ensure both lists have the same length
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length")

    combined_list = []
    for i in range(len(list1)):
        combined_list.append((list1[i], list2[i]))
    
    return combined_list
def check_cannot_answer(answer):
    prefixes = ["I don't know"]
    do_not_know = any([prefix in answer for prefix in prefixes])
    print("DO NOT KNOW: ", do_not_know)
    return do_not_know

def top_k_indexes(lst, k):
    # Check if k is greater than the length of the list
    if k > len(lst):
        k = len(lst)
    # Get the indexes of the list sorted by their values in descending order
    sorted_indexes = sorted(range(len(lst)), key=lambda i: lst[i], reverse=True)
    # Return the first k indexes from the sorted list
    return sorted_indexes[:k]

def get_top_k(important_ids, importance_scores, k):
    top_k=top_k_indexes(importance_scores, k)
    topk_ids = [important_ids[j] for j in top_k]
    topk_scores = [importance_scores[j] for j in top_k]
    return topk_ids,topk_scores
def add_specific_indexes(lst, indexes_to_add):
    indexes_to_add = sorted(indexes_to_add)
    return [item for idx, item in enumerate(lst) if idx in indexes_to_add]
def remove_specific_indexes(lst, indexes_to_remove):
    return [item for idx, item in enumerate(lst) if idx not in indexes_to_remove]
def clean_str(s):
    try:
        s=str(s)
    except:
        print('Error: the output cannot be converted to a string')
    s=s.strip()
    if len(s)>1 and s[-1] == ".":
        s=s[:-1]
    return s.lower()
def split_context(level, contexts):
    assert isinstance(contexts, list)
    if len(contexts)>1: #the context is already segmented
        return contexts
    else:
        if level =="sentence":
            all_texts = contexts_to_sentences(contexts)
        elif level =="segment":
            all_texts = contexts_to_segments(contexts)
        elif level =="paragraph":
            all_texts = contexts_to_paragraphs(contexts)
        else:
            raise ValueError("Invalid explanation level.")
    return all_texts

def check_overlap(str1, str2, n):
    len1 = len(str1)
    len2 = len(str2)
    
    if str1 in str2 or str2 in str1:
        return True
    # Check overlap by comparing suffix of str1 with prefix of str2
    for i in range(1, min(len1, len2) + 1):
        if i > n and str1[-i:] == str2[:i]:
            return True
    
    # Check overlap by comparing prefix of str1 with suffix of str2
    for i in range(1, min(len1, len2) + 1):
        if i > n and str1[:i] == str2[-i:]:
            return True
    
    return False

def get_gt_ids(all_texts, injected_adv):
    gt_ids =[]
    gt_texts = []
    for j, segment in enumerate(all_texts):
        for malicious_text in injected_adv:
            if check_overlap(segment,malicious_text,10):
                gt_ids.append(j)
                gt_texts.append(all_texts[j])
    return gt_ids,gt_texts

def min_subset_to_contain(gt_text, texts):
    candidates =[]
    for i in range(len(texts)):
        for j in range(i+1,len(texts)):
            #print("candidate:",''.join(texts[i:j]))
            if gt_text in ''.join(texts[i:j]).replace('  ',' '):
                candidates.append(texts[i:j])
    #print(candidates)
    if len(candidates) >0:
        return min(candidates, key=len)
    else:
        return []

def mean_of_percent(values,percent = 1):
    # Step 1: Sort the list in descending order
    sorted_values = sorted(values, reverse=True)
    
    # Step 2: Determine the number of elements in the top 20%
    top_percent_count = max(1, int(len(sorted_values) * percent))
    print("top_percent_count: ", top_percent_count)
    # Step 3: Extract the top 20% values
    top_values = sorted_values[:top_percent_count]
    # Step 4: Calculate and return the mean of the top 20% values
    if len(top_values) ==0:
        return 0 

    mean_top = sum(top_values) / len(top_values)
    return mean_top

def is_value_in_dicts(dictionary, value_to_check):
    for value in dictionary.values():
        if isinstance(value, (np.ndarray, list)):
            # If value is an array or list, check if any/all elements match
            if np.array_equal(value, value_to_check):  # For numpy arrays
                return True
        else:
            if value == value_to_check:
                return True
    return False


def wait_for_available_gpu_memory(required_memory_gb, device=0, check_interval=5):
    """
    Waits until the required amount of GPU memory is available.
    Args:
    required_memory_gb (float): Required GPU memory in gigabytes.
    device (int): GPU device index (default is 0)
    check_interval (int): Time interval in seconds between memory checks.
    Returns:
    None
    """
    required_memory_bytes = required_memory_gb * 1e9  # Convert GB to bytes
    while True:
        try:
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(device)
            info = nvmlDeviceGetMemoryInfo(handle)
            available_memory = info.free
            if available_memory >= required_memory_bytes:
                print(f"Sufficient GPU memory available: {available_memory / 1e9:.2f} GB")
                nvmlShutdown()
                return
            else:
                print(f"Waiting for GPU memory. Available: {available_memory / 1e9:.2f} GB, Required: {required_memory_gb:.2f} GB")
            nvmlShutdown()
        except NVMLError as error:
            print(f"Error getting GPU memory: {error}")
            # Fallback to PyTorch method
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                total_memory = torch.cuda.get_device_properties(device).total_memory
                allocated_memory = torch.cuda.memory_allocated(device)
                available_memory = total_memory - allocated_memory
                if available_memory >= required_memory_bytes:
                    print(f"Sufficient GPU memory available (PyTorch): {available_memory / 1e9:.2f} GB")
                    return 1
                else:
                    print(f"Waiting for GPU memory (PyTorch). Available: {available_memory / 1e9:.2f} GB, Required: {required_memory_gb:.2f} GB")
            else:
                print("CUDA is not available")
        time.sleep(check_interval)

@contextmanager
def temp_attn_impl(model, impl: str = "sdpa"):
    """
    Temporarily switch a HF model's attention implementation.
    Works across HF versions by touching config, core module, and per-layer attrs.
    impl ∈ {"sdpa", "flash_attention_2"}.
    Usage:
        with temp_attn_impl(model, "sdpa"):
            out = model.model(...)
    """
    core = getattr(model, "model", model)  # e.g., LlamaForCausalLM.model
    cfgs = [getattr(model, "config", None), getattr(core, "config", None)]

    # Snapshot old values
    old_model_impl = (
        getattr(core, "attn_implementation", None)
        if hasattr(core, "attn_implementation")
        else getattr(core, "_attn_implementation", None)
    )
    old_cfg_vals = []
    for cfg in cfgs:
        if cfg is None:
            old_cfg_vals.append(None)
        else:
            old_cfg_vals.append((
                getattr(cfg, "attn_implementation", None),
                getattr(cfg, "_attn_implementation", None),
            ))
    old_layer_impls = []
    for layer in getattr(core, "layers", []):
        old_layer_impls.append(getattr(layer.self_attn, "attn_implementation", None))

    # Try official setter if present (some models expose it)
    if hasattr(model, "set_default_attn_implementation"):
        try:
            model.set_default_attn_implementation(impl)
        except Exception:
            pass

    # Set config-level flags
    for cfg in cfgs:
        if cfg is None: continue
        if hasattr(cfg, "attn_implementation"):
            cfg.attn_implementation = impl
        if hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = impl

    # Set core module flag
    if hasattr(core, "attn_implementation"):
        core.attn_implementation = impl
    if hasattr(core, "_attn_implementation"):
        core._attn_implementation = impl

    # Set per-layer flags
    for layer in getattr(core, "layers", []):
        if hasattr(layer.self_attn, "attn_implementation"):
            layer.self_attn.attn_implementation = impl

    # Optional: nudge PyTorch SDPA kernel prefs when switching to sdpa
    sdp_ctx = None
    try:
        if impl == "sdpa":
            import torch
            sdp_ctx = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
            sdp_ctx.__enter__()
    except Exception:
        sdp_ctx = None  # ignore if not available

    try:
        yield
    finally:
        # Restore per-layer flags
        for layer, prev in zip(getattr(core, "layers", []), old_layer_impls):
            if hasattr(layer.self_attn, "attn_implementation"):
                layer.self_attn.attn_implementation = prev
        # Restore core flags
        if hasattr(core, "attn_implementation"):
            core.attn_implementation = old_model_impl
        if hasattr(core, "_attn_implementation"):
            core._attn_implementation = old_model_impl
        # Restore config flags
        for cfg, prev in zip(cfgs, old_cfg_vals):
            if cfg is None or prev is None: continue
            ai, _ai = prev
            if hasattr(cfg, "attn_implementation"):
                cfg.attn_implementation = ai
            if hasattr(cfg, "_attn_implementation"):
                cfg._attn_implementation = _ai
        # Exit SDPA kernel override
        if sdp_ctx is not None:
            sdp_ctx.__exit__(None, None, None)
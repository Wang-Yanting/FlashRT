from .GPT import GPT
from .Llama import Llama
from .HF_model import HF_model
from .Deepseek import Deepseek
from .SecAlign import SecAlign
from .LlamaGuard import LlamaGuard
from .PromptGuard import PromptGuard
from .Code_model import Code_model
import json

def load_json(file_path):
    with open(file_path) as file:
        results = json.load(file)
    return results

def create_model(config_path = None, model_path = None, api_key = None, device = "cuda:0"):
    """
    Factory method to create a LLM instance, the user can use either a config_file or model_name+api_key to specify the model.
    """

    if config_path!=None:
        config = load_json(config_path)
    elif model_path != None and api_key != None:
        config = { 
        "model_info":{
            "provider":None,
            "name": model_path
        },
        "api_key_info":{
            "api_keys":[
                api_key
            ],
            "api_key_use": 0
        },
        "params":{
            "temperature":0.001,
            "max_output_tokens":100
        }
    }
    else:
        raise ValueError("ERROR: Either config_path or both model_name and api_key must be provided")
    
    name = config["model_info"]["name"].lower()
    if 'gpt' in name:
        model = GPT(config)
    elif 'code' in name:
        model = Code_model(config,device)
    elif 'llama-guard' in name:
        model = LlamaGuard(config,device)
    elif 'prompt-guard' in name:
        model = PromptGuard(config,device)
    elif 'llama' in name:
        model = Llama(config,device)
    elif 'secalign' in name:
        model = SecAlign(config,device)

    else:
        model = HF_model(config,device)
    return model

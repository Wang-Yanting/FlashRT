import torch
from transformers import  AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from .Model import Model
import os
import signal
import math
from src.util.utils import temp_attn_impl
def handle_timeout(sig, frame):
    raise TimeoutError('took too long')
signal.signal(signal.SIGALRM, handle_timeout)

class SecAlign(Model):
    def __init__(self, config, device = "cuda:0"):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])

        hf_token = os.environ.get("HF_TOKEN", "")
        is_70b = "70" in self.name
        if is_70b:
            base_repo = "meta-llama/Llama-3.1-70B-Instruct"
            adapter_repo = "facebook/Meta-SecAlign-70B"
        else:
            base_repo = "meta-llama/Llama-3.1-8B-Instruct"
            adapter_repo = "facebook/Meta-SecAlign-8B"
        self.tokenizer = AutoTokenizer.from_pretrained(base_repo, use_auth_token=hf_token)
        print("Start downloading model!")
        model = AutoModelForCausalLM.from_pretrained(
            base_repo,
            torch_dtype=torch.bfloat16,
            device_map=device,
            #attn_implementation="flash_attention_2",
            use_auth_token=hf_token
        )
        print("load secalign!")
        self.model = PeftModel.from_pretrained(model, adapter_repo, token=hf_token, torch_dtype=torch.bfloat16, device_map=device)
        # Check if self.model is a PEFT model
        if isinstance(self.model, PeftModel):
            print("self.model is a PEFT model.")
        else:
            print("self.model is NOT a PEFT model.")
        for p in model.base_model.parameters():
            p.requires_grad = False
        
        print('Download complete!', flush = True)
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        self.messages = [
        {"role": "user",   "content": "You are a helpful assistant. Your answer should be short and consice."},
        {"role": "input",  "content": " "}
    ]
    def query(self, msg, max_tokens=128000):

        messages = self.messages
        messages[1]["content"] = msg

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)
        attention_mask = torch.ones(input_ids.shape, device=self.model.device)
        try:
            signal.alarm(60)

            output_tokens = self.model.generate(
                input_ids,
                max_length=max_tokens,
                attention_mask=attention_mask,
                eos_token_id=self.terminators,
                top_k=50,
                do_sample=False
            )
            signal.alarm(0)
        except TimeoutError as exc:
            print("time out")
            return("time out")
        # Decode the generated tokens back to text
        result = self.tokenizer.decode(output_tokens[0][input_ids.shape[-1]:], skip_special_tokens=True)

        return result


    def get_prompt_length(self,msg):
        messages = self.messages
        messages[1]["content"] = msg
        # Get tokenized output as text first, then tokenize to avoid tensor/device issues
        try:
            # Try with return_tensors first
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                padding=False,
                truncation=False
            )
            if isinstance(input_ids, torch.Tensor):
                return len(input_ids[0])
            else:
                # If it's a list, get length from first element
                return len(input_ids[0]) if isinstance(input_ids[0], (list, torch.Tensor)) else len(input_ids)
        except (ValueError, RuntimeError) as e:
            # Fallback: tokenize without tensors
            formatted_text = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            tokens = self.tokenizer.encode(formatted_text, add_special_tokens=False)
            return len(tokens)
    def cut_context(self,msg,max_length):
        tokens = self.tokenizer.encode(msg, add_special_tokens=True)

        # Truncate the tokens to a maximum length
        truncated_tokens = tokens[:max_length]

        # Decode the truncated tokens back to text
        truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        return truncated_text
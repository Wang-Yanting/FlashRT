import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .Model import Model
class HF_model(Model):
    def __init__(self, config, device="cuda:0"):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])

        hf_token = os.environ.get("HF_TOKEN", "")
        self.tokenizer = AutoTokenizer.from_pretrained(self.name, use_auth_token=hf_token, trust_remote_code=True)
        print('Start downloading model from Hugging Face!')
        self.model = AutoModelForCausalLM.from_pretrained(
            self.name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            use_auth_token=hf_token,
            trust_remote_code=True
        )

        print('Download complete!', flush = True)
    def query(self, msg, max_tokens=128000):    
        messages = self.messages
        if len(messages) == 1:
            messages[0]["content"] = msg
        else:
            messages[1]["content"] = msg
        
        # Check if tokenizer has a chat template
        if self.tokenizer.chat_template is None:
            # For models without chat template (e.g., base CodeLlama), use the user content directly
            text = messages[0]["content"] if len(messages) == 1 else messages[1]["content"]
        else:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
       
            # The original code attempted to add a list of integers to a tensor, which is invalid.
            # If you meant to append specific token IDs to the input_ids tensor, use torch.cat:
            # e.g., append [200005, 35644, 200008] as new tokens

            #extra_tokens = torch.tensor([[200005, 35644, 200008]], device=model_inputs.input_ids.device)
            #model_inputs.input_ids = torch.cat([model_inputs.input_ids, extra_tokens], dim=1)
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=self.max_output_tokens,
            temperature=self.temperature,
            output_scores=True
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def get_prompt_length(self,msg):
        messages = self.messages
        if len(messages) == 1:
            messages[0]["content"] = msg
        else:
            messages[1]["content"] = msg
        
        # Check if tokenizer has a chat template
        if self.tokenizer.chat_template is None:
            # For models without chat template (e.g., base CodeLlama), tokenize the user content directly
            text = messages[0]["content"] if len(messages) == 1 else messages[1]["content"]
            input_ids = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        else:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)
        return len(input_ids[0])

    def cut_context(self, msg, max_length):
        tokens = self.tokenizer.encode(msg, add_special_tokens=True)
        truncated_tokens = tokens[:max_length]
        truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        return truncated_text
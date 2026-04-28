import torch
from transformers import  AutoTokenizer, AutoModelForCausalLM

from .Model import Model
import os
import signal
import math
from src.util.utils import temp_attn_impl
def handle_timeout(sig, frame):
    raise TimeoutError('took too long')
signal.signal(signal.SIGALRM, handle_timeout)

class LlamaGuard(Model):
    def __init__(self, config, device = "cuda:0"):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])

        hf_token = os.environ.get("HF_TOKEN", "")

        print("Start downloading model!")

        self.guard_tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.guard_model = AutoModelForCausalLM.from_pretrained(self.name, torch_dtype=torch.bfloat16, device_map=device)

        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", use_auth_token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map=device,
            #attn_implementation="flash_attention_2",
            use_auth_token=hf_token
        )
        print('Download complete!', flush = True)
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    def moderate(self,msgs):

        chat = msgs[1:] #remove system message

        input_ids = self.guard_tokenizer.apply_chat_template(chat, return_tensors="pt").to(self.guard_model.device)
        output = self.guard_model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
        prompt_len = input_ids.shape[-1]
        return self.guard_tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

    def get_guard_logprob(self, msgs):
        # Prepares the input following the llama-guard-3 chat template
        chat = msgs[1:] #remove system message

        input_ids = self.guard_tokenizer.apply_chat_template(chat, return_tensors="pt").to(self.guard_model.device)
        prompt_len = input_ids.shape[-1]

        # token id(s) for candidate outputs (e.g., "safe" or "unsafe")
        safe_token = self.guard_tokenizer("safe", add_special_tokens=False)["input_ids"]
        unsafe_token = self.guard_tokenizer("unsafe", add_special_tokens=False)["input_ids"]

        # Forward pass to get log probabilities
        with torch.no_grad():
            outputs = self.guard_model(input_ids=input_ids)
            logits = outputs.logits  # shape [1, seq_len, vocab]

        # Get logprobs at the first generation step after prompt
        next_token_logits = logits[0, prompt_len-1, :]  # last prompt token, next-token prediction
        logprobs = torch.log_softmax(next_token_logits, dim=-1)

        # Compute logprob of "safe" (may be multi-token, so sum logprobs for each token in sequence)
        current_input = input_ids[0].tolist()
        logprob_safe = 0.0
        for idx, token_id in enumerate(safe_token):
            if idx == 0:
                logprob_safe += logprobs[token_id].item()
                # Prepare for next-token prediction
                next_inputs = input_ids[0].unsqueeze(0)
                generated = torch.tensor([[token_id]], device=self.guard_model.device)
            else:
                # Predict next token
                output = self.guard_model(
                    input_ids=torch.cat([next_inputs, generated], dim=-1)
                )
                logits = output.logits
                logprobs = torch.log_softmax(logits[0, -1], dim=-1)
                logprob_safe += logprobs[token_id].item()
                generated = torch.cat([generated, torch.tensor([[token_id]], device=self.guard_model.device)], dim=-1)

        return logprob_safe

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
        guardrail_msg = messages.copy()
        guardrail_msg.append({"role": "assistant", "content": result})
        guard_logprob = self.get_guard_logprob(guardrail_msg)
        guard_output = self.moderate(guardrail_msg)
        print("guard_output: ", guard_output)
        print("logprob of safe: ", guard_logprob)
        if "unsafe" in guard_output:
            return guard_output
        return result


    def get_prompt_length(self,msg):
        messages = self.messages
        messages[1]["content"] = msg
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        return len(input_ids[0])
    def cut_context(self,msg,max_length):
        tokens = self.tokenizer.encode(msg, add_special_tokens=True)

        # Truncate the tokens to a maximum length
        truncated_tokens = tokens[:max_length]

        # Decode the truncated tokens back to text
        truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        return truncated_text
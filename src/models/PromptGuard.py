import torch
from transformers import  AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

from .Model import Model
import os
import signal
import math
from src.util.utils import temp_attn_impl
import re
import random
def handle_timeout(sig, frame):
    raise TimeoutError('took too long')
signal.signal(signal.SIGALRM, handle_timeout)

class PromptGuard(Model):
    def __init__(self, config, device = "cuda:0"):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])

        hf_token = os.environ.get("HF_TOKEN", "")

        print("Start downloading model!")

        self.guard_tokenizer = AutoTokenizer.from_pretrained(self.name, use_auth_token=hf_token)
        self.guard_model = AutoModelForSequenceClassification.from_pretrained(self.name, use_auth_token=hf_token)
        self.guard_model = self.guard_model.to("cuda:0")

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
    def split_to_300_word_segments(self,text):
        words = re.findall(r'\S+', text)
        segments = []
        for i in range(0, len(words), 300):
            segment = ' '.join(words[i:i+300])
            segments.append(segment)
        return segments
    def moderate(self,msg):
        # Split the message into segments of up to 200 words each
        

        msg_segments = self.split_to_300_word_segments(msg)
        safe = True
        for segment in msg_segments:
            inputs = self.guard_tokenizer(segment, return_tensors="pt").to(self.guard_model.device)

            with torch.no_grad():
                logits = self.guard_model(**inputs).logits
            predicted_class_id = logits.argmax().item()
            predicted_label = self.guard_model.config.id2label[predicted_class_id]
            #print("predicted_label: ", self.guard_model.config.id2label[predicted_class_id])
            if "LABEL_1" in predicted_label:
                safe = False
                break
        if safe:
            return "safe"
        else:
            return "unsafe"
       

    def get_guard_logprob(self, msg,payload):
        # Prepares the input following the llama-guard-3 chat template

        msg_segments = self.split_to_300_word_segments(msg)
        logprob = -0.1
        for segment in msg_segments:
            if payload not in segment:
                continue
            inputs = self.guard_tokenizer(segment, return_tensors="pt").to(self.guard_model.device)

            with torch.no_grad():
                logits = self.guard_model(**inputs).logits
            predicted_class_id = logits.argmax().item()
            predicted_label = self.guard_model.config.id2label[predicted_class_id]
            #print("predicted_label: ", self.guard_model.config.id2label[predicted_class_id])
            logit_0 = logits[0][0].item()  # logits shape: (1, n_labels); index 0 for LABEL_0
            logprob= torch.log_softmax(logits, dim=1)[0][0].item()
            if random.choice(range(10))==1:
                print("logprob_0: ", logprob)
            return logprob
        if logprob ==-0.1:
            total_logprob = 0
            for segment in msg_segments:
                inputs = self.guard_tokenizer(segment, return_tensors="pt").to(self.guard_model.device)
                with torch.no_grad():
                    logits = self.guard_model(**inputs).logits
                predicted_class_id = logits.argmax().item()
                predicted_label = self.guard_model.config.id2label[predicted_class_id]
                #print("predicted_label: ", self.guard_model.config.id2label[predicted_class_id])
                if "LABEL_1" in predicted_label:
                    logit_0 = logits[0][0].item()  # logits shape: (1, n_labels); index 0 for LABEL_0
                    logprob= torch.log_softmax(logits, dim=1)[0][0].item()

                    total_logprob += logprob
            if random.choice(range(10))==1:
                print("total_logprob: ", total_logprob)
            return total_logprob
        return -0.1




    def query(self, msg, max_tokens=128000):
        guard_output = self.moderate(msg)
        print("guard_output: ", guard_output)
        if "unsafe" in guard_output:
            return guard_output
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
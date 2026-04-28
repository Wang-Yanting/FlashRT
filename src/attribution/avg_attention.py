
import numpy as np
from src.util.utils import *
import time
import torch.nn.functional as F
import gc
from .attention_utils import *
from src.util.utils import temp_attn_impl
from peft import PeftModel
class AvgAttentionAttribution:
    def __init__(self,  llm,ratio = 0.1, verbose =1):
        
        self.model = llm.model # Use float16 for the model
        self.tokenizer = llm.tokenizer

        self.layers = [15,16,17,18,19]

        #self.layers = range(len(llm.model.model.layers))
        self.variant = "attention_output"
        self.ratio = ratio

    def attribute_segment(self, prompt_ids,target_ids,instruction_slice,segment_size = 50):
        if self.variant == "attention_output":
            return self.attribute_segment_output(prompt_ids,target_ids,instruction_slice,segment_size = segment_size)
        elif self.variant == "attention_instruction":
            return self.attribute_segment_instruction(prompt_ids,target_ids,instruction_slice,segment_size = segment_size)
  
 
    def attribute_segment_output(self, prompt_ids,target_ids,instruction_slice,segment_size = 50):
        start_time = time.time()
        model = self.model
        tokenizer = self.tokenizer
        model.eval()  # Set model to evaluation mode
        
        self.prompt_length = len(prompt_ids)

        prompt_ids = torch.tensor(prompt_ids, device=model.device)
        target_ids = torch.tensor(target_ids, device=model.device)
        #except Exception as e:
        #    prompt_ids = torch.tensor(prompt_ids, device="cuda:0")
        #    target_ids = torch.tensor(target_ids, device="cuda:0")
        
        with torch.no_grad():
            #with temp_attn_impl(model, "flash_attention_2"):
            
            outputs = model(torch.cat([prompt_ids.unsqueeze(0), target_ids.unsqueeze(0)], dim=-1), output_hidden_states=True)
            
            
        hidden_states = outputs.hidden_states
        end_time = time.time()
        #print("forward pass time attribution: ", end_time - start_time)
        start_time = time.time()
        with torch.no_grad():
            batch_size = 1  # Process 4 layers at a time
            avg_attentions = None  # Initialize to None for accumulative average
            for i in self.layers:
                if isinstance(model, PeftModel):
                    attentions = get_attention_weights_one_layer(model.base_model.model, hidden_states, i, attribution_start=self.prompt_length)
                else:
                    attentions = get_attention_weights_one_layer(model, hidden_states, i, attribution_start=self.prompt_length)
                batch_mean = attentions
                if avg_attentions is None:
                    avg_attentions = batch_mean[:, :, :, :]
                else:
                    # Ensure both tensors are on the same device before accumulation
                    if avg_attentions.device != batch_mean.device:
                        batch_mean = batch_mean.to(avg_attentions.device)
                    avg_attentions += batch_mean[:, :, :, :]
            avg_attentions = (avg_attentions / (len(self.layers) / batch_size)).mean(dim=0).mean(dim=(0, 1)).to(torch.float16)
        end_time = time.time()
        #print("attention calculation time attribution: ", end_time - start_time)

        start_time = time.time()
        # Convert attention scores to importance values
        importance_values = avg_attentions.to(torch.float32).cpu().numpy()
        #print(importance_values.shape)
        #print(instruction_slice.stop)
        # Divide into segments and get segment scores
        
        num_segments = math.ceil(len(importance_values[instruction_slice.stop:]) / segment_size)
        
        segment_scores = []
        
        for seg_idx in range(num_segments):
            start_idx = instruction_slice.stop+seg_idx * segment_size
            end_idx = min(instruction_slice.stop+(seg_idx + 1) * segment_size, len(importance_values))
            segment_score = importance_values[start_idx:end_idx].mean()
            segment_scores.append((start_idx, end_idx, segment_score))
        #print(segment_scores)
        # Sort segments by importance
        segment_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Select top segments based on ratio
        num_top_segments = max(1, int(self.ratio * num_segments))

        selected_segments = segment_scores[:num_top_segments]
        
        # Get all token indices from selected segments
        top_k_indices = []
        for start_idx, end_idx, _ in selected_segments:
            top_k_indices.extend(range(start_idx, end_idx))
        
        end_time = time.time()
        #print("importance values calculation time attribution: ", end_time - start_time)
        return top_k_indices, importance_values, end_time - start_time, None


    def attribute_segment_instruction(self, prompt_ids, target_ids,instruction_slice):
        start_time = time.time()
        model = self.model
        tokenizer = self.tokenizer
        model.eval()  # Set model to evaluation mode
        
        self.prompt_length = len(prompt_ids)
        prompt_ids = torch.tensor(prompt_ids, device=model.device)
        
        with torch.no_grad():
            outputs = model(prompt_ids.unsqueeze(0), output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        # Get start and end indices of instruction slice
        slice_start, slice_end = instruction_slice.start, instruction_slice.stop
        
        with torch.no_grad():
            batch_size = 1
            avg_attentions = None
            for i in self.layers:
                # Get attention weights for the whole prompt
                attentions = get_attention_weights_one_layer(model, hidden_states, i, attribution_start=slice_start, attribution_end=slice_end, reverse = True)
                
                # Zero out attention before instruction slice
                #print(attentions.shape)
                batch_mean = attentions
                if avg_attentions is None:
                    avg_attentions = batch_mean[:, :, :, :]
                else:
                    avg_attentions += batch_mean[:, :, :, :]
                    
            avg_attentions = (avg_attentions / (len(self.layers) / batch_size)).mean(dim=(0, 1,3)).to(torch.float16)
            
        gc.collect()
        torch.cuda.empty_cache()
        # Convert attention scores to importance values
        importance_values = avg_attentions.to(torch.float32).cpu().numpy()
        importance_values[:slice_start]=0
        # Divide into segments and get segment scores
        segment_size = 50
        num_segments = math.ceil(len(importance_values[slice_end:]) / segment_size)
        segment_scores = []
        
        for seg_idx in range(num_segments):
            start_idx = seg_idx * segment_size+slice_end
            end_idx = min((seg_idx + 1) * segment_size+slice_end, len(importance_values))
            segment_score = importance_values[start_idx:end_idx].mean()
            segment_scores.append((start_idx, end_idx, segment_score))
        #print(segment_scores)
        # Sort segments by importance
        segment_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Select top segments based on ratio
        num_top_segments = max(1, int(self.ratio * num_segments))
        selected_segments = segment_scores[:num_top_segments]
        
        # Get all token indices from selected segments
        top_k_indices = []
        for start_idx, end_idx, _ in selected_segments:
            top_k_indices.extend(range(start_idx, end_idx))
        
        end_time = time.time()
        gc.collect()
        torch.cuda.empty_cache()
        #print(f"Topk_indices: {top_k_indices}")
       # print(f"num_top_segments: {num_top_segments}")
        return top_k_indices, importance_values, end_time - start_time, None
from src.models import create_model
import os
from src.util.utils import *
class Attacker:
    def __init__(self, args):
        self.attack_strategy = args.prompt_injection_attack
        self.model_name = args.model_name
        self.benchmark = args.benchmark
    def inject(self, clean_data, query = None):
        raise NotImplementedError("ERROR: Interface doesn't have the implementation for inject")

    def get_injected_prompt(self):
        return [self.inject_data]

class OptimizationAttacker(Attacker):
    def __init__(self,args):
        super().__init__(args)

    def insert_malicious_instruction(self,clean_data,adv,query,payload,target_answer,position = "mid"):
        if self.benchmark != "PoisonedRAG":
            all_sentences = contexts_to_sentences([clean_data])
        else:
            all_sentences = clean_data.split("\n\n")
            all_sentences = ["\n\n"+sentence for sentence in all_sentences]
        if position == "mid":
            num_sentences = len(all_sentences)
            inject_position = int(num_sentences/2)
            context_left = ''.join(all_sentences[:inject_position])
            context_right = ''.join(all_sentences[inject_position:])
        elif position == "start":
            context_left = ''
            context_right = ''.join(all_sentences) #add \n\n to mimic the first retrieved document for RAG
        elif position == "end":
            context_left = ''.join(all_sentences)
            context_right = ''
        elif position == "one_quater":
            context_left = ''.join(all_sentences[:int(len(all_sentences)/4)])
            context_right = ''.join(all_sentences[int(len(all_sentences)/4):])
        elif position == "three_quater":
            context_left = ''.join(all_sentences[:int(len(all_sentences)*3/4)])
            context_right = ''.join(all_sentences[int(len(all_sentences)*3/4):])

        return context_left+adv[0]+payload+adv[1]+context_right,context_left,context_right,payload

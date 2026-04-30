
from .NanoGCGAttacker import NanoGCGAttacker
from .NanoGCGPlusAttacker import NanoGCGPlusAttacker
from .FlashRTAttacker import FlashRTAttacker
from .ContextClippingAttacker import ContextClippingAttacker
from .AutoDANFlashRTAttacker import AutoDANFlashRTAttacker

def create_attacker(args, model):
    attack_strategy = args.prompt_injection_attack
    if attack_strategy == 'nano_gcg':
        return NanoGCGAttacker(args, model)
    elif attack_strategy == 'nano_gcg_plus':
        return NanoGCGPlusAttacker(args, model)
    elif attack_strategy == 'flash_rt':
        return FlashRTAttacker(args, model)
    elif attack_strategy == "context_clipping":
        return ContextClippingAttacker(args, model)
    elif attack_strategy == "autodan_flashrt":
        return AutoDANFlashRTAttacker(args, model)
    err_msg = f"{attack_strategy} is not a valid attack strategy."
    raise ValueError(err_msg)

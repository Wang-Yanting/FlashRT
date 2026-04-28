import os
import time
import glob

import torch

print("GPUs available:", torch.cuda.device_count())
if torch.cuda.device_count() == 0:
    mode = "slurm"
else:
    mode = "local"

total_jobs = 0
gpu_count = 0
GPU_INTERVAL_MINUTES = 100

gpus = [0] if mode == "local" else [0]

# Models that won't fit on a single GPU — run with CUDA_VISIBLE_DEVICES spanning
# all GPUs and let HuggingFace's device_map="auto" shard them.
MULTI_GPU_MODELS = {"secalign-70b", "llama3.1-70b"}
all_gpus = list(range(torch.cuda.device_count())) if mode == "local" else [0]


def cuda_visible_for(model_name, gpu_id):
    if model_name in MULTI_GPU_MODELS:
        return ",".join(str(g) for g in all_gpus)
    return str(gpu_id)


def run(dataset_name,
        model_name,
        prompt_injection_attack,
        data_num,
        max_length,
        inject_position,
        context_right_recompute_ratio,
        heuristic,
        restart_number,
        data_start, segment_size, log_file):

    global gpu_count, total_jobs

    if mode == "local" and gpu_count == 0 and total_jobs > 0:
        print(f"\n[run.py] Waiting {GPU_INTERVAL_MINUTES} minutes before next GPU cycle...")
        time.sleep(GPU_INTERVAL_MINUTES * 60)
        print(f"[run.py] Resuming job submission\n")

    gpu_id = gpus[gpu_count]
    gpu_count = (gpu_count + 1) % len(gpus)

    if mode == "local":
        cmd = (
            f"(export CUDA_VISIBLE_DEVICES={cuda_visible_for(model_name, gpu_id)} && "
            f"python -u main.py "
            f"  --dataset_name \"{dataset_name}\" "
            f"  --prompt_injection_attack \"{prompt_injection_attack}\" "
            f"  --model_name \"{model_name}\" "
            f"  --position \"{inject_position}\" "
            f"  --context_right_recompute_ratio {context_right_recompute_ratio} "
            f"  --heuristic \"{heuristic}\" "
            f"  --retrieval_k {restart_number} "
            f"  --data_start {data_start} "
            f"  --max_length {max_length} "
            f"  --segment_size {segment_size} "
            f"  --verbose 0 "
            f"  --data_num {data_num} "
            f"  > \"{log_file}\" 2>&1) &"
        )
    elif mode == "slurm":
        cmd = (
            f"sbatch run.sh \"{dataset_name}\" \"{prompt_injection_attack}\" "
            f"\"{model_name}\" \"{inject_position}\" {context_right_recompute_ratio} "
            f"\"{heuristic}\" \"{restart_number}\" \"{data_start}\" \"{data_num}\" "
            f"\"{max_length}\" \"{segment_size}\" \"{log_file}\""
        )

    print(cmd)
    os.system(cmd)
    return 1

dataset_names = ['musique']
prompt_injection_attack = 'flash_rt'
model_names = ['llama3.1-8b']
data_num = 50
context_right_recompute_ratio = 0.2
segment_size = 50
max_length = 15000
heuristics = ["simple"]
data_start = 0

for dataset_name in dataset_names:
    for model_name in model_names:

        if dataset_name in ["narrativeqa", "musique", "gov_report"]:
            inject_position = "mid"
        else:
            inject_position = "start"

        n_retrieval_k = 100
        for heuristic in heuristics:
            log_dir = f'./log/{model_name}/{dataset_name}/{inject_position}/{prompt_injection_attack}'
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            log_file = (
                f'{log_dir}/{dataset_name}_{data_start}_{model_name}_{prompt_injection_attack}'
                f'_{context_right_recompute_ratio}_{heuristic}_{n_retrieval_k}_{max_length}_{segment_size}.txt'
            )
            total_jobs += run(
                dataset_name,
                model_name,
                prompt_injection_attack,
                data_num,
                max_length,
                inject_position,
                context_right_recompute_ratio,
                heuristic,
                n_retrieval_k,
                data_start,
                segment_size,
                log_file=log_file,
            )

print(f"Started {total_jobs} jobs in total")

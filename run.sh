#!/bin/bash
#SBATCH --job-name=pytorch
#SBATCH --account=bfzb-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --mem=120g
#SBATCH --time=48:00:00
# 参数解析
DATASET_NAME=$1
PROMPT_INJECTION_ATTACK=$2
MODEL_NAME=$3
INJECT_POSITION=$4
CONTEXT_RIGHT_RECOMPUTE_RATIO=$5
HEURISTIC=$6
N_RETRIEVAL=$7
DATA_START=$8   
DATA_NUM=$9
MAX_LENGTH="${10}"
SEGMENT_SIZE="${11}"
LOG_FILE="${12}"

# 加载环境
conda activate fast_prompt_new
export HF_HOME=/work/nvme/bfzb/hf_cache
export HUGGINGFACE_HUB_CACHE="$HF_HOME"

# Load credentials from .env (same directory as this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi
echo "job is starting on `hostname`"
# 后台执行并写入日志
python -u main.py \
  --dataset_name "${DATASET_NAME}" \
  --prompt_injection_attack "${PROMPT_INJECTION_ATTACK}" \
  --model_name "${MODEL_NAME}" \
  --position "${INJECT_POSITION}" \
  --context_right_recompute_ratio "${CONTEXT_RIGHT_RECOMPUTE_RATIO}" \
  --heuristic "${HEURISTIC}" \
  --retrieval_k "${N_RETRIEVAL}" \
  --data_start "${DATA_START}" \
  --max_length "${MAX_LENGTH}" \
  --segment_size "${SEGMENT_SIZE}" \
  --verbose 0 \
  --data_num "${DATA_NUM}" \
  2>&1 | tee "${LOG_FILE}"



#!/bin/bash
#SBATCH --job-name=repnoise
#SBATCH --output=artifacts/tamia/logs/run-%j.out
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=03:00:00
set -euo pipefail

# ============================================
# CLUSTER SPECIFIC CONFIG
# ============================================
export PATH="$HOME/.local/bin:$PATH"
export XDG_CACHE_HOME="$SCRATCH/.cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export UV_CACHE_DIR="$PWD/artifacts/tamia/uv-cache"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"

# Tamia compute nodes use cached packages, models, and datasets.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_OFFLINE=1
export UV_LINK_MODE=copy
export TOKENIZERS_PARALLELISM=false
module load python/3.10

# ============================================
# APPLICATION SPECIFIC CONFIG
# ============================================
base_model=${1:-meta-llama/Llama-2-7b-chat-hf}
model_name=${2:-llama2-7b-beavertail--align-fft-repnoise--upstream-paper-config--def-sp}
defence_dir="artifacts/tamia/models/$model_name"
results_dir="artifacts/tamia/evals/$model_name"
attack_learning_rates=(5e-5 8e-5 1e-4 2e-4)

uv venv --python "$(command -v python)" "$UV_PROJECT_ENVIRONMENT"
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" -c requirements.txt \
    torch transformers datasets accelerate pandas numpy sentencepiece protobuf lm_eval

# Defence: paper settings are the defaults of representation_noising.train.
export CUDA_VISIBLE_DEVICES=0,1,2,3
if [[ ! -f "$defence_dir/.trained" ]]; then
    "$UV_PROJECT_ENVIRONMENT/bin/python" -m representation_noising.train \
        --model "$base_model" --output-dir "$defence_dir"
    touch "$defence_dir/.trained"
fi

# Evaluate the defended model before the attacks.
if [[ ! -f "$results_dir/results.json" ]]; then
    "$UV_PROJECT_ENVIRONMENT/bin/python" -m representation_noising.evaluation \
        --model "$defence_dir" --results-path "$results_dir/results.json"
fi

# Each GPU trains and evaluates one attack, keeping its model afterwards.
mkdir -p artifacts/tamia/logs
pids=()
for gpu in 0 1 2 3; do
    attack_lr=${attack_learning_rates[gpu]}
    attack_name="$model_name-att-sp--relearn-fft-lr$attack_lr-ms250"
    attack_dir="artifacts/tamia/models/$attack_name"
    attack_results_dir="artifacts/tamia/evals/$attack_name"
    (
        export CUDA_VISIBLE_DEVICES=$gpu
        if [[ ! -f "$attack_dir/.trained" ]]; then
            "$UV_PROJECT_ENVIRONMENT/bin/python" -m representation_noising.train \
                --model "$defence_dir" --output-dir "$attack_dir" --attack \
                --size 1000 --batch-size 4 --learning-rate "$attack_lr" --device-map auto
            touch "$attack_dir/.trained"
        fi
        if [[ ! -f "$attack_results_dir/results.json" ]]; then
            "$UV_PROJECT_ENVIRONMENT/bin/python" -m representation_noising.evaluation \
                --model "$attack_dir" --results-path "$attack_results_dir/results.json"
        fi
    ) >> "artifacts/tamia/logs/$attack_name.log" 2>&1 &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
exit "$failed"

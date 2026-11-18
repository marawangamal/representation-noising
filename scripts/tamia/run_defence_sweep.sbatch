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
#SBATCH --array=0-8
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
export AXOLOTL_DO_NOT_TRACK=1

# ============================================
# APPLICATION SPECIFIC CONFIG
# ============================================
# Inputs: DEFENCE_MODEL_NAME [DEFENCE_SEPARATOR [BASE_MODEL_NAME_OR_PATH [ATTACK_CONFIG]]]
defence_model_name=${1:?usage: run_defence_sweep.sbatch DEFENCE_MODEL_NAME [DEFENCE_SEPARATOR [BASE_MODEL_NAME_OR_PATH [ATTACK_CONFIG]]]}
defence_separator=${2:-' '}
base_model_name_or_path=${3:-meta-llama/Llama-2-7b-chat-hf}
attack_config=${4:-config/llama2-7b--beavertails--align-fft-repnoise--attack-fft.yml}
defence_alphas=(0.1 1.0 10.0) # Harmful response suppression (default=1.0)
defence_betas=(0.0001 0.001 0.01) # Representation noise strength (default=0.001)
sweep_index=${SLURM_ARRAY_TASK_ID:?submit with sbatch}
defence_alpha=${defence_alphas[sweep_index / ${#defence_betas[@]}]}
defence_beta=${defence_betas[sweep_index % ${#defence_betas[@]}]}

case "$defence_separator" in
    ' ') defence_separator_abbrv=sp ;;
    $'\n') defence_separator_abbrv=nl ;;
    *) echo "Defence separator must be a space or newline" >&2; exit 1 ;;
esac
defence_model_name="$defence_model_name--def-$defence_separator_abbrv--alpha$defence_alpha--beta$defence_beta"

defence_dir="artifacts/tamia/models/$defence_model_name"
results_dir="artifacts/tamia/evals/$defence_model_name"
attack_learning_rates=(5e-5 8e-5 1e-4 2e-4)
attack_max_steps=(5)

# Setup.
uv sync --frozen
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
mkdir -p artifacts/tamia/logs
attack_separator_abbrv=$(python -c '
import sys
import yaml
with open(sys.argv[1]) as handle:
    config = yaml.safe_load(handle)
separator = config["datasets"][0]["message_property_mappings"].get("response_separator", "\n")
print({" ": "sp", "\n": "nl"}[separator])
' "$attack_config")

# Defence.
export CUDA_VISIBLE_DEVICES=0,1,2,3
if [[ ! -f "$defence_dir/.trained" ]]; then
    python -m representation_noising.train \
        --model "$base_model_name_or_path" \
        --output-dir "$defence_dir" \
        --harmful-response-separator "$defence_separator" \
        --alpha "$defence_alpha" \
        --beta "$defence_beta"
    touch "$defence_dir/.trained"
fi

# Defence evaluation before the attacks.
if [[ ! -f "$results_dir/results-1000.json" ]]; then
    python -m representation_noising.evaluation \
        --model "$defence_dir" \
        --results-path "$results_dir/results-1000.json" \
        --utility-samples 1000
fi


# Attack.
for max_steps in "${attack_max_steps[@]}"; do
    pids=()
    for gpu in 0 1 2 3; do
        attack_lr=${attack_learning_rates[gpu]}
        attack_name="$defence_model_name--relearn-fft-att-$attack_separator_abbrv-lr$attack_lr-ms$max_steps"
        attack_dir="artifacts/tamia/models/$attack_name"
        attack_results_dir="artifacts/tamia/evals/$attack_name"
        (
            export CUDA_VISIBLE_DEVICES=$gpu
            if [[ ! -f "$attack_dir/.trained" ]]; then
                axolotl train "$attack_config" --launcher python \
                    --base-model "$defence_dir" \
                    --output-dir "$attack_dir" \
                    --learning-rate "$attack_lr" \
                    --max-steps "$max_steps"
                touch "$attack_dir/.trained"
            fi
            if [[ ! -f "$attack_results_dir/results-1000.json" ]]; then
                python -m representation_noising.evaluation \
                    --model "$attack_dir" \
                    --results-path "$attack_results_dir/results-1000.json" \
                    --utility-samples 1000
            fi
        ) >> "artifacts/tamia/logs/$attack_name.log" 2>&1 &
        pids+=("$!")
    done

    failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    if (( failed )); then
        exit "$failed"
    fi
done

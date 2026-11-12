#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --job-name=repnoise-eval
#SBATCH --output=artifacts/tamia/logs/eval-%j.out
set -euo pipefail

# Cluster environment: install .venv and download inputs on the login node first.
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
source .venv/bin/activate

# Run from the repository root; each GPU trains and evaluates one attack.
model=${1:?usage: 2_eval.sbatch MODEL OUTPUT_PREFIX [TRAIN_ARGS...]}
output_prefix=${2:?missing output prefix}
shift 2
mkdir -p artifacts/tamia/logs
learning_rates=(5e-5 8e-5 1e-4 2e-4)
pids=()
for gpu in 0 1 2 3; do
    lr=${learning_rates[gpu]}
    output="$output_prefix--relearn-fft-lr$lr-ms250"
    name=$(basename "$output")
    (
        export CUDA_VISIBLE_DEVICES=$gpu
        python -m representation_noising.train \
            --model "$model" --output-dir "$output" --attack \
            --size 1000 --batch-size 4 --learning-rate "$lr" --device-map auto "$@"
        python -m representation_noising.evaluation \
            --model "$output" --results-path "artifacts/tamia/evals/$name/results.json"
    ) > "artifacts/tamia/logs/$name.log" 2>&1 &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
exit "$failed"

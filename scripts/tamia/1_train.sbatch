#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --job-name=repnoise-train
#SBATCH --output=artifacts/tamia/logs/train-%j.out
set -euo pipefail

# Cluster environment: prepare the shared caches on the login node first.
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$SCRATCH/.cache/uv" UV_OFFLINE=1 UV_LINK_MODE=copy
module load python/3.10
venv="$SLURM_TMPDIR/.venv"
uv venv --python "$(command -v python)" "$venv"
uv pip install --python "$venv/bin/python" -c requirements.txt \
    torch transformers datasets accelerate pandas numpy sentencepiece protobuf lm_eval
source "$venv/bin/activate"

# Run from the repository root. RepNoise uses the author's loss and paired data.
output=${1:?usage: 1_train.sbatch OUTPUT [TRAIN_ARGS...]}
shift
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m representation_noising.train --output-dir "$output" "$@"
name=$(basename "$output")
python -m representation_noising.evaluation \
    --model "$output" --results-path "artifacts/tamia/evals/$name/results.json"

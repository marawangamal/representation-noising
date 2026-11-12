"""Download inputs on the login node before submitting offline jobs."""

import os
import os.path as osp

os.environ["HF_HOME"] = osp.join(os.environ["SCRATCH"], ".cache/huggingface")

from datasets import load_dataset
from huggingface_hub import snapshot_download
from lm_eval.tasks import TaskManager, get_task_dict

# Download models
snapshot_download("meta-llama/Llama-2-7b-chat-hf")
snapshot_download("domenicrosati/deberta-v3-xsmall-beavertails-harmful-qa-classifier")

# Download train data
# Paired BeaverTails training data is already tracked under data/.

# Download eval data
load_dataset("PKU-Alignment/BeaverTails", split="30k_test")
get_task_dict(["mmlu", "truthfulqa_mc1", "arc_easy", "hellaswag"], TaskManager())

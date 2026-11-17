"""Run the demo notebook's RepNoise defense or harmful fine-tuning loop."""

import argparse
import json
import os
import os.path as osp

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, set_seed

from representation_noising.datasets import construct_beavertails_dataset
from representation_noising.loss import rep_noise_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--attack", action="store_true")
    parser.add_argument("--size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-map", default="balanced")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    if min(args.size, args.batch_size, args.epochs) < 1:
        parser.error("size, batch size, and epochs must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("max steps must be positive")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=False)
    with open(osp.join(args.output_dir, "training_args.json"), "w") as handle:
        json.dump(vars(args), handle, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    harmful, harmless, _ = construct_beavertails_dataset(
        tokenizer,
        attack=args.attack,
        attack_size=args.size,
        defence_size=args.size,
        train_batch_size=args.batch_size,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device_map
    )
    model.config.use_cache = False
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0
    )
    steps = len(harmful) * args.epochs
    if args.max_steps is not None:
        steps = min(steps, args.max_steps)
    scheduler = get_scheduler(
        "cosine", optimizer, num_warmup_steps=int(steps * 0.1), num_training_steps=steps
    )

    step = 0
    model.train()
    for epoch in range(args.epochs):
        for harmful_batch, harmless_batch in tqdm(
            zip(harmful, harmless), total=len(harmful)
        ):
            harmful_batch = {
                k: v.to(device) for k, v in harmful_batch.items() if torch.is_tensor(v)
            }
            harmless_batch = {
                k: v.to(device) for k, v in harmless_batch.items() if torch.is_tensor(v)
            }
            if args.attack:
                loss = model(
                    input_ids=harmful_batch["input_ids"],
                    attention_mask=harmful_batch["attention_mask"],
                    labels=harmful_batch["input_ids"],
                ).loss
            else:
                loss = rep_noise_loss(
                    model,
                    harmful_batch,
                    harmless_batch,
                    alpha=args.alpha,
                    beta=args.beta,
                )
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            print(json.dumps({"step": step, "loss": loss.item()}), flush=True)
            if step >= steps:
                break
        if step >= steps:
            break

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

#!/usr/bin/env python3
"""Plot RepNoise harmfulness and capability across attack learning rates."""

import argparse
import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rootdir", default=".")
    parser.add_argument(
        "--outpath",
        default="artifacts/figures/llama2-7b-beavertail--repnoise--lr-trajectory.pdf",
    )
    args = parser.parse_args()

    base = "llama2-7b-beavertail--align-fft-repnoise--upstream-paper-config"
    colors = sns.color_palette("colorblind", 3)
    configs = [
        {
            "path": "results-def-nl-att-nl/tamia/"
            + base
            + "--relearn-fft-lr{lr}-ms250",
            "name": r"RepNoise (def=\n, att=\n) [original]",
            "color": colors[0],
        },
        {
            "path": "results-def-nl-att-sp/tamia/"
            + base
            + "--def-nl-att-sp--relearn-fft-lr{lr}-ms250",
            "name": r"RepNoise (def=\n, att=sp)",
            "color": colors[1],
        },
        {
            "path": "artifacts/tamia/evals/"
            + base
            + "--def-sp-att-sp--relearn-fft-lr{lr}-ms250",
            "name": "RepNoise (def=sp, att=sp)",
            "color": colors[2],
        },
        {
            "path": "results/tamia/llama2-7b-beavertail--data-overlapping--relearn-fft-author-codebase-lr{lr}-ms250",
            "name": "Base (fine-tuned)",
            "color": "black",
        },
    ]
    learning_rates = ["5e-5", "8e-5", "1e-4", "2e-4"]
    task_metrics = {
        "mmlu": "acc,none",
        "truthfulqa_mc1": "acc,none",
        "arc_easy": "acc_norm,none",
        "hellaswag": "acc_norm,none",
    }

    rows = []
    for cfg in configs:
        for lr in learning_rates:
            model = cfg["path"].format(lr=lr)
            with open(osp.join(args.rootdir, model, "results.json")) as handle:
                result = json.load(handle)
            capability = sum(
                result["post_utility"][task][metric]
                for task, metric in task_metrics.items()
            ) / len(task_metrics)
            for evaluation, score in [
                ("Harmfulness ↓", result["harmfulness"]),
                ("Capability ↑", capability),
            ]:
                rows.append(
                    {
                        "Method": cfg["name"],
                        "LR": lr,
                        "Eval": evaluation,
                        "Score (%)": 100 * score,
                    }
                )

    df = pd.DataFrame(rows)
    sns.set_theme(context="paper", style="ticks", font_scale=1.5)
    grid = sns.catplot(
        data=df,
        x="LR",
        y="Score (%)",
        hue="Method",
        col="Eval",
        order=learning_rates,
        col_order=["Harmfulness ↓", "Capability ↑"],
        kind="point",
        errorbar=None,
        palette={cfg["name"]: cfg["color"] for cfg in configs},
        height=3,
        aspect=1.4,
    )
    grid.set(ylim=(0, 100))
    grid.set_axis_labels("Attack learning rate", "Score (%)")
    grid.set_titles("{col_name}")
    os.makedirs(osp.dirname(args.outpath) or ".", exist_ok=True)
    grid.figure.savefig(args.outpath, bbox_inches="tight")

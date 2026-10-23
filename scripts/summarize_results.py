#!/usr/bin/env python3
"""Rebuild compact tables and figures from the three saved experiment results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = [
    "baseline",
    "completion_kl0.1",
    "completion_kl5",
    "balanced_kl0.1",
    "balanced_kl5",
]
LABELS = {
    "baseline": "SFT",
    "completion_kl0.1": "Watch, KL=0.1",
    "completion_kl5": "Watch, KL=5",
    "balanced_kl0.1": "Mixed, KL=0.1",
    "balanced_kl5": "Mixed, KL=5",
}
COLORS = {
    "baseline": "#374151",
    "completion_kl0.1": "#dc2626",
    "completion_kl5": "#2563eb",
    "balanced_kl0.1": "#d97706",
    "balanced_kl5": "#15803d",
}
POLICY_METRICS = [
    "n",
    "raw_recall@10",
    "raw_ndcg@10",
    "raw_map@10",
    "predicted_watch@10",
    "predicted_return@10",
    "predicted_mixed_reward@10",
    "catalog_coverage@10",
    "top1_max_share",
    "history_repeat_rate@10",
]
RM_METRICS = ["n", "watch_mae", "watch_pearson", "return_auc_7d"]


def load_runs(directory: Path) -> list[dict]:
    paths = sorted((directory / "seeds").glob("seed_*.json"))
    if not paths:
        paths = sorted(directory.glob("seed_*/results.json"))
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if [run["seed"] for run in runs] != [42, 43, 44]:
        raise ValueError("Expected saved results for seeds 42, 43 and 44")
    return runs


def build_tables(runs: list[dict], output: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    policy_rows = []
    rm_rows = []
    rollout_rows = []
    for run in runs:
        for model, metrics in run["policy_metrics"].items():
            policy_rows.append(
                {"seed": run["seed"], "model": model, **{key: metrics[key] for key in POLICY_METRICS}}
            )
        rm_metrics = run["reward_model_metrics"]["test"]
        rm_rows.append(
            {"seed": run["seed"], **{key: rm_metrics[key] for key in RM_METRICS}}
        )
        rollout_rows.extend({"seed": run["seed"], **row} for row in run["rollouts"])
    policy = pd.DataFrame(policy_rows)
    reward = pd.DataFrame(rm_rows)
    rollout = pd.DataFrame(rollout_rows)
    policy.to_csv(output / "metrics_by_seed.csv", index=False)
    reward.to_csv(output / "reward_model_by_seed.csv", index=False)
    rollout.to_csv(output / "rollouts_by_seed.csv", index=False)
    return policy, reward, rollout


def plot_tradeoff(policy: pd.DataFrame, figures: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for model in MODEL_ORDER:
        rows = policy[policy.model == model]
        axis.scatter(
            rows["predicted_watch@10"], rows["raw_ndcg@10"],
            color=COLORS[model], alpha=0.25, s=35,
        )
        axis.scatter(
            rows["predicted_watch@10"].mean(), rows["raw_ndcg@10"].mean(),
            color=COLORS[model], s=90, label=LABELS[model],
        )
    axis.set_xlabel("Predicted watch reward@10")
    axis.set_ylabel("Raw NDCG@10")
    axis.set_title("Reward–relevance trade-off")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figures / "tradeoff.png", dpi=180)
    plt.close(figure)


def plot_rollout(rollout: pd.DataFrame, figures: Path) -> None:
    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2))
    for model in MODEL_ORDER:
        rows = rollout[rollout.model == model].groupby("step", as_index=False).mean(numeric_only=True)
        left.plot(rows.step, rows.predicted_watch, marker="o", color=COLORS[model], label=LABELS[model])
        right.plot(
            rows.step, 100 * rows.within_rollout_repeat_rate,
            marker="o", color=COLORS[model], label=LABELS[model],
        )
    left.set(title="Predicted watch reward", xlabel="Generated step", ylabel="Reward")
    right.set(title="Repeated generated item", xlabel="Generated step", ylabel="Share, %")
    for axis in (left, right):
        axis.set_xticks(range(1, 6))
        axis.grid(alpha=0.2)
    right.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(figures / "rollout.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    figures = args.output / "figures"
    figures.mkdir(exist_ok=True)
    runs = load_runs(args.input)
    policy, reward, rollout = build_tables(runs, args.output)
    plot_tradeoff(policy, figures)
    plot_rollout(rollout, figures)
    print(f"Saved compact tables and figures to {args.output}")


if __name__ == "__main__":
    main()

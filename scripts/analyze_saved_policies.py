#!/usr/bin/env python3
"""Decompose mixed reward and audit saved policies with independently seeded RMs."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kion_rl.data import collate_examples
from kion_rl.model import RewardModel
from kion_rl.train import get_device, mixed_reward_components, predicted_reward


MODEL_ORDER = [
    "baseline",
    "completion_kl0.1",
    "completion_kl5",
    "balanced_kl0.1",
    "balanced_kl5",
]


def _load_recommendations(path: Path, examples) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path)
    if len(frame) != len(examples):
        raise ValueError(f"{path}: expected {len(examples)} rows, found {len(frame)}")
    expected = [(example.user_id, example.date, example.target) for example in examples]
    observed = list(zip(frame.user_id, frame.date, frame.target))
    if observed != expected:
        raise ValueError(f"{path}: recommendation rows do not match the fixed test examples")
    recommendations = np.stack(
        frame.raw_top10.map(lambda value: np.fromstring(value, sep=" ", dtype=np.int64))
    )
    if recommendations.shape != (len(examples), 10):
        raise ValueError(f"{path}: expected exactly ten recommendations per row")
    return frame, recommendations


def _repeat_matrix(examples, recommendations: np.ndarray) -> np.ndarray:
    repeated = np.zeros_like(recommendations, dtype=np.float32)
    for row, example in enumerate(examples):
        repeated[row] = np.isin(recommendations[row], example.history)
    return repeated


def build_reward_decomposition(
    artifacts: Path,
    seeds: list[int],
    examples,
    popularity: np.ndarray,
    saved: dict[tuple[int, str], tuple[pd.DataFrame, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        run = json.loads((artifacts / f"seed_{seed}" / "results.json").read_text())
        for model in MODEL_ORDER:
            frame, recommendations = saved[(seed, model)]
            repeated = _repeat_matrix(examples, recommendations)
            mean_popularity = float(popularity[recommendations].mean())
            watch_component, return_component, novelty_component, repeat_component = (
                mixed_reward_components(
                    float(frame.predicted_watch.mean()),
                    float(frame.predicted_return.mean()),
                    mean_popularity,
                    float(repeated.mean()),
                )
            )
            total = watch_component + return_component + novelty_component + repeat_component
            expected = float(run["policy_metrics"][model]["predicted_mixed_reward@10"])
            if not np.isclose(total, expected, atol=2e-6):
                raise ValueError(
                    f"seed={seed} model={model}: reconstructed mixed reward {total:.8f} "
                    f"does not match saved value {expected:.8f}"
                )
            rows.append(
                {
                    "seed": seed,
                    "model": model,
                    "watch_component": watch_component,
                    "return_component": return_component,
                    "novelty_component": novelty_component,
                    "repeat_component": repeat_component,
                    "predicted_mixed_reward@10": total,
                    "mean_popularity@10": mean_popularity,
                    "history_repeat_rate@10": float(repeated.mean()),
                }
            )
    return pd.DataFrame(rows)


@torch.no_grad()
def _score_recommendations(
    model: RewardModel,
    examples,
    recommendations: np.ndarray,
    popularity: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    popularity_tensor = torch.tensor(popularity, device=device)
    watch_sum = return_sum = mixed_sum = 0.0
    count = 0
    for start in range(0, len(examples), batch_size):
        stop = min(start + batch_size, len(examples))
        batch = collate_examples(examples[start:stop])
        history = batch["history"].to(device)
        lengths = batch["lengths"].to(device)
        raw = torch.tensor(recommendations[start:stop], device=device)
        k = raw.size(1)
        response = model(
            history.repeat_interleave(k, dim=0),
            lengths.repeat_interleave(k),
            raw.reshape(-1),
        ).sigmoid().reshape(len(raw), k, 2)
        positions = torch.arange(history.size(1), device=device).view(1, 1, -1)
        valid = positions < lengths.view(-1, 1, 1)
        repeated = ((history.unsqueeze(1) == raw.unsqueeze(2)) & valid).any(dim=2).float()
        mixed = predicted_reward(
            "balanced",
            response[:, :, 0],
            response[:, :, 1],
            popularity_tensor[raw],
            repeated,
        )
        watch_sum += float(response[:, :, 0].sum())
        return_sum += float(response[:, :, 1].sum())
        mixed_sum += float(mixed.sum())
        count += raw.numel()
    return {
        "n_recommendations": count,
        "predicted_watch@10": watch_sum / count,
        "predicted_return@10": return_sum / count,
        "predicted_mixed_reward@10": mixed_sum / count,
    }


def build_cross_seed_audit(
    artifacts: Path,
    config: dict,
    seeds: list[int],
    examples,
    popularity: np.ndarray,
    saved: dict[tuple[int, str], tuple[pd.DataFrame, np.ndarray]],
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    kwargs = {key: config[key] for key in ("d_model", "n_layers", "n_heads", "dropout")}
    rows = []
    for audit_seed in seeds:
        model = RewardModel(len(popularity), config["max_len"], **kwargs)
        state = torch.load(
            artifacts / f"seed_{audit_seed}" / "reward_model.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state)
        model.to(device)
        for policy_seed in seeds:
            for policy_name in MODEL_ORDER:
                _, recommendations = saved[(policy_seed, policy_name)]
                metrics = _score_recommendations(
                    model, examples, recommendations, popularity, device, batch_size
                )
                rows.append(
                    {
                        "policy_seed": policy_seed,
                        "audit_rm_seed": audit_seed,
                        "same_seed_rm": policy_seed == audit_seed,
                        "model": policy_name,
                        **metrics,
                    }
                )
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    audit = pd.DataFrame(rows)
    for row in audit[audit.same_seed_rm].to_dict(orient="records"):
        run = json.loads(
            (artifacts / f"seed_{row['policy_seed']}" / "results.json").read_text()
        )
        expected = run["policy_metrics"][row["model"]]
        for metric in (
            "predicted_watch@10",
            "predicted_return@10",
            "predicted_mixed_reward@10",
        ):
            if not np.isclose(row[metric], expected[metric], atol=2e-6):
                raise ValueError(
                    f"seed={row['policy_seed']} model={row['model']}: recomputed {metric} "
                    f"does not match the published result"
                )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts" / "rerun_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    config = json.loads((args.artifacts / "config.json").read_text())
    seeds = [int(seed) for seed in config["seeds"]]
    with (args.artifacts / "prepared.pkl").open("rb") as handle:
        data = pickle.load(handle)
    saved = {}
    for seed in seeds:
        for model in MODEL_ORDER:
            path = args.artifacts / f"seed_{seed}" / f"recommendations_{model}.csv"
            saved[(seed, model)] = _load_recommendations(path, data.test)

    args.output.mkdir(parents=True, exist_ok=True)
    decomposition = build_reward_decomposition(
        args.artifacts, seeds, data.test, data.popularity, saved
    )
    decomposition.to_csv(args.output / "reward_decomposition_by_seed.csv", index=False)

    device = get_device(args.device)
    print(f"Auditing fixed recommendation lists on {device}", flush=True)
    audit = build_cross_seed_audit(
        args.artifacts,
        config,
        seeds,
        data.test,
        data.popularity,
        saved,
        device,
        args.batch_size,
    )
    audit.to_csv(args.output / "audit_reward_by_seed.csv", index=False)
    print(f"Saved diagnostics to {args.output}", flush=True)


if __name__ == "__main__":
    main()

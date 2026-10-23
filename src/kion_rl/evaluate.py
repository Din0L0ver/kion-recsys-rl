from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import NextItemDataset, collate_examples
from .train import _repeat_indicator, _to_device, predicted_reward


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


@torch.no_grad()
def evaluate_reward_model(
    model,
    examples,
    device,
    batch_size: int,
    watch_baseline: float,
    return_baseline: float,
) -> tuple[dict, pd.DataFrame]:
    model.eval()
    rows = []
    loader = DataLoader(
        NextItemDataset(examples), batch_size=batch_size, shuffle=False, collate_fn=collate_examples
    )
    for batch in loader:
        batch = _to_device(batch, device)
        prediction = model(batch["history"], batch["lengths"], batch["target"]).sigmoid()
        for actual_watch, actual_return, predicted_watch, predicted_return in zip(
            batch["watch"].cpu().numpy(),
            batch["returned"].cpu().numpy(),
            prediction[:, 0].cpu().numpy(),
            prediction[:, 1].cpu().numpy(),
        ):
            rows.append((actual_watch, actual_return, predicted_watch, predicted_return))
    frame = pd.DataFrame(
        rows, columns=["actual_watch", "actual_return", "predicted_watch", "predicted_return"]
    )
    correlation = frame.actual_watch.corr(frame.predicted_watch)
    metrics = {
        "n": len(frame),
        "watch_mae": float(np.abs(frame.actual_watch - frame.predicted_watch).mean()),
        "watch_rmse": float(np.sqrt(np.square(frame.actual_watch - frame.predicted_watch).mean())),
        "constant_watch_mae": float(np.abs(frame.actual_watch - watch_baseline).mean()),
        "constant_watch_rmse": float(np.sqrt(np.square(frame.actual_watch - watch_baseline).mean())),
        "watch_pearson": float(correlation) if np.isfinite(correlation) else float("nan"),
        "completion_auc_80pct": _auc(
            (frame.actual_watch >= 0.8).to_numpy(), frame.predicted_watch.to_numpy()
        ),
        "return_auc_7d": _auc(frame.actual_return.to_numpy(), frame.predicted_return.to_numpy()),
        "return_brier": float(np.square(frame.actual_return - frame.predicted_return).mean()),
        "constant_return_brier": float(np.square(frame.actual_return - return_baseline).mean()),
        "actual_watch_mean": float(frame.actual_watch.mean()),
        "predicted_watch_mean": float(frame.predicted_watch.mean()),
        "actual_return_rate": float(frame.actual_return.mean()),
        "predicted_return_mean": float(frame.predicted_return.mean()),
    }
    return metrics, frame


def _metric_sums(recommendations: torch.Tensor, targets: torch.Tensor) -> tuple[float, float, float]:
    matches = recommendations.eq(targets.unsqueeze(1))
    hits = matches.any(dim=1).float()
    ranks = matches.float().argmax(dim=1) + 1
    ranks = ranks.float()
    reciprocal_rank = torch.where(hits.bool(), 1.0 / ranks, torch.zeros_like(ranks))
    ndcg = torch.where(hits.bool(), 1.0 / torch.log2(ranks + 1), torch.zeros_like(ranks))
    return float(hits.sum()), float(ndcg.sum()), float(reciprocal_rank.sum())


@torch.no_grad()
def evaluate_policy(
    model,
    reward_model,
    examples,
    popularity: np.ndarray,
    device,
    batch_size: int,
    k: int = 10,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate raw top-k recommendations without target-aware filtering."""
    model.eval()
    reward_model.eval()
    loader = DataLoader(
        NextItemDataset(examples), batch_size=batch_size, shuffle=False, collate_fn=collate_examples
    )
    n = 0
    raw_recall = raw_ndcg = raw_map = 0.0
    repeat_total = 0.0
    recommendation_rows = []
    top1 = Counter()
    all_recommended = set()
    popularity_tensor = torch.tensor(popularity, device=device)
    watch_sum = return_sum = mixed_reward_sum = 0.0
    for batch in loader:
        batch = _to_device(batch, device)
        logits = model(batch["history"], batch["lengths"])
        logits[:, :2] = -torch.inf
        raw = logits.topk(k, dim=1).indices
        a, b, c = _metric_sums(raw, batch["target"])
        raw_recall += a
        raw_ndcg += b
        raw_map += c
        repeated = torch.stack(
            [_repeat_indicator(batch["history"], batch["lengths"], raw[:, j]) for j in range(k)],
            dim=1,
        )
        repeat_total += float(repeated.sum())

        expanded_history = batch["history"].repeat_interleave(k, dim=0)
        expanded_lengths = batch["lengths"].repeat_interleave(k)
        response = reward_model(
            expanded_history, expanded_lengths, raw.reshape(-1)
        ).sigmoid().reshape(len(raw), k, 2)
        mixed_reward = predicted_reward(
            "balanced",
            response[:, :, 0],
            response[:, :, 1],
            popularity_tensor[raw],
            repeated,
        )
        watch_sum += float(response[:, :, 0].sum())
        return_sum += float(response[:, :, 1].sum())
        mixed_reward_sum += float(mixed_reward.sum())
        n += len(batch["target"])

        for user, date, target, raw_row, row_response in zip(
            batch["user_id"], batch["date"], batch["target"].cpu(), raw.cpu(), response.cpu()
        ):
            values = raw_row.tolist()
            target_value = int(target)
            if target_value in values:
                rank = values.index(target_value) + 1
                raw_hit = 1.0
                raw_ndcg_row = 1.0 / np.log2(rank + 1)
                raw_ap = 1.0 / rank
            else:
                raw_hit = raw_ndcg_row = raw_ap = 0.0
            recommendation_rows.append(
                {
                    "user_id": int(user),
                    "date": date,
                    "target": target_value,
                    "raw_top10": " ".join(map(str, values)),
                    "raw_hit": raw_hit,
                    "raw_ndcg": raw_ndcg_row,
                    "raw_ap": raw_ap,
                    "predicted_watch": float(row_response[:, 0].mean()),
                    "predicted_return": float(row_response[:, 1].mean()),
                }
            )
            top1[values[0]] += 1
            all_recommended.update(values)

    denominator = max(n * k, 1)
    metrics = {
        "n": n,
        "raw_recall@10": raw_recall / max(n, 1),
        "raw_ndcg@10": raw_ndcg / max(n, 1),
        "raw_map@10": raw_map / max(n, 1),
        "history_repeat_rate@10": repeat_total / denominator,
        "catalog_coverage@10": len(all_recommended) / max(len(popularity) - 2, 1),
        "top1_max_share": max(top1.values(), default=0) / max(n, 1),
        "predicted_watch@10": watch_sum / denominator,
        "predicted_return@10": return_sum / denominator,
        "predicted_mixed_reward@10": mixed_reward_sum / denominator,
    }
    return metrics, pd.DataFrame(recommendation_rows)


@torch.no_grad()
def rollout_policy(
    model,
    reward_model,
    examples,
    popularity: np.ndarray,
    device,
    steps: int = 5,
    limit: int = 500,
    reward_name: str = "completion",
) -> pd.DataFrame:
    model.eval()
    reward_model.eval()
    batch = collate_examples(examples[:limit])
    batch = _to_device(batch, device)
    history = batch["history"]
    lengths = batch["lengths"]
    original_history = history.clone()
    original_lengths = lengths.clone()
    generated = []
    popularity_tensor = torch.tensor(popularity, device=device)
    rows = []
    for step in range(1, steps + 1):
        logits = model(history, lengths)
        logits[:, :2] = -torch.inf
        action = logits.argmax(dim=1)
        response = reward_model(history, lengths, action).sigmoid()
        original_repeat = _repeat_indicator(original_history, original_lengths, action)
        if generated:
            generated_tensor = torch.stack(generated, dim=1)
            rollout_repeat = generated_tensor.eq(action.unsqueeze(1)).any(dim=1).float()
        else:
            rollout_repeat = torch.zeros_like(original_repeat)
        reward = predicted_reward(
            reward_name,
            response[:, 0],
            response[:, 1],
            popularity_tensor[action],
            torch.maximum(original_repeat, rollout_repeat),
        )
        counts = torch.unique(action, return_counts=True)[1]
        rows.append(
            {
                "step": step,
                "predicted_watch": float(response[:, 0].mean()),
                "predicted_return": float(response[:, 1].mean()),
                "objective_reward": float(reward.mean()),
                "original_history_repeat_rate": float(original_repeat.mean()),
                "within_rollout_repeat_rate": float(rollout_repeat.mean()),
                "mean_popularity": float(popularity_tensor[action].mean()),
                "unique_items": int(action.unique().numel()),
                "top1_max_share": float(counts.max()) / len(action),
            }
        )
        generated.append(action)
        new_width = min(history.size(1) + 1, model.backbone.max_len)
        updated = torch.zeros(len(history), new_width, dtype=history.dtype, device=device)
        for row in range(len(history)):
            sequence = torch.cat((history[row, : lengths[row]], action[row : row + 1]))[-new_width:]
            updated[row, : len(sequence)] = sequence
        history = updated
        lengths = (lengths + 1).clamp(max=new_width)
    return pd.DataFrame(rows)


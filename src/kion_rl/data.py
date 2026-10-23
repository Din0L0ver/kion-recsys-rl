from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


PAD = 0
UNK = 1


@dataclass
class Example:
    user_id: int
    history: list[int]
    target: int
    watch: float
    returned_7d: float
    date: str


@dataclass
class PreparedData:
    train: list[Example]
    reward_train: list[Example]
    val: list[Example]
    test: list[Example]
    item_to_token: dict[int, int]
    token_to_item: list[int]
    popularity: np.ndarray
    summary: dict


def _future_return_labels(frame: pd.DataFrame) -> np.ndarray:
    """1 when the user has an interaction on a later calendar day within 7 days."""
    unique_days = frame[["user_id", "last_watch_dt"]].drop_duplicates()
    unique_days = unique_days.sort_values(["user_id", "last_watch_dt"])
    unique_days["next_distinct_dt"] = unique_days.groupby("user_id")["last_watch_dt"].shift(-1)
    lookup = unique_days.set_index(["user_id", "last_watch_dt"])["next_distinct_dt"]
    key = pd.MultiIndex.from_frame(frame[["user_id", "last_watch_dt"]])
    next_day = lookup.reindex(key).to_numpy(dtype="datetime64[ns]")
    current = frame["last_watch_dt"].to_numpy(dtype="datetime64[ns]")
    gap = (next_day - current) / np.timedelta64(1, "D")
    return ((gap >= 1) & (gap <= 7)).astype(np.float32)


def prepare_kion(
    csv_path: str | Path,
    max_len: int,
    train_end: str = "2021-07-25",
    val_end: str = "2021-08-01",
    max_train: int | None = 120_000,
    max_eval: int | None = 5_000,
    min_item_train_count: int = 3,
    seed: int = 42,
    test_start: str = "2021-08-09",
    test_end: str = "2021-08-15",
) -> PreparedData:
    """Build leakage-safe examples using only interactions on strictly earlier days.

    KION timestamps have day resolution. Same-day events therefore cannot be ordered
    reliably and are never included in one another's history.
    """
    frame = pd.read_csv(csv_path)
    frame["last_watch_dt"] = pd.to_datetime(frame["last_watch_dt"], format="%Y-%m-%d")
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)
    if not train_end_ts < val_end_ts < test_start_ts <= test_end_ts:
        raise ValueError("Expected train_end < val_end < test_start <= test_end")
    if val_end_ts + pd.Timedelta(days=7) >= test_start_ts:
        raise ValueError("Validation outcomes must mature strictly before test_start")
    if test_end_ts + pd.Timedelta(days=7) > frame.last_watch_dt.max():
        raise ValueError("Test outcomes require seven complete future days")
    train_median = float(frame.loc[frame.last_watch_dt <= train_end_ts, "watched_pct"].median())
    if not np.isfinite(train_median):
        raise ValueError("Training period needs observed watched_pct values")
    missing_watch = int(frame.watched_pct.isna().sum())
    frame["watched_pct"] = frame["watched_pct"].fillna(train_median)
    frame["watched_pct"] = frame["watched_pct"].clip(0, 100)
    frame["returned_7d"] = _future_return_labels(frame)
    frame = frame.sort_values(["user_id", "last_watch_dt"], kind="stable").reset_index(drop=True)

    train_rows = frame[frame.last_watch_dt <= train_end_ts]
    counts = train_rows.item_id.value_counts()
    known_items = sorted(counts[counts >= min_item_train_count].index.astype(int).tolist())
    item_to_token = {item: i + 2 for i, item in enumerate(known_items)}
    token_to_item = [-1, -1] + known_items
    vocab_size = len(token_to_item)

    popularity = np.zeros(vocab_size, dtype=np.float32)
    known_counts = counts.reindex(known_items).fillna(0).to_numpy(dtype=np.float32)
    popularity[2:] = np.log1p(known_counts) / max(float(np.log1p(known_counts).max()), 1.0)

    # Select target rows *before* the Python history-building loop. This keeps
    # laptop profiles fast and memory-bounded while all corpus statistics above
    # still use the complete archive.
    first_date = frame.groupby("user_id", sort=False)["last_watch_dt"].transform("min")
    has_prior_day = frame.last_watch_dt > first_date
    known_target = frame.item_id.isin(item_to_token)
    masks = {
        "train": has_prior_day & known_target & (frame.last_watch_dt <= train_end_ts),
        "reward_train": has_prior_day
        & known_target
        & (frame.last_watch_dt <= train_end_ts - pd.Timedelta(days=7)),
        "val": has_prior_day
        & known_target
        & (frame.last_watch_dt > train_end_ts)
        & (frame.last_watch_dt <= val_end_ts),
        "test": has_prior_day & known_target & frame.last_watch_dt.between(test_start_ts, test_end_ts),
    }
    raw_counts = {key: int(mask.sum()) for key, mask in masks.items()}
    limits = {"train": max_train, "reward_train": max_train, "val": max_eval, "test": max_eval}
    selected: dict[str, set[int]] = {}
    for offset, (key, mask) in enumerate(masks.items()):
        candidates = np.flatnonzero(mask.to_numpy())
        limit = limits[key]
        if limit and len(candidates) > limit:
            local_rng = np.random.default_rng(seed + offset)
            candidates = local_rng.choice(candidates, size=limit, replace=False)
        selected[key] = set(map(int, candidates))

    selected_rows = set().union(*selected.values())
    selected_users = frame.loc[sorted(selected_rows), "user_id"].unique()
    selected_frame = frame[frame.user_id.isin(selected_users)]
    buckets: dict[str, list[Example]] = {"train": [], "reward_train": [], "val": [], "test": []}

    for user_id, group in selected_frame.groupby("user_id", sort=False):
        history: list[int] = []
        for day, day_rows in group.groupby("last_watch_dt", sort=True):
            # All targets on a day see only strictly earlier days.
            if history:
                encoded_history = history[-max_len:]
                for row_index, row in zip(day_rows.index, day_rows.itertuples(index=False)):
                    destinations = [key for key, indices in selected.items() if row_index in indices]
                    if not destinations:
                        continue
                    target = item_to_token.get(int(row.item_id), UNK)
                    example = Example(
                        user_id=int(user_id),
                        history=list(encoded_history),
                        target=target,
                        watch=float(row.watched_pct) / 100.0,
                        returned_7d=float(row.returned_7d),
                        date=day.strftime("%Y-%m-%d"),
                    )
                    for destination in destinations:
                        buckets[destination].append(example)
            for item in day_rows.item_id.astype(int):
                history.append(item_to_token.get(item, UNK))

    train = buckets["train"]
    reward_train = buckets["reward_train"]
    val = buckets["val"]
    test = buckets["test"]
    eval_eligible = {
        "val": has_prior_day & (frame.last_watch_dt > train_end_ts) & (frame.last_watch_dt <= val_end_ts),
        "test": has_prior_day & frame.last_watch_dt.between(test_start_ts, test_end_ts),
    }
    summary = {
        "source_rows": int(len(frame)),
        "source_users": int(frame.user_id.nunique()),
        "source_items": int(frame.item_id.nunique()),
        "date_min": frame.last_watch_dt.min().strftime("%Y-%m-%d"),
        "date_max": frame.last_watch_dt.max().strftime("%Y-%m-%d"),
        "train_end": train_end,
        "val_end": val_end,
        "test_start": test_start,
        "test_end": test_end,
        "imputation_watch_median_train": train_median,
        "missing_watch_rows": missing_watch,
        "vocab_items": len(known_items),
        "raw_examples": raw_counts,
        "sampled_examples": {
            "train": len(train), "reward_train": len(reward_train), "val": len(val), "test": len(test)
        },
        "eval_item_coverage": {
            key: float((eval_eligible[key] & known_target).sum()) / max(int(eval_eligible[key].sum()), 1)
            for key in ("val", "test")
        },
        "same_day_policy": "a target sees interactions from strictly earlier dates only",
        "reward_censoring_policy": "RM train matures by train_end; validation matures before test_start; test has full 7d follow-up",
    }
    return PreparedData(
        train=train,
        reward_train=reward_train,
        val=val,
        test=test,
        item_to_token=item_to_token,
        token_to_item=token_to_item,
        popularity=popularity,
        summary=summary,
    )


class NextItemDataset(Dataset):
    def __init__(self, examples: list[Example]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Example:
        return self.examples[index]


def collate_examples(examples: Iterable[Example]) -> dict[str, torch.Tensor | list]:
    examples = list(examples)
    lengths = torch.tensor([len(x.history) for x in examples], dtype=torch.long)
    width = int(lengths.max())
    histories = torch.full((len(examples), width), PAD, dtype=torch.long)
    for row, example in enumerate(examples):
        histories[row, : len(example.history)] = torch.tensor(example.history, dtype=torch.long)
    return {
        "history": histories,
        "lengths": lengths,
        "target": torch.tensor([x.target for x in examples], dtype=torch.long),
        "watch": torch.tensor([x.watch for x in examples], dtype=torch.float32),
        "returned": torch.tensor([x.returned_7d for x in examples], dtype=torch.float32),
        "user_id": [x.user_id for x in examples],
        "date": [x.date for x in examples],
    }


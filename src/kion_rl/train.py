from __future__ import annotations

import copy
import itertools
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import NextItemDataset, collate_examples


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _loader(examples, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        NextItemDataset(examples),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_examples,
        generator=generator,
        num_workers=0,
    )


def _to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def train_policy_supervised(
    model,
    train_examples,
    val_examples,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> list[dict]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    model.to(device)
    history = []
    best_val = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        loader = _loader(train_examples, batch_size, True, seed + epoch)
        for batch in tqdm(loader, desc=f"SFT {epoch}/{epochs}", leave=False):
            batch = _to_device(batch, device)
            logits = model(batch["history"], batch["lengths"])[:, 2:]
            loss = F.cross_entropy(logits, batch["target"] - 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(loss.detach()) * len(batch["target"])
            seen += len(batch["target"])
        val_loss = next_item_loss(model, val_examples, device, batch_size)
        selected = val_loss < best_val
        if selected:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
        history.append(
            {"epoch": epoch, "train_loss": running_loss / seen, "val_loss": val_loss, "selected": selected}
        )
        print(f"SFT epoch {epoch}: train={running_loss / seen:.4f} val={val_loss:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


@torch.no_grad()
def next_item_loss(model, examples, device: torch.device, batch_size: int) -> float:
    model.eval()
    total = 0.0
    seen = 0
    for batch in _loader(examples, batch_size, False, 0):
        batch = _to_device(batch, device)
        logits = model(batch["history"], batch["lengths"])[:, 2:]
        loss = F.cross_entropy(logits, batch["target"] - 2, reduction="sum")
        total += float(loss)
        seen += len(batch["target"])
    return total / max(seen, 1)


def train_reward_model(
    model,
    train_examples,
    val_examples,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    return_weight: float = 0.35,
) -> list[dict]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    model.to(device)
    history = []
    best_val = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        loader = _loader(train_examples, batch_size, True, seed + 100 + epoch)
        for batch in tqdm(loader, desc=f"RM {epoch}/{epochs}", leave=False):
            batch = _to_device(batch, device)
            output = model(batch["history"], batch["lengths"], batch["target"])
            watch_loss = F.binary_cross_entropy_with_logits(output[:, 0], batch["watch"])
            return_loss = F.binary_cross_entropy_with_logits(output[:, 1], batch["returned"])
            loss = watch_loss + return_weight * return_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(batch["target"])
            seen += len(batch["target"])
        val_loss = reward_loss(model, val_examples, device, batch_size, return_weight)
        selected = val_loss < best_val
        if selected:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
        history.append(
            {"epoch": epoch, "train_loss": total / seen, "val_loss": val_loss, "selected": selected}
        )
        print(f"RM epoch {epoch}: train={total / seen:.4f} val={val_loss:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


@torch.no_grad()
def reward_loss(model, examples, device, batch_size, return_weight) -> float:
    model.eval()
    total = 0.0
    seen = 0
    for batch in _loader(examples, batch_size, False, 0):
        batch = _to_device(batch, device)
        output = model(batch["history"], batch["lengths"], batch["target"])
        watch_loss = F.binary_cross_entropy_with_logits(
            output[:, 0], batch["watch"], reduction="sum"
        )
        return_loss = F.binary_cross_entropy_with_logits(
            output[:, 1], batch["returned"], reduction="sum"
        )
        total += float(watch_loss + return_weight * return_loss)
        seen += len(batch["target"])
    return total / max(seen, 1)


@dataclass
class RLSpec:
    name: str
    reward: str
    kl_coef: float
    steps: int
    learning_rate: float
    anchor_coef: float = 0.05


def _repeat_indicator(history: torch.Tensor, lengths: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(history.size(1), device=history.device).unsqueeze(0)
    valid = positions < lengths.unsqueeze(1)
    return ((history == action.unsqueeze(1)) & valid).any(dim=1).float()


def predicted_reward(
    reward_name: str,
    watch: torch.Tensor,
    returned: torch.Tensor,
    popularity: torch.Tensor,
    repeated: torch.Tensor,
) -> torch.Tensor:
    if reward_name == "completion":
        return watch
    if reward_name == "balanced":
        return sum(mixed_reward_components(watch, returned, popularity, repeated))
    raise ValueError(f"unknown reward: {reward_name}")


def mixed_reward_components(watch, returned, popularity, repeated):
    """Return the four additive terms of the project's mixed reward."""
    return (
        0.65 * watch,
        0.25 * returned,
        0.10 * (1.0 - popularity),
        -0.20 * repeated,
    )


def train_policy_rl(
    baseline,
    reward_model,
    examples,
    popularity: np.ndarray,
    device: torch.device,
    spec: RLSpec,
    batch_size: int,
    seed: int,
    disable_dropout: bool = False,
):
    set_seed(seed)
    policy = copy.deepcopy(baseline).to(device)
    reference = copy.deepcopy(baseline).to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    reward_model.eval()
    for parameter in reward_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=spec.learning_rate, weight_decay=0.0)
    popularity_tensor = torch.tensor(popularity, device=device)
    loader = _loader(examples, batch_size, True, seed + 500)
    iterator = itertools.cycle(loader)
    logs = []
    policy.train()
    if disable_dropout:
        policy.eval()  # eval disables dropout, not autograd.
    for step in tqdm(range(1, spec.steps + 1), desc=f"RL {spec.name}", leave=False):
        batch = _to_device(next(iterator), device)
        logits = policy(batch["history"], batch["lengths"])[:, 2:]
        with torch.no_grad():
            reference_logits = reference(batch["history"], batch["lengths"])[:, 2:]
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        sampled_index = torch.multinomial(probs, num_samples=1).squeeze(1)
        action = sampled_index + 2
        chosen_log_prob = log_probs.gather(1, sampled_index.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            response = reward_model(batch["history"], batch["lengths"], action).sigmoid()
            repeated = _repeat_indicator(batch["history"], batch["lengths"], action)
            action_popularity = popularity_tensor[action]
            reward = predicted_reward(
                spec.reward, response[:, 0], response[:, 1], action_popularity, repeated
            )
            advantage = (reward - reward.mean()) / (reward.std() + 1e-6)

        ref_log_probs = F.log_softmax(reference_logits, dim=-1)
        kl = (probs * (log_probs - ref_log_probs)).sum(dim=-1).mean()
        policy_gradient = -(advantage * chosen_log_prob).mean()
        anchor = F.cross_entropy(logits, batch["target"] - 2)
        loss = policy_gradient + spec.kl_coef * kl + spec.anchor_coef * anchor
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % max(spec.steps // 10, 1) == 0 or step == spec.steps:
            row = {
                "step": step,
                "reward": float(reward.mean()),
                "predicted_watch": float(response[:, 0].mean()),
                "predicted_return": float(response[:, 1].mean()),
                "repeat_rate": float(repeated.mean()),
                "popularity": float(action_popularity.mean()),
                "kl": float(kl.detach()),
                "anchor_loss": float(anchor.detach()),
            }
            logs.append(row)
            print(
                f"{spec.name} step {step}: reward={row['reward']:.3f} "
                f"KL={row['kl']:.3f} repeat={row['repeat_rate']:.3f}"
            )
    return policy, logs


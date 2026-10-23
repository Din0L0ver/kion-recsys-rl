#!/usr/bin/env python3
"""Run the fixed, temporally separated 2x2 experiment; preserve original artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from kion_rl.data import prepare_kion
from kion_rl.model import GPT2Rec, RewardModel
from kion_rl.train import RLSpec, get_device, set_seed, train_policy_supervised, train_reward_model, train_policy_rl
from kion_rl.evaluate import evaluate_policy, evaluate_reward_model, rollout_policy


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def fingerprint():
    paths = [ROOT / "run_repeated.py"] + sorted((ROOT / "src/kion_rl").rglob("*.py"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true", help="Separate small verification, not reported results")
    args = parser.parse_args()
    device = get_device(args.device)
    out = ROOT / "artifacts" / ("rerun_smoke" if args.smoke else "rerun_v1")
    out.mkdir(parents=True, exist_ok=True)
    manifest = fingerprint()
    manifest_path = out / "source_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise SystemExit("Source changed since this experiment started. Use a new output version.")
    write_json(manifest_path, manifest)
    config = dict(data_seed=42, seeds=[42, 43, 44], max_train=200000, max_eval=3000,
                  epochs=2, steps=100, batch_size=128, d_model=64, n_layers=2,
                  n_heads=4, dropout=0.1, max_len=32, rl_dropout=False)
    if args.smoke:
        config.update(seeds=[42], max_train=256, max_eval=64, epochs=1, steps=2)
    write_json(out / "config.json", config)
    print(f"Preparing fixed data; device={device}", flush=True)
    cache = out / "prepared.pkl"
    if cache.exists():
        with cache.open("rb") as handle:
            data = pickle.load(handle)
    else:
        data = prepare_kion(ROOT / "data/raw/data_original/interactions.csv", max_len=32,
                            max_train=config["max_train"], max_eval=config["max_eval"], seed=42)
        with cache.open("wb") as handle:
            pickle.dump(data, handle)
    write_json(out / "data_summary.json", data.summary)
    print(json.dumps(data.summary), flush=True)
    for split in ("train", "reward_train", "val", "test"):
        pd.DataFrame([{"user_id": e.user_id, "date": e.date, "target": e.target} for e in getattr(data, split)]).to_csv(out / f"{split}_identities.csv", index=False)
    rng = np.random.default_rng(2026)
    rollout_examples = [data.test[i] for i in rng.choice(len(data.test), min(500, len(data.test)), replace=False)]
    for seed in config["seeds"]:
        run = out / f"seed_{seed}"
        run.mkdir(exist_ok=True)
        if (run / "results.json").exists():
            print(f"Already completed seed {seed}", flush=True)
            continue
        start = time.monotonic()
        set_seed(seed)
        kwargs = {k: config[k] for k in ("d_model", "n_layers", "n_heads", "dropout")}
        baseline = GPT2Rec(len(data.token_to_item), 32, **kwargs)
        rm = RewardModel(len(data.token_to_item), 32, **kwargs)
        print(f"Starting independent training seed={seed}", flush=True)
        sft_log = train_policy_supervised(baseline, data.train, data.val, device, config["epochs"], 128, 3e-4, seed)
        rm_log = train_reward_model(rm, data.reward_train, data.val, device, config["epochs"], 128, 3e-4, seed)
        watch_mean = float(np.mean([e.watch for e in data.reward_train]))
        return_mean = float(np.mean([e.returned_7d for e in data.reward_train]))
        rm_metrics = {}
        for split in ("val", "test"):
            rm_metrics[split], frame = evaluate_reward_model(rm, getattr(data, split), device, 128, watch_mean, return_mean)
            examples = getattr(data, split)
            frame["user_id"] = [e.user_id for e in examples]
            frame["target"] = [e.target for e in examples]
            frame["date"] = [e.date for e in examples]
            frame.to_csv(run / f"rm_predictions_{split}.csv", index=False)
        torch.save({k: v.cpu() for k, v in rm.state_dict().items()}, run / "reward_model.pt")
        specs = [RLSpec(f"{reward}_kl{kl:g}", reward, kl, config["steps"], 8e-5)
                 for reward in ("completion", "balanced") for kl in (0.1, 5.0)]
        metrics, logs, rollouts = {}, {}, []
        for spec in [None] + specs:
            name = "baseline" if spec is None else spec.name
            if spec is None:
                policy = baseline
            else:
                policy, logs[name] = train_policy_rl(baseline, rm, data.train, data.popularity, device,
                                                    spec, 128, seed + 100, disable_dropout=True)
            torch.save({k: v.cpu() for k, v in policy.state_dict().items()}, run / f"{name}.pt")
            metrics[name], recs = evaluate_policy(
                policy, rm, data.test, data.popularity, device, 128
            )
            recs.to_csv(run / f"recommendations_{name}.csv", index=False)
            rollout = rollout_policy(policy, rm, rollout_examples, data.popularity, device,
                                     reward_name="completion" if spec is None else spec.reward)
            rollout["model"] = name
            rollouts.extend(rollout.to_dict(orient="records"))
            print(f"seed={seed} {name}: {json.dumps(metrics[name])}", flush=True)
        result = dict(seed=seed, rl_seed=seed+100, config=config, specs=[asdict(s) for s in specs],
                      policy_metrics=metrics, reward_model_metrics=rm_metrics,
                      training=dict(sft=sft_log, reward_model=rm_log, rl=logs), rollouts=rollouts,
                      elapsed_seconds=time.monotonic()-start,
                      runtime=dict(device=str(device), python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__))
        write_json(run / "results.json", result)
        print(f"Completed seed={seed} in {result['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()

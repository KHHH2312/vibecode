"""Checkpoint schema v2 — full-state resume (plan.md)."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .constants import CHECKPOINT_SCHEMA_VERSION


class CheckpointError(RuntimeError):
    pass


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def build_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    ema: Any | None = None,
    stage: str = "A",
    epoch: int = 0,
    global_step: int = 0,
    best_exact: float | None = None,
    sampler_state: dict | None = None,
    hard_negative_version: int = 0,
    config: dict | None = None,
    split_sha256: str = "",
    code_sha: str = "",
    inference_policy_sha256: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": model.state_dict(),
        "ema": ema.state_dict() if ema is not None and hasattr(ema, "state_dict") else None,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "stage": stage,
        "epoch": epoch,
        "global_step": global_step,
        "best_exact": best_exact,
        "rng": _rng_state(),
        "sampler": sampler_state,
        "hard_negative_version": hard_negative_version,
        "config": config or {},
        "split_sha256": split_sha256,
        "code_sha": code_sha,
        "inference_policy_sha256": inference_policy_sha256,
    }


def atomic_torch_save(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def save_checkpoint(state: dict, path: str | Path) -> None:
    if state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError(f"refusing to save schema {state.get('schema_version')}")
    atomic_torch_save(state, path)


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    expected_split_sha256: str | None = None,
    expected_inference_sha256: str | None = None,
    strict_model: bool = True,
) -> dict[str, Any]:
    path = Path(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "schema_version" not in state:
        raise CheckpointError(
            "legacy weights-only or unknown checkpoint — may warm-start model but "
            "must NOT be reported as a resumed run"
        )
    if state["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError(
            f"schema {state['schema_version']} != {CHECKPOINT_SCHEMA_VERSION}; migration required"
        )
    if expected_split_sha256 and state.get("split_sha256") != expected_split_sha256:
        raise CheckpointError("split_sha256 mismatch")
    if expected_inference_sha256 and state.get("inference_policy_sha256") != expected_inference_sha256:
        raise CheckpointError("inference_policy_sha256 mismatch")

    missing, unexpected = model.load_state_dict(state["model"], strict=strict_model)
    if strict_model and (missing or unexpected):
        raise CheckpointError(f"model load missing={missing} unexpected={unexpected}")

    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    if state.get("rng") is not None:
        _set_rng_state(state["rng"])
    return state


def is_full_resume_checkpoint(path: str | Path) -> bool:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        return isinstance(state, dict) and state.get("schema_version") == CHECKPOINT_SCHEMA_VERSION
    except Exception:
        return False


def config_sha256(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()

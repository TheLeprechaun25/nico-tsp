"""Checkpoints: the full run config is stored with the weights and loading is strict."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from .config import Config
from .model import NICOPolicy


def rng_state() -> Dict[str, Any]:
    state = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    # RNG states must be CPU byte tensors, also when the checkpoint was loaded onto a GPU
    torch.set_rng_state(state["torch"].cpu())
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(path: str | Path, model: NICOPolicy, cfg: Config, *, epoch: int, stage: str,
                    optimizer=None, scheduler=None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    torch.save(
        {
            "format": "nico-v1",
            "config": cfg.to_dict(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": int(epoch),
            "stage": stage,
            "rng_state": rng_state(),
        },
        tmp,
    )
    tmp.replace(path)


def load_checkpoint(path: str | Path, device="cpu") -> Tuple[NICOPolicy, Config, Dict[str, Any]]:
    data = torch.load(path, map_location=device, weights_only=False)
    if data.get("format") != "nico-v1":
        raise ValueError(f"{path} is not a nico-v1 checkpoint")
    cfg = Config.from_dict(data["config"])
    model = NICOPolicy(cfg.model, cfg.problem).to(device)
    model.load_state_dict(data["model"], strict=True)
    return model, cfg, data

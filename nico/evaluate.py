"""Evaluation of a trained policy as a standalone search or as a refinement module."""

from __future__ import annotations

import time
from typing import Dict, Optional

import torch

from .config import SearchConfig
from .problems.base import Instance, index_instance, tile_instance
from .search import improve


def _chunk_solutions(problem, chunk: Instance, init, restarts: int, offset: int) -> torch.Tensor:
    """(restarts * b, L) initial solutions for one chunk, restart-major."""
    b = chunk["cost_matrix"].size(0)
    if torch.is_tensor(init):
        sol = init[offset:offset + b].to(chunk["cost_matrix"].device)
        if sol.dim() == 2:
            return sol.repeat(restarts, 1)
        return sol[:, :restarts].transpose(0, 1).reshape(restarts * b, -1)
    if init == "identity" and restarts == 1:
        return problem.identity_solutions(chunk)
    if init in ("identity", "random"):
        return problem.random_solutions(tile_instance(chunk, restarts))
    raise ValueError(f"Unknown init {init!r}")


@torch.no_grad()
def evaluate(model, problem, instance: Instance, search_cfg: SearchConfig, *, steps: int,
             init="identity", restarts: int = 1, batch_size: int = 100, sample: bool = True,
             time_budget_s: Optional[float] = None, record_trace: bool = False) -> Dict[str, torch.Tensor]:
    """Improve every instance for ``steps`` moves and return its best cost.

    ``init`` is "identity" (the stored node order, as in the paper's single-run
    results), "random", or a tensor of initial solutions, (B, L) or (B, R, L).
    With ``restarts`` R > 1 the R searches run in parallel and the best is kept.
    """
    model.eval()
    B = instance["cost_matrix"].size(0)
    device = instance["cost_matrix"].device
    costs, solutions, traces = [], [], []
    start = time.perf_counter()
    for lo in range(0, B, batch_size):
        idx = torch.arange(lo, min(B, lo + batch_size), device=device)
        chunk = index_instance(instance, idx)
        b = idx.numel()
        init_sol = _chunk_solutions(problem, chunk, init, restarts, lo)
        out = improve(model, problem, tile_instance(chunk, restarts), init_sol, steps, search_cfg,
                      sample=sample, time_budget_s=time_budget_s, record_trace=record_trace)
        best = out["best_cost"].view(restarts, b)
        which = best.argmin(dim=0)
        costs.append(best.min(dim=0).values)
        sols = out["best_solution"].view(restarts, b, -1)
        solutions.append(sols[which, torch.arange(b, device=device)])
        if record_trace:
            traces.append(out["trace"].view(restarts, b, -1).min(dim=0).values)
    result = {
        "cost": torch.cat(costs),
        "solution": torch.cat(solutions),
        "seconds": torch.tensor(time.perf_counter() - start),
    }
    if record_trace:
        result["trace"] = torch.cat(traces)
    return result


def gap_percent(cost: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Per-instance optimality gap in percent."""
    return 100.0 * (cost - reference) / reference

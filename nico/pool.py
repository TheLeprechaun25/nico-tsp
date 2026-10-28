"""Persistent training states: each batch continues from the best solutions found for the
same instances in earlier batches, so training reaches the near-optimal states seen when
refining strong solutions (a long-horizon curriculum without an external solver)."""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from .config import Config
from .problems.base import Instance, euclidean_cost_matrix
from .search import initial_solutions


class StatePool:
    """``pool_size`` instances per problem size with their best solution so far.

    Only Euclidean instances are supported (the cost matrix is rebuilt from ``coords``).
    After every update a share ``pool_refresh`` of the updated entries is replaced by
    fresh instances with new initial solutions.
    """

    def __init__(self, cfg: Config, problem, source, device):
        if cfg.data.source != "uniform":
            raise ValueError("The state pool supports uniform Euclidean instances only")
        self.cfg, self.problem, self.source, self.device = cfg, problem, source, device
        self.size = int(cfg.train.pool_size)
        self.refresh = float(cfg.train.pool_refresh)
        self.pools: Dict[int, Dict[str, torch.Tensor]] = {}
        self.routes: Dict[int, int] = {}
        self.fixed = set()
        if cfg.train.pool_data:
            self._preload(cfg.train.pool_data, cfg.train.pool_solutions)

    def _preload(self, data_path: str, solutions_path: str) -> None:
        import numpy as np

        inst = self.problem.load_npz(data_path)
        with np.load(solutions_path) as z:
            raw = z["solutions"]
        inst = {k: v[: len(raw)].to(self.device) for k, v in inst.items()}
        sol = self.problem.from_sequences(raw[: inst["coords"].size(0)]).to(self.device)
        n = inst["coords"].size(1) - 1
        self.routes[n] = int((sol == 0).sum(dim=1).max())
        self.fixed.add(n)
        self.pools[n] = {**self._stored(inst), "solution": sol, "cost": self.problem.cost(inst, sol)}

    def _fresh(self, n: int, count: int) -> Tuple[Instance, torch.Tensor]:
        inst = self.source.sample(count, n, self.device)
        kw = {}
        if n in self.routes:
            kw["num_routes"] = self.routes[n]
        sol = initial_solutions(self.problem, inst, self.cfg, **kw)
        return inst, sol

    def _stored(self, inst: Instance) -> Dict[str, torch.Tensor]:
        return {k: v for k, v in inst.items() if k != "cost_matrix" and torch.is_tensor(v)}

    def sample(self, batch_size: int, n: int) -> Tuple[Instance, torch.Tensor, torch.Tensor]:
        """(instance, start solutions, pool indices) for ``batch_size`` entries of size n."""
        if n not in self.pools:
            count = max(self.size, batch_size)
            inst, sol = self._fresh(n, count)
            if self.problem.name == "cvrp":
                # fixed depot copies per size, with slack so later fresh instances fit
                self.routes[n] = int((sol == 0).sum(dim=1).max()) + 2
                sol = initial_solutions(self.problem, inst, self.cfg, num_routes=self.routes[n])
            self.pools[n] = {**self._stored(inst), "solution": sol, "cost": self.problem.cost(inst, sol)}
        pool = self.pools[n]
        idx = torch.randperm(pool["solution"].size(0), device=self.device)[:batch_size]
        inst = {k: v[idx] for k, v in pool.items() if k not in ("solution", "cost")}
        inst["cost_matrix"] = euclidean_cost_matrix(inst["coords"])
        return inst, pool["solution"][idx].clone(), idx

    @torch.no_grad()
    def update(self, n: int, idx: torch.Tensor, solution: torch.Tensor, cost: torch.Tensor) -> None:
        pool = self.pools[n]
        better = cost < pool["cost"][idx]
        pool["solution"][idx[better]] = solution[better]
        pool["cost"][idx[better]] = cost[better]
        if n in self.fixed:          # preloaded pools keep their instances
            return
        replace = idx[torch.rand(idx.numel(), device=self.device) < self.refresh]
        if replace.numel():
            try:
                inst, sol = self._fresh(n, replace.numel())
            except ValueError:      # a fresh split needed more routes than the pool has
                return
            for k, v in self._stored(inst).items():
                pool[k][replace] = v
            pool["solution"][replace] = sol
            pool["cost"][replace] = self.problem.cost(inst, sol)

    def depth(self) -> float:
        """Mean cost of the pooled solutions relative to their instances' sizes (for logging)."""
        return float(torch.stack([p["cost"].mean() for p in self.pools.values()]).mean()) if self.pools else 0.0

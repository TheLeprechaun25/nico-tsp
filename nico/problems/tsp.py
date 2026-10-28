"""Symmetric TSP (Euclidean or arbitrary symmetric cost matrix)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .base import (
    Instance,
    admissible_pairs_mask,
    apply_two_opt,
    euclidean_cost_matrix,
    tour_cost,
    two_opt_delta,
    two_opt_delta_matrix,
)


class TSP:
    name = "tsp"

    # --------------------------------------------------------------- instances
    @staticmethod
    def from_coords(coords: torch.Tensor) -> Instance:
        coords = coords.float()
        return {"coords": coords, "cost_matrix": euclidean_cost_matrix(coords)}

    def sample_instances(self, batch_size: int, size: int, device, generator=None) -> Instance:
        coords = torch.rand(batch_size, size, 2, device=device, generator=generator)
        return self.from_coords(coords)

    @staticmethod
    def load_pickle(path: str | Path, num: Optional[int] = None) -> Instance:
        """Load a list of (n, 2) coordinate arrays (the format of the test sets in data/)."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        coords = torch.as_tensor(np.asarray(data[:num] if num else data), dtype=torch.float32)
        return TSP.from_coords(coords)

    # --------------------------------------------------------------- solutions
    @staticmethod
    def solution_length(instance: Instance) -> int:
        return instance["cost_matrix"].size(1)

    def identity_solutions(self, instance: Instance) -> torch.Tensor:
        B, V, _ = instance["cost_matrix"].shape
        return torch.arange(V, device=instance["cost_matrix"].device).expand(B, V).clone()

    def random_solutions(self, instance: Instance, generator=None) -> torch.Tensor:
        B, V, _ = instance["cost_matrix"].shape
        noise = torch.rand(B, V, device=instance["cost_matrix"].device, generator=generator)
        return noise.argsort(dim=1)

    # ------------------------------------------------------------- operations
    @staticmethod
    def cost(instance: Instance, solution: torch.Tensor) -> torch.Tensor:
        return tour_cost(instance["cost_matrix"], solution)

    @staticmethod
    def apply_move(instance: Instance, solution: torch.Tensor, moves: torch.Tensor) -> torch.Tensor:
        return apply_two_opt(solution, moves)

    n_ops = 1

    def move_values(self, instance: Instance, solution: torch.Tensor) -> torch.Tensor:
        """(B, L * L) exact cost change of every move, +inf where invalid."""
        B, L = solution.shape
        delta, _ = two_opt_delta_matrix(instance["cost_matrix"], solution)
        return delta.masked_fill(admissible_pairs_mask(L, solution.device), float("inf")).view(B, L * L)

    @staticmethod
    def invalid_moves(instance: Instance, solution: torch.Tensor) -> torch.Tensor:
        """(B, L, L) bool, True where a move is not allowed."""
        B, L = solution.shape
        return admissible_pairs_mask(L, solution.device).expand(B, L, L)

    @staticmethod
    def is_feasible(instance: Instance, solution: torch.Tensor) -> torch.Tensor:
        B, V, _ = instance["cost_matrix"].shape
        ref = torch.arange(V, device=solution.device).expand(B, V)
        return (solution.sort(dim=1).values == ref).all(dim=1)

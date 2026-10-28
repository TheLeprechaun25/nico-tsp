"""Instance representation and the 2-opt primitives shared by every problem.

An *instance batch* is a ``dict[str, Tensor]`` whose tensors share the leading
batch dimension. Every instance carries ``cost_matrix`` (B, V, V), the
symmetric objective; everything else (``coords``, ``demand``, road features, ...)
is optional model input or problem data.

A *solution* is a (B, L) tensor of node ids read as a cyclic sequence. Its
solution edges are ``e_k = (sol[k], sol[k+1])``, k = 0..L-1 (indices mod L).
A 2-opt move ``(i, j)`` with ``i < j`` cuts ``e_i`` and ``e_j`` and reverses
``sol[i+1 : j+1]``.
"""

from __future__ import annotations

from typing import Dict

import torch

Instance = Dict[str, torch.Tensor]


def euclidean_cost_matrix(coords: torch.Tensor) -> torch.Tensor:
    """Exact pairwise Euclidean distances, (B, V, 2) -> (B, V, V)."""
    return torch.cdist(coords, coords, compute_mode="donot_use_mm_for_euclid_dist")


def index_instance(instance: Instance, index: torch.Tensor) -> Instance:
    return {k: v.index_select(0, index) if torch.is_tensor(v) else v for k, v in instance.items()}


def repeat_instance(instance: Instance, repeats: int) -> Instance:
    """Repeat each instance ``repeats`` times consecutively (b0, b0, ..., b1, b1, ...)."""
    return {
        k: v.repeat_interleave(int(repeats), dim=0) if torch.is_tensor(v) else v
        for k, v in instance.items()
    }


def tile_instance(instance: Instance, copies: int) -> Instance:
    """Stack ``copies`` copies of the whole batch (b0, b1, ..., b0, b1, ...)."""
    return {
        k: v.repeat(int(copies), *([1] * (v.dim() - 1))) if torch.is_tensor(v) else v
        for k, v in instance.items()
    }


def to_device(instance: Instance, device) -> Instance:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in instance.items()}


def tour_cost(cost_matrix: torch.Tensor, solution: torch.Tensor) -> torch.Tensor:
    """Cost of cyclic sequences, (B, V, V) x (B, L) -> (B,)."""
    B, V, _ = cost_matrix.shape
    nxt = torch.roll(solution, shifts=-1, dims=1)
    flat = (solution * V + nxt).long()
    return cost_matrix.reshape(B, V * V).gather(1, flat).sum(dim=1)


def edge_costs(cost_matrix: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Costs d[b, u[b, ...], v[b, ...]] for index tensors of shape (B, ...)."""
    B, V, _ = cost_matrix.shape
    flat = (u * V + v).long().reshape(B, -1)
    return cost_matrix.reshape(B, V * V).gather(1, flat).view(u.shape)


@torch.no_grad()
def apply_two_opt(solution: torch.Tensor, moves: torch.Tensor) -> torch.Tensor:
    """Apply one 2-opt move per row; ``moves`` is (B, 2) with i < j."""
    B, L = solution.shape
    lo = (moves[:, 0] + 1).view(B, 1)
    hi = moves[:, 1].view(B, 1)
    idx = torch.arange(L, device=solution.device).view(1, L).expand(B, L)
    in_seg = (idx >= lo) & (idx <= hi)
    return solution.gather(1, torch.where(in_seg, lo + hi - idx, idx))


_ADMISSIBLE_CACHE: dict = {}


def admissible_pairs_mask(L: int, device) -> torch.Tensor:
    """(L, L) bool, True where (i, j) is NOT an admissible 2-opt move.

    Admissible: i < j, j >= i + 2, and (i, j) != (0, L - 1) (which would cut two
    adjacent edges of the cycle).
    """
    key = (str(device), int(L))
    mask = _ADMISSIBLE_CACHE.get(key)
    if mask is None:
        ar = torch.arange(L, device=device)
        mask = ar.view(L, 1) >= ar.view(1, L)          # i >= j
        mask[ar, (ar + 1) % L] = True                    # j == i + 1 and (0, L-1)
        mask[ar, (ar - 1) % L] = True
        _ADMISSIBLE_CACHE[key] = mask
    return mask


def two_opt_delta_matrix(cost_matrix: torch.Tensor, solution: torch.Tensor):
    """Cost change of every 2-opt pair (i, j), (B, L, L), and the solution edge costs (B, L).

    Delta_ij = d(s_i, s_j) + d(s_{i+1}, s_{j+1}) - d(s_i, s_{i+1}) - d(s_j, s_{j+1}). With the cost matrix
    permuted into tour order, P = C[s][:, s], the second term is P shifted by one along both axes and the
    edge costs are P_{i, i+1}, so one permutation gather replaces four scattered gathers.
    Same values as :func:`two_opt_delta` over all pairs (entries for inadmissible pairs are meaningless).
    """
    B, L = solution.shape
    V = cost_matrix.size(-1)
    rows = cost_matrix.gather(1, solution.unsqueeze(-1).expand(B, L, V))
    P = rows.gather(2, solution.unsqueeze(1).expand(B, L, L))
    nxt = torch.roll(torch.arange(L, device=solution.device), -1)
    edge = P.gather(2, nxt.view(1, L, 1).expand(B, L, 1)).squeeze(-1)                # d(s_i, s_{i+1})
    delta = P + torch.roll(P, shifts=(-1, -1), dims=(1, 2))
    delta -= edge.unsqueeze(2)
    delta -= edge.unsqueeze(1)
    return delta, edge


def two_opt_delta(cost_matrix: torch.Tensor, solution: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """Cost change of every move in ``pairs`` (M, 2) for every row, -> (B, M).

    Exact for symmetric costs: only the two removed and two inserted edges change.
    """
    a = solution
    b = torch.roll(a, shifts=-1, dims=1)
    i, j = pairs[:, 0], pairs[:, 1]
    ai, aj, bi, bj = a[:, i], a[:, j], b[:, i], b[:, j]
    return (
        edge_costs(cost_matrix, ai, aj) + edge_costs(cost_matrix, bi, bj)
        - edge_costs(cost_matrix, ai, bi) - edge_costs(cost_matrix, aj, bj)
    )

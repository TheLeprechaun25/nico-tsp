"""Exact K-step 2-opt oracle with set-valued targets (Section 6.1).

For a state s, the oracle returns the set A*(s) of first moves that start a
sequence of K moves with minimum total cost change. K = 1 is the best feasible
move; K = 2 is an exact streaming dynamic program over all pairs of moves (TSP, and the CVRP
with 2-opt only, where the capacity feasibility of the second move is evaluated on the tour after
the first). Costs are read from ``instance["cost_matrix"]``, so the oracle serves
Euclidean and road-network instances alike.
"""

from __future__ import annotations

from typing import Dict

import torch

from .problems.base import Instance, admissible_pairs_mask, edge_costs, two_opt_delta

_TABLES: Dict[tuple, Dict[str, torch.Tensor]] = {}


def move_table(L: int, device) -> Dict[str, torch.Tensor]:
    """All admissible moves (M, 2) and, per move, the position map of its reversal."""
    key = (str(device), int(L))
    if key not in _TABLES:
        pairs = (~admissible_pairs_mask(L, device)).nonzero().long()
        M = pairs.size(0)
        pos = torch.arange(L, device=device).view(1, L).expand(M, L)
        lo, hi = (pairs[:, 0] + 1).view(M, 1), pairs[:, 1].view(M, 1)
        maps = torch.where((pos >= lo) & (pos <= hi), lo + hi - pos, pos)
        _TABLES[key] = {
            "pairs": pairs,
            "maps": maps,
            "i": pairs[:, 0], "j": pairs[:, 1],
            "i1": (pairs[:, 0] + 1) % L, "j1": (pairs[:, 1] + 1) % L,
        }
    return _TABLES[key]


def one_step_values(problem, instance: Instance, solution: torch.Tensor) -> torch.Tensor:
    """(B, M) cost change of each admissible 2-opt move; +inf where the move is infeasible."""
    table = move_table(solution.size(1), solution.device)
    delta = two_opt_delta(instance["cost_matrix"], solution, table["pairs"])
    invalid = problem.invalid_moves(instance, solution)[:, table["i"], table["j"]]
    return delta.masked_fill(invalid, float("inf"))


@torch.no_grad()
def _best_second_move(cost: torch.Tensor, solution: torch.Tensor, table, block_m1: int,
                      block_m2: int, problem=None, instance: Instance = None) -> torch.Tensor:
    """min over m2 of delta(m2 | m1 applied), for every first move m1 -> (B, M).

    Evaluates M^2 moves per instance without materialising the M^2 tours: the
    endpoints of m2 on the tour after m1 are read through m1's position map. For the
    CVRP, the route, prefix load and route load of every position of the tour after m1
    are computed once per m1 and m2 is masked by the 2-opt feasibility rule (Proposition 4).
    """
    B, L = solution.shape
    maps, i, j, i1, j1 = table["maps"], table["i"], table["j"], table["i1"], table["j1"]
    M = maps.size(0)
    cvrp = problem is not None and problem.name == "cvrp"
    out = torch.full((B, M), float("inf"), device=solution.device)
    for a in range(0, M, block_m1):
        map1 = maps[a:a + block_m1]                                # (B1, L)
        B1 = map1.size(0)
        tours = solution.unsqueeze(1).expand(B, B1, L)
        if cvrp:
            after = tours.gather(2, map1.unsqueeze(0).expand(B, B1, L)).reshape(B * B1, L)
            sub = {"demand": instance["demand"].repeat_interleave(B1, 0)}
            info = problem.route_info(sub, after)
            r, P, T = (x.view(B, B1, L) for x in (info["route"], info["prefix"], info["load"] - info["prefix"]))
            Q = instance["capacity"].view(B, 1, 1) + 1e-6
        best = torch.full((B, B1), float("inf"), device=solution.device)
        for c in range(0, M, block_m2):
            sl = slice(c, c + block_m2)
            ends = [map1[:, x[sl]].unsqueeze(0).expand(B, -1, -1) for x in (i, j, i1, j1)]
            ai, aj, bi, bj = (tours.gather(2, e) for e in ends)
            d = (edge_costs(cost, ai, aj) + edge_costs(cost, bi, bj)
                 - edge_costs(cost, ai, bi) - edge_costs(cost, aj, bj))
            if cvrp:   # positions i, j of m2 on the tour after m1
                pi, pj = (x[sl].view(1, 1, -1).expand(B, B1, -1) for x in (i, j))
                g = lambda t, q: t.gather(2, q)
                bad = (g(r, pi) != g(r, pj)) & ((g(P, pi) + g(P, pj) > Q) | (g(T, pi) + g(T, pj) > Q))
                d = d.masked_fill(bad, float("inf"))
            best = torch.minimum(best, d.amin(dim=2))
        out[:, a:a + B1] = best
    return out


@torch.no_grad()
def optimal_first_moves(problem, instance: Instance, solution: torch.Tensor, lookahead: int,
                        atol: float = 1e-9, rtol: float = 1e-9, block_m1: int = 128,
                        block_m2: int = 8192):
    """Returns (target mask (B, M) over ``move_table(L)["pairs"]``, optimal value (B,))."""
    if lookahead not in (1, 2):
        raise ValueError("The exact oracle supports lookahead 1 or 2")
    if lookahead == 2 and problem.name not in ("tsp", "cvrp"):
        raise ValueError("lookahead 2 is implemented for the TSP and the CVRP")
    if getattr(problem, "n_ops", 1) > 1:
        raise ValueError("use flat_optimal_moves for multi-operator problems")
    values = one_step_values(problem, instance, solution)
    if lookahead == 2:
        table = move_table(solution.size(1), solution.device)
        values = values + _best_second_move(instance["cost_matrix"], solution, table, block_m1, block_m2,
                                            problem, instance)
    best = values.amin(dim=1)
    target = values <= (best + atol + rtol * best.abs()).unsqueeze(1)
    return target & torch.isfinite(values), best


@torch.no_grad()
def flat_optimal_moves(problem, instance: Instance, solution: torch.Tensor, atol: float = 1e-9,
                       rtol: float = 1e-9):
    """One-step oracle over every operator: (target mask (B, O*L*L) over flat move ids, best value (B,))."""
    values = problem.move_values(instance, solution)
    best = values.amin(dim=1)
    target = values <= (best + atol + rtol * best.abs()).unsqueeze(1)
    return target & torch.isfinite(values), best


@torch.no_grad()
def flat_optimal_moves_k2(problem, instance: Instance, solution: torch.Tensor, atol: float = 1e-9,
                          rtol: float = 1e-9, block: int = 32):
    """Exact two-step oracle over every operator (multi-operator CVRP): for each feasible first move m1,
    the best feasible second move on the solution after m1, by applying m1 and evaluating all moves.
    Returns (target mask (B, O*L*L) over flat move ids, best two-step value (B,))."""
    B, L = solution.shape
    v1 = problem.move_values(instance, solution)                       # (B, F), inf = infeasible
    F = v1.size(1)
    feas = torch.isfinite(v1)
    kmax = int(feas.sum(1).max())
    # feasible first moves of each row, padded with the first one (padding is masked below)
    order = torch.argsort((~feas).float(), dim=1, stable=True)[:, :kmax]          # (B, kmax)
    valid = feas.gather(1, order)
    total = torch.full((B, F), float("inf"), device=solution.device)
    for a in range(0, kmax, block):
        ids = order[:, a:a + block]                                    # (B, c)
        ids = torch.where(valid[:, a:a + block], ids, order[:, :1])    # pad with a feasible move
        c = ids.size(1)
        moves = torch.stack([ids // L, ids % L], -1).reshape(B * c, 2)  # flat id = (op * L + i) * L + j
        rep = {k: v.repeat_interleave(c, 0) if torch.is_tensor(v) and v.dim() > 0 else v
               for k, v in instance.items()}
        s1 = problem.apply_move(rep, solution.repeat_interleave(c, 0), moves)
        v2 = problem.move_values(rep, s1).amin(dim=1).view(B, c)
        val = (v1.gather(1, ids) + v2).masked_fill(~valid[:, a:a + block], float("inf"))
        total.scatter_(1, ids, torch.minimum(total.gather(1, ids), val))
    best = total.amin(dim=1)
    target = total <= (best + atol + rtol * best.abs()).unsqueeze(1)
    return target & torch.isfinite(total), best

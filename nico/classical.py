"""Batched classical 2-opt searches that keep exploring after a local optimum (Phase 3 baselines).

Every method uses the same per-step unit as the learned policy: one evaluation of the full 2-opt
neighborhood of the current solution and at most one move. Budgets are given in steps and/or
seconds, and the best-so-far cost is traced per step for anytime curves.

    multistart   steepest 2-opt descent; at a local optimum, restart from a fresh random tour
    ils          iterated 2-opt: steepest descent; at a local optimum, return to the best tour and
                 apply ``kick`` random 2-opt moves with segment length <= ``kick_len`` (same move
                 class as the policy), or a random double-bridge (``perturbation="double_bridge"``,
                 a 4-opt move, reported separately)
    tabu         best admissible move each step (also worsening ones); a move is tabu if it would
                 re-insert an edge removed during the last ``tenure`` steps, unless it improves on
                 the best cost found (aspiration)

TSP only (symmetric costs).
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import torch

from .problems.base import Instance, apply_two_opt, tour_cost

EPS = 1e-7


def _move_values(problem, instance, sol):
    B, L = sol.shape
    return problem.move_values(instance, sol).view(B, -1, L, L)[:, 0]    # first operator (2-opt)


def _apply(problem, instance, sol, moves, n_ops):
    """moves (B, 2) = (op * L + i, j); plain 2-opt when the problem has one operator."""
    return apply_two_opt(sol, moves) if n_ops == 1 else problem.apply_move(instance, sol, moves)


def _random_tours(B, n, device, gen):
    return torch.rand(B, n, device=device, generator=gen).argsort(dim=1)


def _random_kicks(sol, k, max_len, gen):
    """Apply k random 2-opt moves (i, j) with 2 <= j - i <= max_len to every row (j <= L - 2,
    so the inadmissible pair (0, L - 1) never occurs)."""
    B, L = sol.shape
    hi = max(2, min(max_len, L - 3))
    for _ in range(k):
        span = torch.randint(2, hi + 1, (B,), device=sol.device, generator=gen)
        i = (torch.rand(B, device=sol.device, generator=gen) * (L - 1 - span)).long()
        sol = apply_two_opt(sol, torch.stack([i, i + span], 1))
    return sol


def _double_bridge(sol, gen):
    B, L = sol.shape
    cuts = torch.rand(B, L - 1, device=sol.device, generator=gen).argsort(1)[:, :3].sort(1).values + 1
    p1, p2, p3 = cuts[:, 0:1], cuts[:, 1:2], cuts[:, 2:3]
    pos = torch.arange(L, device=sol.device).view(1, L).expand(B, L)
    # new order: A C B D with A=[0,p1) B=[p1,p2) C=[p2,p3) D=[p3,L)
    la, lb, lc = p1, p2 - p1, p3 - p2
    src = torch.where(pos < la, pos,
          torch.where(pos < la + lc, pos - la + p2,
          torch.where(pos < la + lc + lb, pos - la - lc + p1, pos)))
    return sol.gather(1, src)


def _feasible_kicks(problem, instance, sol, k, max_len, gen):
    """k random 2-opt moves drawn uniformly among the admissible (e.g. capacity-feasible) ones with
    segment length <= max_len: the ILS perturbation for constrained problems (CVRP)."""
    B, L = sol.shape
    ar = torch.arange(L, device=sol.device)
    short = (ar.view(1, L) - ar.view(L, 1)) <= max_len
    for _ in range(k):
        ok = torch.isfinite(_move_values(problem, instance, sol)) & short
        w = ok.view(B, -1).float()
        w[w.sum(1) == 0] = 1.0 / (L * L)                      # no feasible move: keep the solution
        flat = torch.multinomial(w, 1, generator=gen).squeeze(1)
        moves = torch.stack([flat // L, flat % L], 1)
        has = ok.view(B, -1).any(1)
        sol = torch.where(has.view(B, 1), apply_two_opt(sol, moves), sol)
    return sol


@torch.no_grad()
def search(method: str, problem, instance: Instance, solution: torch.Tensor, *, steps: Optional[int] = None,
           time_budget_s: Optional[float] = None, kick: int = 3, kick_len: int = 50,
           perturbation: str = "two_opt", tenure: int = 8, seed: int = 0,
           record_trace: bool = False) -> Dict[str, torch.Tensor]:
    if steps is None and time_budget_s is None:
        raise ValueError("give steps and/or time_budget_s")
    B, L = solution.shape
    device = solution.device
    gen = torch.Generator(device=device).manual_seed(seed)
    cost_matrix = instance["cost_matrix"]
    sol = solution.clone()
    cost = tour_cost(cost_matrix, sol)
    best, best_sol = cost.clone(), sol.clone()
    rows = torch.arange(B, device=device)
    ops = tuple(getattr(problem, "operators", ("2opt",))) if getattr(problem, "name", "tsp") == "cvrp" else ("2opt",)
    O = len(ops)
    if method == "tabu":
        removed_at = torch.full((B, L, L), -10 ** 9, dtype=torch.long, device=device)
    trace, times = [best.clone()], [0.0]
    start = time.perf_counter()
    step = 0
    while True:
        if steps is not None and step >= steps:
            break
        if time_budget_s is not None and time.perf_counter() - start >= time_budget_s:
            break
        val = problem.move_values(instance, sol).view(B, O, L, L)        # inf = invalid
        if method == "tabu":
            a, b = sol, sol.roll(-1, 1)
            ai, aj = a.unsqueeze(2).expand(B, L, L), a.unsqueeze(1).expand(B, L, L)
            bi, bj = b.unsqueeze(2).expand(B, L, L), b.unsqueeze(1).expand(B, L, L)
            r = rows.view(B, 1, 1)
            recent = lambda u, v: (step - removed_at[r, u, v]) <= tenure
            # a move is tabu if it inserts a recently removed edge: 2-opt inserts (a_i, a_j), (b_i, b_j),
            # 2-opt* inserts (a_i, b_j), (a_j, b_i)
            tabu = torch.stack([recent(ai, aj) | recent(bi, bj) if op == "2opt" else recent(ai, bj) | recent(aj, bi)
                                for op in ops], 1)
            aspire = (cost.view(B, 1, 1, 1) + val) < best.view(B, 1, 1, 1) - EPS
            allowed = torch.where(tabu & ~aspire, torch.full_like(val, float("inf")), val)
            flat = allowed.view(B, -1).argmin(1)
            none = torch.isinf(allowed.view(B, -1).gather(1, flat.view(B, 1)).squeeze(1))
            flat = torch.where(none, val.view(B, -1).argmin(1), flat)
            i, j = (flat // L) % L, flat % L
            u1, v1, u2, v2 = sol[rows, i], sol[rows, (i + 1) % L], sol[rows, j], sol[rows, (j + 1) % L]
            for u, v in ((u1, v1), (u2, v2)):
                removed_at[rows, u, v] = step
                removed_at[rows, v, u] = step
            cost = cost + val.view(B, -1).gather(1, flat.view(B, 1)).squeeze(1)
            sol = _apply(problem, instance, sol, torch.stack([flat // L, j], 1), O)
        else:
            d, flat = val.view(B, -1).min(1)
            improving = d < -EPS
            moves = torch.stack([flat // L, flat % L], 1)                # (op * L + i, j)
            new = _apply(problem, instance, sol, moves, O)
            sol = torch.where(improving.view(B, 1), new, sol)
            cost = torch.where(improving, cost + d, cost)
            stuck = ~improving
            if bool(stuck.any()):
                idx = stuck.nonzero(as_tuple=True)[0]
                if method == "multistart":
                    sol[idx] = sol[idx].gather(1, _random_tours(idx.numel(), L, device, gen))
                elif method == "ils":
                    base = torch.where((cost < best).view(B, 1), sol, best_sol)[idx]
                    if getattr(problem, "name", "tsp") == "cvrp":   # random kicks must stay feasible
                        sub = {k: v[idx] if torch.is_tensor(v) and v.dim() > 0 else v for k, v in instance.items()}
                        base = _feasible_kicks(problem, sub, base, kick, kick_len, gen)
                    else:
                        base = _double_bridge(base, gen) if perturbation == "double_bridge" \
                            else _random_kicks(base, kick, kick_len, gen)
                    sol[idx] = base
                else:
                    raise ValueError(f"Unknown method {method!r}")
                cost[idx] = tour_cost(cost_matrix[idx], sol[idx])
        better = cost < best - EPS
        best = torch.where(better, cost, best)
        best_sol[better] = sol[better]
        step += 1
        if record_trace:
            trace.append(best.clone())
            times.append(time.perf_counter() - start)
    # exact re-evaluation guards against drift of the incremental costs
    best = tour_cost(cost_matrix, best_sol)
    out = {"best_cost": best, "best_solution": best_sol, "steps": torch.tensor(step),
           "seconds": torch.tensor(time.perf_counter() - start)}
    if record_trace:
        out["trace"] = torch.stack(trace, 1)
        out["trace_time"] = torch.tensor(times)
    return out


# ----------------------------------------------------------------------------------------------
# Greedy 3-opt (table baseline): steepest descent over all seven non-identity reconnections of
# every triple of tour edges. With tour positions i < j < k, the removed edges are (t_i, t_i+1),
# (t_j, t_j+1), (t_k, t_k+1); S1 = t_i+1..t_j and S2 = t_j+1..t_k are reconnected as
#   1: S1' S2   2: S1 S2'   3: S2' S1'   (2-opt moves)
#   4: S1' S2'  5: S2 S1    6: S2 S1'   7: S2' S1   (pure 3-opt moves), with ' = reversed.
# Each added length is A[i, j] + B[i, k] + C[j, k] over the distance matrix in tour order, so a
# chunk of i's is evaluated as one broadcast sum over the full (j, k) grid.

def _three_opt_terms(P):
    """Per reconnection: the (i,j), (i,k), (j,k) term matrices of its added length, P in tour order."""
    Pn = P.roll(-1, 2)                  # P[x, y+1]
    Np = P.roll(-1, 1)                  # P[x+1, y]
    NN = Np.roll(-1, 2)                 # P[x+1, y+1]
    L = P.size(1)
    idx = torch.arange(L, device=P.device)
    r = P[:, idx, (idx + 1) % L]                      # removed edge lengths r_x = P[x, x+1]
    row, col = r.unsqueeze(2).expand_as(P), r.unsqueeze(1).expand_as(P)   # M[x, y] = r_x / r_y
    zero = torch.zeros_like(P)
    return [
        (P + NN, zero, col),                          # 1: a-c, b-d, keeps e-f (r_k)
        (row, zero, P + NN),                          # 2: keeps a-b (r_i); c-e, d-f
        (col, P + NN, zero),                          # 3: a-e, b-f, keeps c-d (r_j)
        (P, Np, NN),                                  # 4: a-c, b-e, d-f
        (Pn, Np, Pn),                                 # 5: a-d, e-b, c-f
        (Pn, NN, P),                                  # 6: a-d, b-f, c-e
        (NN, P, Pn),                                  # 7: a-e, d-b, c-f
    ], r


def _apply_three_opt(t, i, j, k, kind):
    a, s1, s2, z = t[: i + 1], t[i + 1: j + 1], t[j + 1: k + 1], t[k + 1:]
    mid = {1: (s1.flip(0), s2), 2: (s1, s2.flip(0)), 3: (s2.flip(0), s1.flip(0)), 4: (s1.flip(0), s2.flip(0)),
           5: (s2, s1), 6: (s2, s1.flip(0)), 7: (s2.flip(0), s1)}[kind]
    return torch.cat([a, *mid, z])


@torch.no_grad()
def three_opt_descent(cost_matrix: torch.Tensor, solution: torch.Tensor, chunk: int = 8,
                      max_steps: Optional[int] = None) -> torch.Tensor:
    """Steepest 3-opt descent (all seven reconnections) to a 3-opt local optimum."""
    B, L = solution.shape
    sol = solution.clone()
    active = torch.ones(B, dtype=torch.bool, device=sol.device)
    pos = torch.arange(L, device=sol.device)
    ok_jk = pos.view(L, 1) < pos.view(1, L)                                        # j < k
    step = 0
    while bool(active.any()) and (max_steps is None or step < max_steps):
        idx = active.nonzero(as_tuple=True)[0]
        t = sol[idx]
        b = t.size(0)
        P = cost_matrix[idx].gather(1, t.unsqueeze(2).expand(b, L, L)).gather(2, t.unsqueeze(1).expand(b, L, L))
        terms, r = _three_opt_terms(P)
        removed_jk = r.unsqueeze(2) + r.unsqueeze(1)                                # r_j + r_k on the (j,k) grid
        best = torch.full((b,), float("inf"), device=sol.device)
        arg = torch.zeros(b, 4, dtype=torch.long, device=sol.device)                # i, j, k, kind
        for lo in range(0, L - 2, chunk):
            I = pos[lo: min(L - 2, lo + chunk)]
            c = I.numel()
            mask = (I.view(c, 1, 1) < pos.view(1, L, 1)) & ok_jk.view(1, L, L)       # i < j < k
            base = r[:, I].view(b, c, 1, 1) + removed_jk.unsqueeze(1)               # removed length
            for kind, (A, Bt, C) in enumerate(terms, start=1):
                add = A[:, I].unsqueeze(3) + Bt[:, I].unsqueeze(2) + C.unsqueeze(1)     # (b, c, L, L)
                delta = (add - base).masked_fill(~mask, float("inf")).view(b, -1)
                d, flat = delta.min(1)
                better = d < best
                if bool(better.any()):
                    best = torch.where(better, d, best)
                    ii, rem = flat // (L * L), flat % (L * L)
                    new = torch.stack([I[ii], rem // L, rem % L, torch.full_like(ii, kind)], 1)
                    arg = torch.where(better.view(b, 1), new, arg)
        improving = best < -1e-6
        active[idx[~improving]] = False
        rows = improving.nonzero(as_tuple=True)[0]
        if rows.numel():
            new = torch.stack([_apply_three_opt(t[r], *arg[r].tolist()) for r in rows.tolist()])
            # accept only exact improvements: float32 deltas of cost-neutral moves can look negative on the GPU
            ok = tour_cost(cost_matrix[idx[rows]], new) < tour_cost(cost_matrix[idx[rows]], t[rows]) - 1e-6
            sol[idx[rows[ok]]] = new[ok]
            active[idx[rows[~ok]]] = False
        step += 1
    return sol


# ----------------------------------------------------------------------------------------------
# Learning-free policies over exact gains, run through the same loop as NICO (search.improve):
#   Gain-Boltzmann       pi(i, j) ∝ exp(beta * omega_ij), with NICO's tabu mask and temperature 1
#   simulated annealing  pi(i, j) ∝ exp(omega_ij / T_t), no tabu, T_t geometric from T0 to T1
#                        (rejection-free / heat-bath annealing: one full neighbourhood evaluation and
#                        one move per step, the same per-step unit as NICO)
# omega_ij is the gain of NICO's decoder: the cost decrease in units of the incumbent's mean edge length,
# clipped to [-10, 10] (search.policy_logits computes it because ``cfg.delta_bias`` is set).

class GainBoltzmann(torch.nn.Module):
    """Stand-in for a NICO model whose logits are beta * omega_ij (no learned term)."""

    def __init__(self, beta: float = 1.0):
        super().__init__()
        from .config import ModelConfig
        self.cfg = ModelConfig(delta_bias=True)
        self.beta = float(beta)

    def forward(self, instance, solution, move_hist, invalid, gain, progress=None):
        B = solution.size(0)
        return (self.beta * gain.view(B, -1)).masked_fill(invalid.reshape(B, -1), float("-inf"))

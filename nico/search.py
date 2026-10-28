"""Search state, action masking and policy-driven local search.

Used identically by imitation learning, reinforcement learning and evaluation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .config import SearchConfig
from .features import active_edges
from .problems.base import Instance


@dataclass
class SearchState:
    solution: torch.Tensor      # (B, L)
    move_hist: torch.Tensor     # (B, history_len, 2) recent moves for the history feature, -1 = empty
    tabu: torch.Tensor          # (B, tabu_tenure, 2) recent moves that are masked, -1 = empty
    # (B, 3) search progress: moves made, moves since the last new best, best cost so far (inf = unknown)
    progress: Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.progress is None:
            B = self.solution.size(0)
            self.progress = torch.zeros(B, 3, device=self.solution.device)
            self.progress[:, 2] = float("inf")

    @classmethod
    def initial(cls, solution: torch.Tensor, cfg: SearchConfig) -> "SearchState":
        B = solution.size(0)
        empty = lambda k: torch.full((B, k, 2), -1, dtype=torch.long, device=solution.device)
        return cls(solution, empty(cfg.history_len), empty(cfg.tabu_tenure))

    def index(self, idx: torch.Tensor) -> "SearchState":
        return SearchState(self.solution[idx], self.move_hist[idx], self.tabu[idx], self.progress[idx])

    def assign(self, idx: torch.Tensor, other: "SearchState") -> None:
        self.solution[idx] = other.solution
        self.move_hist[idx] = other.move_hist
        self.tabu[idx] = other.tabu
        self.progress[idx] = other.progress

    def repeat_interleave(self, repeats: int) -> "SearchState":
        r = lambda t: t.repeat_interleave(int(repeats), dim=0)
        return SearchState(r(self.solution), r(self.move_hist), r(self.tabu), r(self.progress))

    def clone(self) -> "SearchState":
        return SearchState(self.solution.clone(), self.move_hist.clone(), self.tabu.clone(), self.progress.clone())


def progress_features(problem, instance: Instance, state: SearchState) -> torch.Tensor:
    """(B, 3) size-free search-progress inputs: log(1 + moves / L), log(1 + moves since last best / L),
    (current cost - best cost) / best cost (0 before the first move)."""
    L = state.solution.size(1)
    steps, since, best = state.progress.unbind(-1)
    cost = problem.cost(instance, state.solution)
    gap = torch.where(torch.isfinite(best), (cost - best) / best.clamp_min(1e-9), torch.zeros_like(cost))
    return torch.stack([torch.log1p(steps / L), torch.log1p(since / L), gap], dim=-1)


def _push(buffer: torch.Tensor, moves: torch.Tensor) -> torch.Tensor:
    if buffer.size(1) == 0:
        return buffer
    return torch.cat([buffer[:, 1:], moves.unsqueeze(1)], dim=1)


def apply_tabu(logits: torch.Tensor, tabu: torch.Tensor, L: int) -> torch.Tensor:
    """Mask the recent moves (i, j) and their swaps (j, i); rows left without any
    admissible move fall back to the unmasked logits."""
    if tabu.size(1) == 0:
        return logits
    valid = (tabu >= 0).all(dim=-1)
    if not bool(valid.any()):
        return logits
    enc, j = tabu[..., 0].clamp_min(0), tabu[..., 1].clamp(0, L - 1)
    op, i = enc // L, enc % L                      # moves are encoded as (op * L + i, j)
    idx = torch.cat([enc * L + j, op * L * L + j * L + i], dim=1)
    ok = torch.cat([valid, valid], dim=1)
    rows = torch.arange(logits.size(0), device=logits.device).view(-1, 1).expand_as(idx)
    out = logits.clone()
    out[rows[ok], idx[ok]] = float("-inf")
    dead = torch.isneginf(out).all(dim=-1)
    if bool(dead.any()):
        out[dead] = logits[dead]
    return out


def candidate_moves(instance: Instance, solution: torch.Tensor, k: int, operators=None) -> torch.Tensor:
    """(B, L, L) bool, True for 2-opt moves (i, j) that insert at least one edge between k-nearest
    neighbours: sigma_j in kNN(sigma_i) or sigma_{j+1} in kNN(sigma_{i+1}) (symmetrised kNN graph)."""
    if "knn_graph" not in instance or instance["knn_graph"].size(-1) != instance["cost_matrix"].size(-1) \
            or instance.get("_knn_k") != k:
        d = instance["cost_matrix"].clone()
        d.diagonal(dim1=1, dim2=2).fill_(float("inf"))
        idx = d.topk(min(k, d.size(-1) - 1), dim=-1, largest=False).indices
        g = torch.zeros_like(d, dtype=torch.bool).scatter_(2, idx, True)
        instance["knn_graph"] = g | g.transpose(1, 2)
        instance["_knn_k"] = k
    g = instance["knn_graph"]
    B, L = solution.shape
    nxt = torch.roll(solution, -1, 1)
    if operators is not None:
        # CVRP: edges to the depot (node 0) are always candidates, and each operator is tested on the
        # edges it inserts
        g = g.clone()
        g[:, 0, :] = True
        g[:, :, 0] = True
    rows = lambda u: g.gather(1, u.unsqueeze(-1).expand(B, L, g.size(-1)))          # (B, L, V)
    link = lambda u, v: rows(u).gather(2, v.unsqueeze(1).expand(B, L, L))          # [i, j]: g[u_i, v_j]
    if operators is None:
        return link(solution, solution) | link(nxt, nxt)
    prv = torch.roll(solution, 1, 1)
    masks = []
    for name in operators:
        if name == "2opt":            # inserts (s_i, s_j), (s_i+1, s_j+1)
            masks.append(link(solution, solution) | link(nxt, nxt))
        elif name == "2opt*":         # inserts (s_i, s_j+1), (s_j, s_i+1)
            masks.append(link(solution, nxt) | link(nxt, solution))
        elif name == "relocate":      # customer s_p into edge (s_l, s_l+1)
            masks.append(link(solution, solution).transpose(1, 2) | link(nxt, solution).transpose(1, 2))
        else:                         # swap: s_q next to the neighbours of s_p and vice versa
            masks.append(link(prv, solution) | link(nxt, solution))
    return torch.stack(masks, 1)                                                  # (B, O, L, L)


def policy_logits(model, problem, instance: Instance, state: SearchState):
    """Returns (masked logits, logits without the tabu mask), both (B, L*L).

    With ``model.cfg.candidates`` > 0 only moves of the k-nearest-neighbour candidate set are admissible."""
    B, L = state.solution.shape
    k = model.cfg.candidates
    ops = getattr(problem, "operators", None) if problem.name == "cvrp" else None
    cand = candidate_moves(instance, state.solution, k, ops) if k > 0 else None
    prog = progress_features(problem, instance, state) if model.cfg.progress_features else None
    if model.cfg.delta_bias:
        values = problem.move_values(instance, state.solution)            # (B, O*L*L), inf = invalid
        if cand is not None:
            c = cand if cand.dim() == 4 else cand.unsqueeze(1)
            values = values.view(B, -1, L, L).masked_fill(~c, float("inf")).view(B, -1)
        invalid = ~torch.isfinite(values)
        edges = active_edges(problem.name, state.solution).sum(1).clamp_min(1.0)
        scale = (problem.cost(instance, state.solution) / edges).view(B, 1).clamp_min(1e-9)
        gain = torch.where(invalid, torch.zeros_like(values), -values / scale).clamp(-10.0, 10.0)
        base = model(instance, state.solution, state.move_hist, invalid.view(B, -1, L, L), gain, progress=prog)
    else:
        invalid = problem.invalid_moves(instance, state.solution)
        if cand is not None:
            c = cand if cand.dim() == invalid.dim() else cand.view(B, *([1] * (invalid.dim() - 3)), L, L)
            invalid = invalid | ~c
        base = model(instance, state.solution, state.move_hist, invalid, progress=prog)
    return apply_tabu(base, state.tabu, L), base


def reversal_index(L: int, device) -> torch.Tensor:
    """perm[m]: position in the reversed tour of the (reversed) solution edge at position m, (L - 2 - m) mod L."""
    return (L - 2 - torch.arange(L, device=device)) % L


def _map_positions(buffer: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Map the cut positions of a move buffer (B, K, 2) (TSP encoding (i, j), -1 = empty) through perm."""
    valid = buffer >= 0
    return torch.where(valid, perm[buffer.clamp_min(0)], buffer)


def symmetrized_logits(model, problem, instance: Instance, state: SearchState):
    """Policy averaged over the tour and its reverse and over the instance and its mirror image (TSP):
    mean of the four log-probability distributions over the same set of moves. Exactly invariant to tour
    reversal and to mirroring. Returns (masked logits, logits without the tabu mask) like policy_logits."""
    B, L = state.solution.shape
    perm = reversal_index(L, state.solution.device)
    mirrored = dict(instance)
    mirrored["coords"] = instance["coords"] * torch.tensor([-1.0, 1.0], device=instance["coords"].device)
    reversed_state = SearchState(state.solution.flip(1), _map_positions(state.move_hist, perm),
                                 _map_positions(state.tabu, perm), state.progress)
    ref = None
    logps = []
    for inst in (instance, mirrored):
        for rev in (False, True):
            _, base = policy_logits(model, problem, inst, reversed_state if rev else state)
            M = base.view(B, L, L).float()
            M = torch.where(torch.isfinite(M), M, M.transpose(1, 2))      # either order of a move
            if rev:
                M = M[:, perm][:, :, perm]
            if ref is None:
                ref = torch.isfinite(base.view(B, L, L))                  # admissible moves of this state
            logps.append(torch.log_softmax(M.masked_fill(~ref, float("-inf")).reshape(B, -1), dim=-1))
    avg = torch.stack(logps).mean(0)
    return apply_tabu(avg, state.tabu, L), avg


def select_moves(logits: torch.Tensor, L: int, sample: bool = True) -> torch.Tensor:
    if sample:
        flat = torch.distributions.Categorical(logits=logits.float()).sample()
    else:
        flat = logits.argmax(dim=-1)
    return torch.stack([torch.div(flat, L, rounding_mode="floor"), flat % L], dim=-1)


def move_log_prob(logits: torch.Tensor, moves: torch.Tensor, L: int) -> torch.Tensor:
    flat = (moves[:, 0] * L + moves[:, 1]).view(-1, 1)
    return torch.log_softmax(logits.float(), dim=-1).gather(1, flat).squeeze(1)


def advance(problem, instance: Instance, state: SearchState, moves: torch.Tensor) -> SearchState:
    solution = problem.apply_move(instance, state.solution, moves)
    steps, since, best = state.progress.unbind(-1)
    if not bool(torch.isfinite(best).all()):        # first move: the start is the best so far
        best = torch.where(torch.isfinite(best), best, problem.cost(instance, state.solution))
    cost = problem.cost(instance, solution)
    improved = cost < best - 1e-9
    progress = torch.stack([steps + 1, torch.where(improved, torch.zeros_like(since), since + 1),
                            torch.minimum(best, cost)], dim=-1)
    return SearchState(solution, _push(state.move_hist, moves), _push(state.tabu, moves), progress)


@torch.no_grad()
def warmup(model, problem, instance: Instance, state: SearchState, max_steps: int,
           return_best: bool = False, min_steps: int = 0):
    """Run t0 ~ U{0, ..., max_steps} policy moves per instance (sampled independently).

    Returns the new state and the best cost seen per instance (including the start),
    and with ``return_best`` also the best solution seen.
    """
    best = problem.cost(instance, state.solution)
    best_sol = state.solution.clone()
    if max_steps <= 0:
        return (state, best, best_sol) if return_best else (state, best)
    B, L = state.solution.shape
    t0 = torch.randint(min(min_steps, max_steps), max_steps + 1, (B,), device=state.solution.device)
    state = state.clone()
    for step in range(int(t0.max())):
        idx = (t0 > step).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            break
        sub_inst = {k: v[idx] if torch.is_tensor(v) else v for k, v in instance.items()}
        sub = state.index(idx)
        logits, _ = policy_logits(model, problem, sub_inst, sub)
        sub = advance(problem, sub_inst, sub, select_moves(logits, L, sample=True))
        state.assign(idx, sub)
        cost = problem.cost(sub_inst, sub.solution)
        better = cost < best[idx]
        best_sol[idx[better]] = sub.solution[better]
        best[idx] = torch.minimum(best[idx], cost)
    return (state, best, best_sol) if return_best else (state, best)


@torch.no_grad()
def two_opt_descent(problem, instance: Instance, solution: torch.Tensor, max_steps: int) -> torch.Tensor:
    """Steepest descent over the policy's own move set (feasible moves only) to a local optimum."""
    B, L = solution.shape
    solution = solution.clone()
    active = torch.ones(B, dtype=torch.bool, device=solution.device)
    for _ in range(int(max_steps)):
        idx = active.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            break
        sub_inst = {k: v[idx] if torch.is_tensor(v) else v for k, v in instance.items()}
        sol = solution[idx]
        best, flat = problem.move_values(sub_inst, sol).min(dim=1)
        improving = best < -1e-7
        active[idx[~improving]] = False
        if bool(improving.any()):
            rows = idx[improving]
            solution[rows] = problem.apply_move(
                {k: v[rows] if torch.is_tensor(v) else v for k, v in instance.items()},
                sol[improving], torch.stack([flat[improving] // L, flat[improving] % L], 1))
    return solution


@torch.no_grad()
def initial_solutions(problem, instance: Instance, cfg, num_routes=None) -> torch.Tensor:
    """Training start solutions (``cfg.train.init``).

    "random"          random permutation (CVRP: greedily split into routes)
    "mix:s,d"         CVRP: a share s of the rows use a sweep order instead of a random one,
                      and a share d of all rows is then taken to a 2-opt local optimum
    "savings:f"       CVRP: a share f of the rows start from the Clarke-Wright savings solution (the test-time
                      start), the others from a random split; rows are padded with empty routes to one width
    """
    mode = cfg.train.init
    kw = {"num_routes": num_routes} if num_routes is not None else {}
    if mode == "random":
        return problem.random_solutions(instance, **kw)
    if mode.startswith("savings:"):
        solution = problem.random_solutions(instance, **kw)
        rows = (torch.rand(solution.size(0), device=solution.device) < float(mode[8:])).nonzero(as_tuple=True)[0]
        if rows.numel():
            sub = {k: v[rows] if torch.is_tensor(v) else v for k, v in instance.items()}
            savings = problem.savings_solutions(sub, **kw).to(solution.device)
            L = max(solution.size(1), savings.size(1))
            solution = torch.nn.functional.pad(solution, (0, L - solution.size(1)))
            solution[rows] = torch.nn.functional.pad(savings, (0, L - savings.size(1)))
        return solution
    if not mode.startswith("mix:"):
        raise ValueError(f"Unknown train.init {mode!r}")
    sweep_frac, descent_frac = (float(x) for x in mode[4:].split(","))
    if hasattr(problem, "mixed_solutions"):
        solution = problem.mixed_solutions(instance, sweep_frac, **kw)
    else:
        solution = problem.random_solutions(instance, **kw)
    B, L = solution.shape
    rows = (torch.rand(B, device=solution.device) < descent_frac).nonzero(as_tuple=True)[0]
    if rows.numel():
        sub = {k: v[rows] if torch.is_tensor(v) else v for k, v in instance.items()}
        solution[rows] = two_opt_descent(problem, sub, solution[rows], 4 * L)
    return solution


def _interval_mask(moves: torch.Tensor, L: int) -> torch.Tensor:
    """(B, L) bool: the positions i..j+1 (mod L) touched by each 2-opt move (i, j)."""
    pos = torch.arange(L, device=moves.device).view(1, L)
    i, j = moves[:, :1], moves[:, 1:]
    inside = (pos >= i) & (pos <= j + 1)
    return inside | ((j + 1 >= L) & (pos == 0))


def select_disjoint_moves(logits: torch.Tensor, L: int, k: int):
    """Sample up to k moves per row without replacement (Gumbel top-k) and keep, in sampled order, those whose
    position ranges [i, j+1] do not overlap any kept move. Returns moves (B, k, 2) and keep mask (B, k).
    Moves with disjoint ranges touch disjoint parts of the tour, so they can be applied one after the other
    and their cost changes add up."""
    B = logits.size(0)
    g = -torch.log(-torch.log(torch.rand_like(logits, dtype=torch.float32).clamp_min(1e-20)))
    top = (logits.float() + g).topk(k, dim=-1)
    flat, finite = top.indices, torch.isfinite(top.values)
    moves = torch.stack([torch.div(flat, L, rounding_mode="floor"), flat % L], dim=-1)
    occupied = torch.zeros(B, L, dtype=torch.bool, device=logits.device)
    keep = torch.zeros(B, k, dtype=torch.bool, device=logits.device)
    for c in range(k):
        span = _interval_mask(moves[:, c], L)
        ok = finite[:, c] & ~(occupied & span).any(dim=1)
        keep[:, c] = ok
        occupied |= span & ok.unsqueeze(1)
    keep[:, 0] |= finite[:, 0]          # the first sampled move is always applied (as with one move per step)
    return moves, keep


@torch.no_grad()
def improve(model, problem, instance: Instance, solution: torch.Tensor, steps: int,
            cfg: SearchConfig, sample: bool = True, time_budget_s: Optional[float] = None,
            record_trace: bool = False) -> Dict[str, torch.Tensor]:
    """Policy-driven improvement for ``steps`` moves; returns the best solutions found."""
    state = SearchState.initial(solution, cfg)
    L = solution.size(1)
    cost = problem.cost(instance, solution)
    best, best_solution = cost.clone(), solution.clone()
    trace: List[torch.Tensor] = [best.clone()] if record_trace else []
    times: List[float] = [0.0] if record_trace else []
    start = time.perf_counter()
    T0 = cfg.temperature
    T1 = cfg.temperature_end if cfg.temperature_end is not None else T0
    for step in range(int(steps)):
        if time_budget_s is not None and time.perf_counter() - start >= time_budget_s:
            break
        if cfg.symmetrize:
            logits, _ = symmetrized_logits(model, problem, instance, state)
        else:
            logits, _ = policy_logits(model, problem, instance, state)
        T = T0 * (T1 / T0) ** (step / max(int(steps) - 1, 1))
        if cfg.adaptive_temp > 0:      # broader sampling while the search stagnates
            since = state.progress[:, 1]
            T_row = T * (1.0 + cfg.adaptive_temp * torch.log1p(since / L))
            logits = logits / T_row.view(-1, 1)
        elif T != 1.0:
            logits = logits / T
        k = cfg.moves_per_step if cfg.moves_per_step > 0 else max(1, L // 100)
        if k == 1 or not sample:
            state = advance(problem, instance, state, select_moves(logits, L, sample=sample))
        else:
            moves, keep = select_disjoint_moves(logits, L, k)
            for c in range(k):
                rows = keep[:, c].nonzero(as_tuple=True)[0]
                if rows.numel() == 0:
                    continue
                sub_inst = {key: v[rows] if torch.is_tensor(v) else v for key, v in instance.items()}
                state.assign(rows, advance(problem, sub_inst, state.index(rows), moves[rows, c]))
        cost = problem.cost(instance, state.solution)
        better = cost < best
        best = torch.where(better, cost, best)
        best_solution[better] = state.solution[better]
        if record_trace:
            trace.append(best.clone())
            if best.is_cuda:     # wall-clock per step: wait for the queued kernels
                torch.cuda.synchronize(best.device)
            times.append(time.perf_counter() - start)
    out = {"best_cost": best, "best_solution": best_solution}
    if record_trace:
        out["trace"] = torch.stack(trace, dim=1)
        out["trace_time"] = torch.tensor(times)
    return out

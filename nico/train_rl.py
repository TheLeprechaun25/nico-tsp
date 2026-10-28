"""Stage 2: critic-free group-based reinforcement learning (Algorithm 2)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import torch

from .config import Config
from .problems.base import Instance, repeat_instance
from .search import (SearchState, advance, initial_solutions, move_log_prob, policy_logits, select_moves,
                     warmup)


@dataclass
class Segment:
    instance: Instance              # (B*G, ...) group-expanded instances
    states: List[SearchState]       # state before each of the T moves
    moves: List[torch.Tensor]       # (B*G, 2) per step
    logp_old: torch.Tensor          # (T, B*G)
    advantages: torch.Tensor        # (T, B*G), zero outside the credited prefix
    groups: int                     # B
    group_size: int                 # G
    zero_signal_frac: float


def group_advantages(init_cost, best_trace, best_time, B: int, G: int) -> torch.Tensor:
    """Shared-cutoff group-relative advantages (Eqs. 11-12).

    init_cost  (B*G,)    C_ref, best cost reached during warmup
    best_trace (T, B*G)  best cost after each step
    best_time  (B*G,)    last step at which the rollout improved its best, -1 if never

    t* is the step at which the group-best score is first reached (ties broken by
    the earliest time). Every member is scored at t*, advantages are centred within
    the group, divided by (t* + 1) and broadcast to steps t <= t*.
    """
    T = best_trace.size(0)
    score_trace = (init_cost.unsqueeze(0) - best_trace).clamp_min(0.0) / init_cost.clamp_min(1e-6).unsqueeze(0)
    final = score_trace[-1].view(B, G)
    bt = best_time.view(B, G)
    bt_eff = torch.where(bt < 0, torch.full_like(bt, T), bt)
    is_max = final == final.max(dim=1, keepdim=True).values
    t_first = torch.where(is_max, bt_eff, torch.full_like(bt_eff, T + 1)).min(dim=1, keepdim=True).values
    winner = (is_max & (bt_eff == t_first)).float().argmax(dim=1)
    t_star = bt.gather(1, winner.unsqueeze(1)).squeeze(1)                       # (B,), -1 = no improvement

    idx = t_star.clamp_min(0).view(1, B, 1).expand(1, B, G)
    score = score_trace.view(T, B, G).gather(0, idx).squeeze(0)
    score = torch.where((t_star < 0).view(B, 1), torch.zeros_like(score), score)
    denom = torch.where(t_star >= 0, t_star + 1, torch.ones_like(t_star)).float().view(B, 1)
    adv = (score - score.mean(dim=1, keepdim=True)) / denom
    credited = torch.arange(T, device=adv.device).view(T, 1, 1) <= t_star.view(1, B, 1)
    return (adv.view(1, B, G) * credited).reshape(T, B * G)


def final_cost_advantages(init_cost, final_cost, B: int, G: int, T: int) -> torch.Tensor:
    """Group-centred relative final-cost improvement, divided by T and credited to all T steps."""
    score = ((init_cost - final_cost) / init_cost.clamp_min(1e-6)).view(B, G)
    adv = (score - score.mean(dim=1, keepdim=True)) / T
    return adv.reshape(1, B * G).expand(T, B * G).contiguous()


def best_reached_advantages(init_cost, run_min, run_min_time, B: int, G: int, T: int) -> torch.Tensor:
    """Group-centred score of the lowest cost each rollout reaches (even if worse than C_ref),
    divided by (t_min + 1) and credited to the steps up to the time it was reached."""
    score = ((init_cost - run_min) / init_cost.clamp_min(1e-6)).view(B, G)
    tm = run_min_time.view(B, G)
    adv = (score - score.mean(dim=1, keepdim=True)) / (tm + 1).float()
    credited = torch.arange(T, device=adv.device).view(T, 1, 1) <= tm.view(1, B, G)
    return (adv.view(1, B, G) * credited).reshape(T, B * G)


def rotate_group_coords(instance: Instance, B: int, G: int, R: int) -> Instance:
    """Rotate the coordinates of each block of G/R group members by 2*pi*r/R about the centroid.

    Only model inputs change: the objective (``cost_matrix``) is rotation invariant.
    """
    if R <= 1:
        return instance
    coords = instance["coords"].view(B, R, G // R, *instance["coords"].shape[1:])
    centre = coords.mean(dim=-2, keepdim=True)
    angles = torch.arange(R, device=coords.device, dtype=coords.dtype) * (2 * math.pi / R)
    c, s = torch.cos(angles).view(1, R, 1, 1), torch.sin(angles).view(1, R, 1, 1)
    x, y = (coords - centre)[..., 0], (coords - centre)[..., 1]
    rotated = torch.stack([c * x - s * y, s * x + c * y], dim=-1) + centre
    out = dict(instance)
    out["coords"] = rotated.reshape_as(instance["coords"])
    return out


@torch.no_grad()
def _sample_chunked(model, problem, instance: Instance, state: SearchState, L: int, chunk: int):
    """Sampled moves and their log-probabilities, in chunks of ``chunk`` rows to bound memory."""
    rows = state.solution.size(0)
    chunk = chunk if chunk > 0 else rows
    moves, logps = [], []
    for lo in range(0, rows, chunk):
        idx = torch.arange(lo, min(rows, lo + chunk), device=state.solution.device)
        sub_inst = {k: v.index_select(0, idx) if torch.is_tensor(v) else v for k, v in instance.items()}
        logits, _ = policy_logits(model, problem, sub_inst, state.index(idx))
        move = select_moves(logits, L, sample=True)
        moves.append(move)
        logps.append(move_log_prob(logits, move, L))
    return torch.cat(moves), torch.cat(logps)


def rollout_horizon(rl, L: int) -> int:
    """Rollout length: rl.horizon, or max(horizon_min, round(horizon_mult * L)) when horizon_mult > 0."""
    if rl.horizon_mult > 0:
        return max(int(rl.horizon_min), int(round(rl.horizon_mult * L)))
    return int(rl.horizon)


@torch.no_grad()
def collect_segment(frozen, problem, instance: Instance, cfg: Config, solution=None,
                    return_best: bool = False):
    """Roll out G sampled trajectories of T moves per instance with the behaviour policy.

    With ``return_best`` also returns the best solution and cost found per instance
    (over the warmup and all G rollouts), used to update the state pool."""
    rl = cfg.train.rl
    B = instance["cost_matrix"].size(0)
    G = max(1, rl.group_size)
    if solution is None:
        solution = initial_solutions(problem, instance, cfg)
    L = solution.size(1)
    H = rollout_horizon(rl, L)
    lo, hi = 0, int(cfg.train.warmup_mult * L)
    if rl.late_start_frac > 0 and float(torch.rand(())) < rl.late_start_frac:   # start from late-search states
        lo, hi = int(rl.late_start_mult[0] * L), int(rl.late_start_mult[1] * L)
    state, c_ref, warm_sol = warmup(frozen, problem, instance, SearchState.initial(solution, cfg.search),
                                    hi, return_best=True, min_steps=lo)
    best_sol = warm_sol.repeat_interleave(G, dim=0)

    inst = rotate_group_coords(repeat_instance(instance, G), B, G, rl.group_rotations)
    state = state.repeat_interleave(G)
    if rl.ref == "current":
        c_ref = problem.cost(inst, state.solution)          # state is already group-expanded
    elif rl.ref == "warmup_best":
        c_ref = c_ref.repeat_interleave(G)
    else:
        raise ValueError(f"Unknown train.rl.ref {rl.ref!r}")
    best = c_ref.clone()
    best_time = torch.full_like(best, -1, dtype=torch.long)
    run_min = torch.full_like(best, float("inf"))      # lowest cost reached, even if above C_ref
    run_min_time = torch.zeros_like(best_time)

    states, moves, logps, trace = [], [], [], []
    for t in range(H):
        states.append(state)
        move, logp = _sample_chunked(frozen, problem, inst, state, L, rl.ppo_microbatch)
        moves.append(move)
        logps.append(logp)
        state = advance(problem, inst, state, move)
        cost = problem.cost(inst, state.solution)
        improved = cost < best
        best = torch.where(improved, cost, best)
        best_time[improved] = t
        best_sol[improved] = state.solution[improved]
        trace.append(best.clone())
        lower = cost < run_min
        run_min = torch.where(lower, cost, run_min)
        run_min_time[lower] = t

    if rl.reward == "final":
        adv = final_cost_advantages(c_ref, cost, B, G, H)
    elif rl.reward == "best_rel":
        adv = best_reached_advantages(c_ref, run_min, run_min_time, B, G, H)
    elif rl.reward == "best":
        adv = group_advantages(c_ref, torch.stack(trace), best_time, B, G)
    else:
        raise ValueError(f"Unknown train.rl.reward {rl.reward!r}")
    no_signal = adv.view(H, B, G).abs().sum(dim=(0, 2)) <= 1e-12
    seg = Segment(inst, states, moves, torch.stack(logps), adv, B, G, float(no_signal.float().mean()))
    if not return_best:
        return seg
    group_best, which = best.view(B, G).min(dim=1)
    rows = torch.arange(B, device=best.device) * G + which
    return seg, best_sol[rows], group_best


def _rotation_consistency(model, problem, seg: Segment, state: SearchState, R: int) -> torch.Tensor:
    """Jensen-Shannon divergence between the policies on the R rotations of the same state."""
    B, G = seg.groups, seg.group_size
    per = G // R
    pick = lambda t: t.view(B, R, per, *t.shape[1:])[:, 0].reshape(B * per, *t.shape[1:])
    anchor_state = SearchState(pick(state.solution), pick(state.move_hist), pick(state.tabu), pick(state.progress))
    anchor_inst = {k: pick(v) if torch.is_tensor(v) else v for k, v in seg.instance.items()}
    inst = rotate_group_coords(repeat_instance(anchor_inst, R), B * per, R, R)
    logits, _ = policy_logits(model, problem, inst, anchor_state.repeat_interleave(R))
    p = torch.softmax(logits.float(), dim=-1).view(B * per, R, -1)
    m = p.mean(dim=1, keepdim=True)
    return (torch.xlogy(p, p.clamp_min(1e-12)) - p * torch.log(m.clamp_min(1e-12))).sum(dim=-1).mean()


def ppo_backward(model, problem, seg: Segment, cfg: Config, scaler=None, anchor=None) -> Dict[str, float]:
    """Accumulate the gradient of the PPO loss (mean clipped surrogate over credited
    steps, plus the optional rotation-consistency term) and return statistics.

    The loss is summed per time step and per chunk of ``rl.ppo_microbatch`` rollouts,
    with backward after each chunk, so memory is bounded by one chunk of one step
    while the gradient equals that of the full-batch loss.
    """
    rl = cfg.train.rl
    adv = seg.advantages
    if rl.normalize_advantages:
        nz = (adv != 0).float()
        n = nz.sum().clamp_min(1.0)
        mean = (adv * nz).sum() / n
        std = (((adv - mean) ** 2) * nz).sum().div(n).sqrt()
        adv = (adv - mean) / (std + 1e-8) * nz
    L = seg.states[0].solution.size(1)
    T, BG = adv.shape
    active = adv != 0
    n_active = int(active.sum())
    use_consistency = rl.rotation_consistency_coef > 0 and rl.group_rotations > 1
    chunk = rl.ppo_microbatch if rl.ppo_microbatch > 0 else BG
    if use_consistency:
        # rows of one group must stay together for the consistency term
        chunk = max(seg.group_size, chunk // seg.group_size * seg.group_size)
    # Rows with zero advantage contribute nothing to the clipped surrogate, so unless a term
    # needs every row (entropy bonus, rotation consistency) only the active rows are evaluated.
    all_rows = use_consistency or rl.entropy_coef > 0
    policy_sum, clip_count, consistency_sum, kl_sum = 0.0, 0.0, 0.0, 0.0
    use_anchor = anchor is not None and rl.kl_anchor_coef > 0
    for t, state in enumerate(seg.states):
        candidates = (torch.arange(BG, device=adv.device) if all_rows
                      else active[t].nonzero(as_tuple=True)[0])
        for lo in range(0, candidates.numel(), chunk):
            rows = candidates[lo:lo + chunk]
            a_t, act_t = adv[t, rows], active[t, rows]
            if not bool(act_t.any()) and not use_consistency:
                continue
            inst = {k: v.index_select(0, rows) if torch.is_tensor(v) else v for k, v in seg.instance.items()}
            st = state.index(rows)
            logits, _ = policy_logits(model, problem, inst, st)
            logp = move_log_prob(logits, seg.moves[t][rows], L)
            if rl.entropy_coef > 0:
                lp = torch.log_softmax(logits.float(), dim=-1)
                entropy = -(lp.exp() * lp.clamp_min(-1e9)).nansum(-1)
            ratio = torch.exp(logp - seg.logp_old[t, rows])
            surrogate = torch.min(ratio * a_t, ratio.clamp(1 - rl.ppo_clip, 1 + rl.ppo_clip) * a_t)
            loss = -(surrogate * act_t).sum() / max(n_active, 1)
            if rl.entropy_coef > 0:
                loss = loss - rl.entropy_coef * entropy.sum() / (BG * T)
            if use_anchor:   # KL(p_anchor || p_theta) over the admissible moves of each evaluated state
                with torch.no_grad():
                    a_logits, _ = policy_logits(anchor, problem, inst, st)
                    lp_a = torch.log_softmax(a_logits.float(), dim=-1)
                lp = torch.log_softmax(logits.float(), dim=-1)
                both = torch.isfinite(lp_a) & torch.isfinite(lp)
                kl = (lp_a.exp() * (lp_a - lp)).masked_fill(~both, 0.0).sum(-1)
                loss = loss + rl.kl_anchor_coef * kl.sum() / max(n_active, 1)
                kl_sum += float(kl.detach().sum()) / max(n_active, 1)
            policy_sum += float(loss.detach())
            clip_count += float((((ratio - 1).abs() > rl.ppo_clip) & act_t).sum())
            if use_consistency:
                sub = Segment(inst, None, None, None, None, rows.numel() // seg.group_size, seg.group_size, 0.0)
                cons = _rotation_consistency(model, problem, sub, st, rl.group_rotations)
                weight = rows.numel() / BG / T
                consistency_sum += float(cons.detach()) * weight
                loss = loss + rl.rotation_consistency_coef * cons * weight
            if loss.requires_grad:
                (scaler.scale(loss) if scaler is not None and scaler.is_enabled() else loss).backward()
    return {
        "rl/policy_loss": policy_sum,
        "rl/active_frac": float(active.float().mean()),
        "rl/clip_frac": clip_count / max(n_active, 1),
        "rl/zero_signal_frac": seg.zero_signal_frac,
        **({"rl/consistency": consistency_sum} if use_consistency else {}),
        **({"rl/kl_anchor": kl_sum} if use_anchor else {}),
    }


def train_rl_epoch(model, frozen, optimizer, problem, source, cfg: Config, device,
                   pool=None, scaler=None, anchor=None) -> Dict[str, float]:
    rl = cfg.train.rl
    updates = math.ceil(cfg.train.epoch_size / rl.batch_size)
    totals: Dict[str, float] = {}
    for u in range(updates):
        n = int(torch.randint(rl.size_range[0], rl.size_range[1] + 1, (1,)))
        frozen.eval()
        if pool is not None:
            instance, start, idx = pool.sample(rl.batch_size, n)
            seg, best_sol, best_cost = collect_segment(frozen, problem, instance, cfg, start, return_best=True)
            pool.update(n, idx, best_sol, best_cost)
        else:
            instance = source.sample(rl.batch_size, n, device)
            seg = collect_segment(frozen, problem, instance, cfg)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        stats = ppo_backward(model, problem, seg, cfg, scaler, anchor=anchor)
        amp = scaler is not None and scaler.is_enabled()
        if amp:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
        if amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        for k, v in stats.items():
            totals[k] = totals.get(k, 0.0) + v / updates
        if (u + 1) % cfg.train.behavior_refresh == 0:
            frozen.load_state_dict(model.state_dict())
    return totals

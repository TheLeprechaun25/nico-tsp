import math, copy, os, random
from bisect import bisect_left
from dataclasses import dataclass
from typing import List, Dict, Tuple
import torch
from tqdm import tqdm

from problem_tsp import TSP
from validation import (
    validate,
    _get_tabu_cfg,
    _init_tabu_state,
    _apply_tabu_mask,
    _select_actions_from_logits,
    _logp_of_actions_from_logits,
    _update_tabu_state,
)
from utils import move_to, get_inner_model, build_model_args, autocast_context


# =============================================================
# Data structures
# =============================================================
@dataclass
class SegmentBatch:
    """
    One truncated rollout segment batch for PPO updates.
    """
    coords: torch.Tensor                  # (B_eff, N, 2)

    solutions: List[torch.Tensor]         # length T, each (B_eff, N)
    last_k_actions: List[torch.Tensor]    # length T, each (B_eff, K, 2)
    tabu_action_hist: List[torch.Tensor]  # length T, each (B_eff, K_a, 2)
    tabu_edge_hist: List[torch.Tensor]    # length T, each (B_eff, K_e, 2)
    actions: List[torch.Tensor]           # length T, each (B_eff, 2)
    logp_old: List[torch.Tensor]          # length T, each (B_eff,)

    rewards: torch.Tensor                 # (T, B_eff)
    returns: torch.Tensor                 # (T, B_eff)
    advantages: torch.Tensor              # (T, B_eff)

    base_batch_size: int                  # B_base
    group_size: int                       # G
    frac_best_mean: float = 0.0
    zero_signal_inst_frac: float = 0.0


# =============================================================
# Competition-style rewards / advantages
# =============================================================
def _build_reward_score_trace(
    init_cost: torch.Tensor,         # (B_eff,)
    best_cost_t: torch.Tensor,       # (T, B_eff)
    reward_norm: str,
) -> torch.Tensor:
    reward_norm = str(reward_norm).lower()

    if reward_norm == "rel_init":
        denom = init_cost.clamp_min(1e-6).unsqueeze(0)
        return (init_cost.unsqueeze(0) - best_cost_t).clamp_min(0.0) / denom

    if reward_norm == "rel_current":
        prev_best_t = torch.cat([init_cost.unsqueeze(0), best_cost_t[:-1]], dim=0)
        step_gain = (prev_best_t - best_cost_t).clamp_min(0.0) / prev_best_t.clamp_min(1e-6)
        return torch.cumsum(step_gain, dim=0)

    raise ValueError(f"Unsupported rl_reward_norm '{reward_norm}'")


def compute_reward_returns_advantages(
    init_cost: torch.Tensor,                 # (B_eff,)
    best_cost_t: torch.Tensor,               # (T, B_eff)
    best_time: torch.Tensor,                 # (B_eff,) int64, -1 if never improved
    B_base: int,
    B_eff: int,
    T: int,
    G: int,
    reward_norm: str = "rel_init",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    device = init_cost.device
    assert B_eff == B_base * G

    score_trace = _build_reward_score_trace(init_cost, best_cost_t, reward_norm=reward_norm)
    score_full = score_trace[-1]
    score_bg_full = score_full.view(B_base, G)

    bt_bg = best_time.view(B_base, G)
    bt_eff = torch.where(bt_bg < 0, torch.full_like(bt_bg, T), bt_bg)

    max_score = score_bg_full.max(dim=1, keepdim=True).values
    is_max = (score_bg_full == max_score)

    big = torch.full_like(bt_eff, T + 1)
    bt_cand = torch.where(is_max, bt_eff, big)
    t_min = bt_cand.min(dim=1, keepdim=True).values

    winners_mask = is_max & (bt_eff == t_min)
    winner_rel = winners_mask.float().argmax(dim=1)

    t_star = bt_bg.gather(1, winner_rel.unsqueeze(1)).squeeze(1)

    frac_best_mean = float(winners_mask.float().mean(dim=1).mean().detach().item())

    t_star_idx = torch.where(t_star >= 0, t_star, torch.zeros_like(t_star)).long()

    score_trace_bg = score_trace.view(T, B_base, G)
    gather_idx = t_star_idx.view(1, B_base, 1).expand(1, B_base, G)
    score_bg = score_trace_bg.gather(dim=0, index=gather_idx).squeeze(0)

    no_credit = (t_star < 0).view(B_base, 1)
    score_bg = torch.where(no_credit, torch.zeros_like(score_bg), score_bg)

    t_idx = torch.arange(T, device=device).view(T, 1, 1)
    mask_bg = (t_idx <= t_star.view(1, B_base, 1))
    mask = mask_bg.expand(-1, -1, G).reshape(T, B_eff)

    # Normalize score by number of steps
    denom = torch.where(
        t_star >= 0,
        (t_star + 1).to(torch.float32),
        torch.ones_like(t_star, dtype=torch.float32),
    ).view(B_base, 1)
    per_step_score_bg = score_bg / denom
    per_step_score = per_step_score_bg.view(B_eff)

    rewards_t = per_step_score.unsqueeze(0).expand(T, B_eff) * mask.float()
    returns_t = rewards_t.clone()

    adv_bg = score_bg - score_bg.mean(dim=1, keepdim=True)
    adv_bg = adv_bg / denom
    advantages_t = adv_bg.view(B_eff).unsqueeze(0).expand(T, B_eff) * mask.float()

    return rewards_t.detach(), returns_t.detach(), advantages_t.detach(), frac_best_mean


def _index_tabu_state(tabu_state: Dict[str, torch.Tensor], idx: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "action_hist": tabu_state["action_hist"].index_select(0, idx),
        "edge_hist": tabu_state["edge_hist"].index_select(0, idx),
    }


def _scatter_tabu_state(tabu_state: Dict[str, torch.Tensor], idx: torch.Tensor, sub_state: Dict[str, torch.Tensor]) -> None:
    tabu_state["action_hist"][idx] = sub_state["action_hist"]
    tabu_state["edge_hist"][idx] = sub_state["edge_hist"]


def _expand_group_tensor(x: torch.Tensor, num_rotations: int, repeats_per_rotation: int) -> torch.Tensor:
    B_base = int(x.size(0))
    return (
        x.unsqueeze(1)
        .unsqueeze(2)
        .expand(B_base, num_rotations, repeats_per_rotation, *x.shape[1:])
        .contiguous()
        .view(B_base * num_rotations * repeats_per_rotation, *x.shape[1:])
    )


def _expand_group_rollout_state(
    problem,
    coords_base: torch.Tensor,
    sol_base: torch.Tensor,
    best_base: torch.Tensor,
    init_cost_base: torch.Tensor,
    lastk_base: torch.Tensor,
    tabu_state_base: Dict[str, torch.Tensor],
    group_size: int,
    num_rotations: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], int]:
    B_base, N, _ = coords_base.size()
    G = max(1, int(group_size))
    R = max(1, int(num_rotations))

    if R > G:
        raise ValueError(f"rl_group_rotations ({R}) must be <= effective group size ({G})")
    if G % R != 0:
        raise ValueError(f"effective group size ({G}) must be divisible by rl_group_rotations ({R})")

    if G == 1:
        tabu_state = {
            "action_hist": tabu_state_base["action_hist"],
            "edge_hist": tabu_state_base["edge_hist"],
        }
        return coords_base, sol_base, best_base, init_cost_base, lastk_base, tabu_state, G

    if R == 1:
        coords = coords_base.unsqueeze(1).expand(B_base, G, N, 2).contiguous().view(B_base * G, N, 2)
        sol = sol_base.unsqueeze(1).expand(B_base, G, N).contiguous().view(B_base * G, N)
        best = best_base.unsqueeze(1).expand(B_base, G).contiguous().view(B_base * G)
        init_cost = init_cost_base.unsqueeze(1).expand(B_base, G).contiguous().view(B_base * G)
        K_hist = int(lastk_base.size(1))
        lastk = lastk_base.unsqueeze(1).expand(B_base, G, K_hist, 2).contiguous().view(B_base * G, K_hist, 2)
        aK = int(tabu_state_base["action_hist"].size(1))
        eK = int(tabu_state_base["edge_hist"].size(1))
        tabu_state = {
            "action_hist": tabu_state_base["action_hist"].unsqueeze(1).expand(B_base, G, aK, 2).contiguous().view(B_base * G, aK, 2),
            "edge_hist": tabu_state_base["edge_hist"].unsqueeze(1).expand(B_base, G, eK, 2).contiguous().view(B_base * G, eK, 2),
        }
        return coords, sol, best, init_cost, lastk, tabu_state, G

    repeats_per_rotation = G // R
    coords_rot = problem.rotate_coords_evenly(coords_base, R)  # (B_base, R, N, 2)
    coords = (
        coords_rot.unsqueeze(2)
        .expand(B_base, R, repeats_per_rotation, N, 2)
        .contiguous()
        .view(B_base * G, N, 2)
    )
    sol = _expand_group_tensor(sol_base, R, repeats_per_rotation)
    best = _expand_group_tensor(best_base, R, repeats_per_rotation)
    init_cost = _expand_group_tensor(init_cost_base, R, repeats_per_rotation)
    lastk = _expand_group_tensor(lastk_base, R, repeats_per_rotation)
    tabu_state = {
        "action_hist": _expand_group_tensor(tabu_state_base["action_hist"], R, repeats_per_rotation),
        "edge_hist": _expand_group_tensor(tabu_state_base["edge_hist"], R, repeats_per_rotation),
    }
    return coords, sol, best, init_cost, lastk, tabu_state, G


def _rotation_consistency_loss_for_step(
    model,
    coords_t: torch.Tensor,
    sol_t: torch.Tensor,
    lastk_t: torch.Tensor,
    tabu_action_hist_t: torch.Tensor,
    tabu_edge_hist_t: torch.Tensor,
    base_batch_size: int,
    group_size: int,
    num_rotations: int,
    opts,
) -> torch.Tensor:
    B_eff, N, _ = coords_t.size()
    B_base = int(base_batch_size)
    G = int(group_size)
    R = max(1, int(num_rotations))

    if R <= 1 or G <= 1:
        return torch.zeros((), device=coords_t.device, dtype=torch.float32)
    if G % R != 0:
        raise ValueError(f"group_size ({G}) must be divisible by num_rotations ({R})")
    if B_eff != B_base * G:
        raise ValueError(f"Expected B_eff={B_base * G}, got {B_eff}")

    repeats_per_rotation = G // R

    coords_group = coords_t.reshape(B_base, R, repeats_per_rotation, N, 2)
    sol_group = sol_t.reshape(B_base, R, repeats_per_rotation, N)
    K_hist = int(lastk_t.size(1))
    lastk_group = lastk_t.reshape(B_base, R, repeats_per_rotation, K_hist, 2)
    action_hist_len = int(tabu_action_hist_t.size(1))
    edge_hist_len = int(tabu_edge_hist_t.size(1))
    tabu_action_group = tabu_action_hist_t.reshape(B_base, R, repeats_per_rotation, action_hist_len, 2)
    tabu_edge_group = tabu_edge_hist_t.reshape(B_base, R, repeats_per_rotation, edge_hist_len, 2)

    coords_anchor = coords_group[:, 0].contiguous().reshape(B_base * repeats_per_rotation, N, 2)
    sol_anchor = sol_group[:, 0].contiguous().reshape(B_base * repeats_per_rotation, N)
    lastk_anchor = lastk_group[:, 0].contiguous().reshape(B_base * repeats_per_rotation, K_hist, 2)
    tabu_action_anchor = tabu_action_group[:, 0].contiguous().reshape(B_base * repeats_per_rotation, action_hist_len, 2)
    tabu_edge_anchor = tabu_edge_group[:, 0].contiguous().reshape(B_base * repeats_per_rotation, edge_hist_len, 2)

    coords_rot = TSP.rotate_coords_evenly(coords_anchor, R)
    coords_aux = coords_rot.contiguous().reshape(B_base * G, N, 2)
    sol_aux = sol_anchor.unsqueeze(1).expand(B_base * repeats_per_rotation, R, N).contiguous().reshape(B_base * G, N)
    lastk_aux = lastk_anchor.unsqueeze(1).expand(B_base * repeats_per_rotation, R, K_hist, 2).contiguous().reshape(B_base * G, K_hist, 2)
    tabu_action_aux = (
        tabu_action_anchor.unsqueeze(1)
        .expand(B_base * repeats_per_rotation, R, action_hist_len, 2)
        .contiguous()
        .reshape(B_base * G, action_hist_len, 2)
    )
    tabu_edge_aux = (
        tabu_edge_anchor.unsqueeze(1)
        .expand(B_base * repeats_per_rotation, R, edge_hist_len, 2)
        .contiguous()
        .reshape(B_base * G, edge_hist_len, 2)
    )

    with autocast_context(opts):
        logits_all = model(
            coords_aux,
            sol_aux,
            last_k_actions=lastk_aux,
            tabu_edge_hist=tabu_edge_aux,
        )
    if logits_all.dim() == 3:
        logits_all = logits_all.view(B_base * G, -1)

    tabu_cfg = _get_tabu_cfg(opts)
    tabu_state_aux = {
        "action_hist": tabu_action_aux,
        "edge_hist": tabu_edge_aux,
    }
    masked_logits = _apply_tabu_mask(logits_all, sol_aux, tabu_state_aux, tabu_cfg)
    masked_logits = masked_logits.to(torch.float32).reshape(B_base * repeats_per_rotation, R, -1)

    has_nan = torch.isnan(masked_logits).any(dim=-1)
    has_any_finite = torch.isfinite(masked_logits).any(dim=-1)
    valid_groups = (~has_nan).all(dim=1) & has_any_finite.all(dim=1)
    if not bool(valid_groups.any()):
        return torch.zeros((), device=coords_t.device, dtype=torch.float32)

    masked_logits = masked_logits[valid_groups]
    logp = torch.log_softmax(masked_logits, dim=-1)
    probs = logp.exp()

    mean_probs = probs.mean(dim=1)
    log_mean_probs = torch.log(mean_probs.clamp_min(1e-12))
    js_terms = torch.xlogy(probs, probs.clamp_min(1e-12)) - probs * log_mean_probs.unsqueeze(1)
    js_div = js_terms.sum(dim=-1).mean()

    if not torch.isfinite(js_div):
        return torch.zeros((), device=coords_t.device, dtype=torch.float32)
    return js_div


def _sample_uniform_stratified_sizes(size_low: int, size_high: int, num_sizes: int) -> List[int]:
    size_low = int(size_low)
    size_high = int(size_high)
    K_eff = max(0, int(num_sizes))
    if K_eff <= 0:
        return []

    rng = max(1, int(size_high - size_low + 1))
    sizes: List[int] = []
    for i in range(K_eff):
        lo = size_low + (i * rng) // K_eff
        hi = size_low + ((i + 1) * rng) // K_eff - 1
        hi = max(lo, hi)
        sizes.append(random.randint(int(lo), int(hi)))
    random.shuffle(sizes)
    return sizes


def _sample_power_stratified_sizes(size_low: int, size_high: int, num_sizes: int, power: float) -> List[int]:
    size_low = int(size_low)
    size_high = int(size_high)
    K_eff = max(0, int(num_sizes))
    if K_eff <= 0:
        return []
    if size_low == size_high:
        return [size_low] * K_eff

    sizes = list(range(size_low, size_high + 1))
    count = float(len(sizes))
    weights = [((n - size_low + 1) / count) ** float(power) for n in sizes]

    cdf = []
    total = 0.0
    for w in weights:
        total += float(w)
        cdf.append(total)
    inv_total = 1.0 / max(total, 1e-12)
    cdf = [x * inv_total for x in cdf]
    cdf[-1] = 1.0

    sampled_sizes: List[int] = []
    for i in range(K_eff):
        u = (i + random.random()) / float(K_eff)
        idx = bisect_left(cdf, u)
        idx = min(idx, len(sizes) - 1)
        sampled_sizes.append(int(sizes[idx]))
    random.shuffle(sampled_sizes)
    return sampled_sizes


def _sample_power_iid_sizes(size_low: int, size_high: int, num_sizes: int, power: float) -> List[int]:
    size_low = int(size_low)
    size_high = int(size_high)
    K_eff = max(0, int(num_sizes))
    if K_eff <= 0:
        return []
    if size_low == size_high:
        return [size_low] * K_eff

    sizes = list(range(size_low, size_high + 1))
    count = float(len(sizes))
    weights = [((n - size_low + 1) / count) ** float(power) for n in sizes]
    return [int(n) for n in random.choices(sizes, weights=weights, k=K_eff)]


def _sample_rl_sizes(size_low: int, size_high: int, num_sizes: int, mode: str, power: float) -> List[int]:
    mode = str(mode).lower()
    if mode == "uniform_stratified":
        return _sample_uniform_stratified_sizes(size_low, size_high, num_sizes)
    if mode == "power_stratified":
        return _sample_power_stratified_sizes(size_low, size_high, num_sizes, power)
    if mode == "power_iid":
        return _sample_power_iid_sizes(size_low, size_high, num_sizes, power)
    raise ValueError(f"Unsupported rl_size_sampling mode '{mode}'")


# =============================================================
# Segment collection (behavior policy)
# =============================================================
@torch.no_grad()
def collect_segments_variable(
    problem,
    model_old,
    batch: Dict[str, torch.Tensor],
    opts,
) -> SegmentBatch:
    """
    Collect segments with variable warmup length per instance
    """
    device = opts.device
    batch = move_to(batch, device)
    model_old.eval()

    coords_base = batch["coords"]
    B_base, N, _ = coords_base.size()
    problem.size = N

    # init
    sol_base = move_to(problem.get_initial_solutions(B_base)[0], device)
    cost_base = problem.get_costs(coords_base, sol_base)
    best_base = cost_base.clone()

    K_hist = opts.action_hist_len
    lastk_base = torch.full((B_base, K_hist, 2), -1, device=device, dtype=torch.long)
    tabu_state_base, tabu_cfg = _init_tabu_state(B_base, device, opts)

    # ------------------ WARMUP ------------------
    warm_mult = float(getattr(opts, "T_warm_mult", 1.0))  # fallback
    T_warm_max = int(warm_mult * N)

    mode = str(getattr(opts, "warmup_mode", "uniform")).lower()

    if T_warm_max <= 0:
        t0_base = torch.zeros((B_base,), device=device, dtype=torch.long)

    elif mode == "uniform":
        # independent per instance
        t0_base = torch.randint(low=0, high=T_warm_max + 1, size=(B_base,), device=device)

    elif mode == "fixed":
        # ONE tfix per batch (random), shared by all instances
        tfix = int(torch.randint(low=0, high=T_warm_max + 1, size=(1,), device=device).item())
        t0_base = torch.full((B_base,), tfix, device=device, dtype=torch.long)

    else:
        # stratified per instance in [0..T_warm_max]
        i = torch.arange(B_base, device=device)
        lo = (i * (T_warm_max + 1)) // B_base
        hi = ((i + 1) * (T_warm_max + 1)) // B_base - 1
        hi = torch.maximum(hi, lo)

        u = torch.rand((B_base,), device=device)
        t0_base = lo + (u * (hi - lo + 1).to(torch.float32)).floor().long()
        t0_base = t0_base[torch.randperm(B_base, device=device)]

    t0_max = int(t0_base.max().item())
    for k in range(t0_max):
        active = (t0_base > k)
        if not active.any():
            break
        idx = active.nonzero(as_tuple=True)[0]

        coords_a = coords_base.index_select(0, idx)
        sol_a = sol_base.index_select(0, idx)
        lastk_a = lastk_base.index_select(0, idx)
        tabu_a = _index_tabu_state(tabu_state_base, idx)

        with autocast_context(opts):
            logits_all_a = model_old(
                coords_a,
                sol_a,
                last_k_actions=lastk_a,
                tabu_edge_hist=tabu_a["edge_hist"],
            )
        if logits_all_a.dim() == 3:
            logits_all_a = logits_all_a.view(logits_all_a.size(0), -1)
        masked_logits_a = _apply_tabu_mask(logits_all_a, sol_a, tabu_a, tabu_cfg)
        exch_a = _select_actions_from_logits(masked_logits_a, N, do_sample=True)

        prev_sol_a = sol_a
        sol_a = move_to(problem.step(sol_a, exch_a), device)
        _update_tabu_state(tabu_a, tabu_cfg, prev_sol_a, exch_a)
        cost_a = problem.get_costs(coords_a, sol_a)

        best_a = best_base.index_select(0, idx)
        new_best_a = torch.minimum(best_a, cost_a)

        sol_base[idx] = sol_a
        best_base[idx] = new_best_a
        _scatter_tabu_state(tabu_state_base, idx, tabu_a)

        if K_hist > 0:
            lastk_a = torch.roll(lastk_a, shifts=-1, dims=1)
            lastk_a[:, -1, :] = exch_a
            lastk_base[idx] = lastk_a

    # reference baseline
    cur_cost_base = problem.get_costs(coords_base, sol_base)
    ref_mode = getattr(opts, "init_cost_ref", "warmup_best")
    if ref_mode == "warmup_last":
        init_cost_base = cur_cost_base.clone()
        best_base = cur_cost_base.clone()
    else:
        init_cost_base = best_base.clone()

    # ------------------ expand to groups ------------------
    coords, sol, best, init_cost, lastk, tabu_state, G = _expand_group_rollout_state(
        problem,
        coords_base=coords_base,
        sol_base=sol_base,
        best_base=best_base,
        init_cost_base=init_cost_base,
        lastk_base=lastk_base,
        tabu_state_base=tabu_state_base,
        group_size=getattr(opts, "grpo_group_size", 1),
        num_rotations=getattr(opts, "rl_group_rotations", 1),
    )

    B_eff = coords.size(0)
    assert B_eff == B_base * G

    # ------------------ truncated window ------------------
    T = int(opts.rl_horizon)

    solutions: List[torch.Tensor] = []
    hist_lastk: List[torch.Tensor] = []
    tabu_action_hist: List[torch.Tensor] = []
    tabu_edge_hist: List[torch.Tensor] = []
    actions: List[torch.Tensor] = []
    logp_old: List[torch.Tensor] = []

    best_cost_trace: List[torch.Tensor] = []
    best_time = torch.full((B_eff,), -1, device=device, dtype=torch.long)

    for t in range(T):
        solutions.append(sol.clone())
        hist_lastk.append(lastk.clone())
        tabu_action_hist.append(tabu_state["action_hist"].clone())
        tabu_edge_hist.append(tabu_state["edge_hist"].clone())

        with autocast_context(opts):
            logits_all = model_old(
                coords,
                sol,
                last_k_actions=lastk,
                tabu_edge_hist=tabu_state["edge_hist"],
            )
        if logits_all.dim() == 3:
            logits_all = logits_all.view(B_eff, -1)
        masked_logits = _apply_tabu_mask(logits_all, sol, tabu_state, tabu_cfg)
        exch = _select_actions_from_logits(masked_logits, N, do_sample=True)
        log_lh = _logp_of_actions_from_logits(masked_logits, exch, N)
        actions.append(exch.clone())
        logp_old.append(log_lh.to(torch.float32).clone())

        prev_sol = sol
        sol = move_to(problem.step(sol, exch), device)
        _update_tabu_state(tabu_state, tabu_cfg, prev_sol, exch)
        cost = problem.get_costs(coords, sol)

        new_best = torch.minimum(best, cost)
        improved = (best - new_best) > 0
        best_time[improved] = t
        best = new_best

        if K_hist > 0:
            lastk = torch.roll(lastk, shifts=-1, dims=1)
            lastk[:, -1, :] = exch

        best_cost_trace.append(best.clone())

    best_cost_t = torch.stack(best_cost_trace, dim=0)

    rewards_t, returns_t, adv_t, frac_best_mean = compute_reward_returns_advantages(
        init_cost=init_cost,
        best_cost_t=best_cost_t,
        best_time=best_time,
        B_base=B_base,
        B_eff=B_eff,
        T=T,
        G=G,
        reward_norm=getattr(opts, "rl_reward_norm", "rel_init"),
    )

    adv_bg = adv_t.view(T, B_base, G)
    has_signal = (adv_bg.abs().sum(dim=(0, 2)) > 1e-12)
    zero_signal_inst_frac = float((~has_signal).float().mean().item())

    return SegmentBatch(
        coords=coords,
        solutions=solutions,
        last_k_actions=hist_lastk,
        tabu_action_hist=tabu_action_hist,
        tabu_edge_hist=tabu_edge_hist,
        actions=actions,
        logp_old=logp_old,
        rewards=rewards_t,
        returns=returns_t,
        advantages=adv_t,
        base_batch_size=B_base,
        group_size=G,
        frac_best_mean=float(frac_best_mean),
        zero_signal_inst_frac=zero_signal_inst_frac,
    )


def ppo_loss_on_segment(model, seg: SegmentBatch, opts):
    device = opts.device
    T, B = seg.rewards.size()
    model.train()

    logp_old = torch.stack(seg.logp_old, dim=0).to(device)     # (T,B)
    actions = torch.stack(seg.actions,  dim=0).to(device)      # (T,B,2)
    adv = seg.advantages.to(device)                            # (T,B)
    coords_in = seg.coords.to(device)

    # normalize advantages over nonzero entries
    adv_norm = not getattr(opts, "no_adv_norm", False)
    if adv_norm:
        adv_mask = (adv != 0).float()
        denom = adv_mask.sum().clamp_min(1.0)
        adv_mean = (adv * adv_mask).sum() / denom
        adv_var = ((adv - adv_mean) ** 2 * adv_mask).sum() / denom
        adv = (adv - adv_mean) / (adv_var.sqrt() + 1e-8)
        adv = adv * adv_mask

    logp_new_list = []
    rotation_consistency_terms = []
    tabu_cfg = _get_tabu_cfg(opts)
    rotation_coef = float(getattr(opts, "rl_rotation_consistency_coef", 0.0))
    num_rotations = max(1, int(getattr(opts, "rl_group_rotations", 1)))

    for t in range(T):
        sol_t = seg.solutions[t].to(device)
        lastk_t = seg.last_k_actions[t].to(device)
        tabu_action_hist_t = seg.tabu_action_hist[t].to(device)
        tabu_edge_hist_t = seg.tabu_edge_hist[t].to(device)
        act_t = actions[t]

        with autocast_context(opts):
            logits_all_t = model(
                coords_in,
                sol_t,
                last_k_actions=lastk_t,
                tabu_edge_hist=tabu_edge_hist_t,
            )
        if logits_all_t.dim() == 3:
            logits_all_t = logits_all_t.view(B, -1)

        tabu_state_t = {
            "action_hist": tabu_action_hist_t,
            "edge_hist": tabu_edge_hist_t,
        }
        masked_logits_t = _apply_tabu_mask(logits_all_t, sol_t, tabu_state_t, tabu_cfg)
        logp_t = _logp_of_actions_from_logits(masked_logits_t, act_t, sol_t.size(1))

        logp_new_list.append(logp_t.to(torch.float32))
        if rotation_coef > 0.0 and num_rotations > 1 and seg.group_size > 1:
            rotation_consistency_terms.append(
                _rotation_consistency_loss_for_step(
                    model,
                    coords_t=coords_in,
                    sol_t=sol_t,
                    lastk_t=lastk_t,
                    tabu_action_hist_t=tabu_action_hist_t,
                    tabu_edge_hist_t=tabu_edge_hist_t,
                    base_batch_size=seg.base_batch_size,
                    group_size=seg.group_size,
                    num_rotations=num_rotations,
                    opts=opts,
                )
            )

    logp_new = torch.stack(logp_new_list, dim=0)  # (T,B)
    log_ratio = logp_new - logp_old
    ratio = torch.exp(log_ratio)

    clip_eps = float(getattr(opts, "ppo_clip", 0.2))
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    min_surr = torch.min(surr1, surr2)

    # -------- Active-aware PPO reduction --------
    active = (adv != 0)
    n_active = int(active.sum().item())
    n_total = int(active.numel())
    f = float(n_active / max(1, n_total))

    policy_loss_all = -min_surr.mean()
    if n_active > 0:
        policy_loss = -min_surr[active].mean()
    else:
        # no signal => no gradient contribution
        policy_loss = policy_loss_all * 0.0

    if rotation_consistency_terms:
        rotation_consistency_loss = torch.stack(rotation_consistency_terms).mean()
    else:
        rotation_consistency_loss = policy_loss_all.new_zeros(())
    if not torch.isfinite(rotation_consistency_loss):
        rotation_consistency_loss = policy_loss_all.new_zeros(())

    total_loss = policy_loss + rotation_coef * rotation_consistency_loss

    ratio_dev_mean = (ratio - 1.0).abs().mean()

    # where PPO actually uses clipped objective
    clip_used = (surr2 < surr1)
    clip_used_frac = clip_used[active].float().mean() if active.any() else torch.tensor(0.0, device=device)

    # how much clipping changes the objective
    clip_impact_mean = (surr1 - min_surr)[active].mean() if active.any() else torch.tensor(0.0, device=device)

    approx_kl = (logp_old - logp_new).mean()
    clipfrac = ((ratio - 1.0).abs() > clip_eps).float().mean()

    stats = {
        "train/total_loss": float(total_loss.detach().item()),
        "train/policy_loss": float(policy_loss.detach().item()),
        "train/policy_loss_all": float(policy_loss_all.detach().item()),
        "train/rotation_consistency_loss": float(rotation_consistency_loss.detach().item()),
        "train/active_frac": float(f),
        "train/n_active": float(n_active),  # store as float for logging consistency
        "train/n_total": float(n_total),

        "train/ratio_mean": float(ratio.detach().mean().item()),
        "train/ratio_dev_mean": float(ratio_dev_mean.detach().item()),
        "train/ratio_min": float(ratio.detach().min().item()),
        "train/ratio_max": float(ratio.detach().max().item()),
        "train/approx_kl": float(approx_kl.detach().item()),
        "train/clipfrac": float(clipfrac.detach().item()),
        "train/clip_used_frac": float(clip_used_frac.detach().item()),
        "train/clip_impact_mean": float(clip_impact_mean.detach().item()),

        "train/adv_mean": float(adv.detach().mean().item()),
        "train/adv_std": float(adv.detach().std(unbiased=False).item()),
        "train/reward_mean": float(seg.rewards.to(device).mean().detach().item()),
        "train/reward_std": float(seg.rewards.to(device).std(unbiased=False).detach().item()),
        "train/frac_best_mean": float(seg.frac_best_mean),
        "train/zero_signal_inst_frac": float(seg.zero_signal_inst_frac),
    }

    return total_loss, stats


def train_rl_epoch(problem, model, optimizer, epoch: int, val_datasets, opts):
    if opts.verbose:
        print("\n")
        print("|", format(f" RL training epoch {epoch} ", "*^60"), "|")
        print(f"Training with lr={optimizer.param_groups[0]['lr']:.3e} ", flush=True)

    size_low, size_high = map(int, getattr(opts, "rl_size_range", (opts.graph_size, opts.graph_size)))

    if opts.verbose:
        print(f"RL size range this epoch: [{size_low}, {size_high}]")
    K = max(1, int(getattr(opts, "sizes_per_update", 1)))

    rl_batch_size = int(getattr(opts, "rl_batch_size", None) or getattr(opts, "batch_size"))
    total_num_episodes = int(math.ceil(opts.epoch_size / rl_batch_size))
    num_updates_target = int(math.ceil(total_num_episodes / K))

    pbar = tqdm(
        total=total_num_episodes,
        disable=getattr(opts, "no_progress_bar", False),
        desc=f"Training RL | epoch {epoch}",
        bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
    )
    stat_keys = [
        "train/total_loss",
        "train/policy_loss", "train/policy_loss_all", "train/rotation_consistency_loss", "train/active_frac",
        "train/n_active", "train/n_total",
        "train/ratio_mean", "train/ratio_dev_mean",
        "train/ratio_min", "train/ratio_max",
        "train/approx_kl",
        "train/clipfrac", "train/clip_used_frac", "train/clip_impact_mean",
        "train/adv_mean", "train/adv_std",
        "train/reward_mean", "train/reward_std",
        "train/frac_best_mean", "train/zero_signal_inst_frac",
        "train/grad_norm_pre_clip",
        "train/grad_norm_post_clip",
    ]
    active_weighted_keys = {
        "train/total_loss",
        "train/policy_loss",
        "train/policy_loss_all", "train/policy_loss_active", "train/rotation_consistency_loss",
        "train/ratio_mean", "train/ratio_dev_mean", "train/ratio_min", "train/ratio_max",
        "train/approx_kl", "train/clipfrac", "train/clip_used_frac", "train/clip_impact_mean",
        "train/adv_mean", "train/adv_std",
        "train/reward_mean", "train/reward_std",
        "train/active_frac", "train/n_active", "train/n_total",
    }

    acc = {k: 0.0 for k in stat_keys}
    n_updates = 0
    n_episodes = 0
    n_ppo_steps = 0

    model.train()
    scaler = getattr(opts, "_grad_scaler", None)

    # snapshot behavior policy
    inner_model = get_inner_model(model)
    model_old = copy.deepcopy(inner_model).to(opts.device)
    model_old.eval()
    for p in model_old.parameters():
        p.requires_grad_(False)

    size_sampling_mode = str(getattr(opts, "rl_size_sampling", "uniform_stratified")).lower()
    size_sampling_power = float(getattr(opts, "rl_size_power", 2.0))
    sampled_sizes_epoch: List[int] = []

    for _ in range(num_updates_target):
        remaining = total_num_episodes - n_episodes
        K_eff = min(K, max(0, remaining))
        if K_eff <= 0:
            break

        sizes = _sample_rl_sizes(
            size_low=size_low,
            size_high=size_high,
            num_sizes=K_eff,
            mode=size_sampling_mode,
            power=size_sampling_power,
        )
        sampled_sizes_epoch.extend(int(n) for n in sizes)

        seg_list: List[SegmentBatch] = []
        for n in sizes:
            n = int(n)
            problem.size = n
            batch = problem.sample_batch(rl_batch_size, opts.graph_type, n, device=opts.device)
            seg_list.append(collect_segments_variable(problem, model_old, batch, opts))
            n_episodes += 1
            pbar.update(1)

        # update epochs over same collected data
        for _ in range(int(opts.ppo_epochs)):
            optimizer.zero_grad(set_to_none=True)

            W = 0.0
            step_sum = {k: 0.0 for k in stat_keys}

            clipfracs = []
            kls = []
            ratio_devs = []
            clip_used_fracs = []
            for seg in seg_list:
                loss_i, stats_i = ppo_loss_on_segment(model, seg, opts)

                clipfracs.append(stats_i.get("train/clipfrac", 0.0))
                kls.append(stats_i.get("train/approx_kl", 0.0))
                ratio_devs.append(stats_i.get("train/ratio_dev_mean", 0.0))
                clip_used_fracs.append(stats_i.get("train/clip_used_frac", 0.0))

                n_active = float(stats_i.get("train/n_active", 0.0))
                w_i = max(1.0, n_active)  # important: avoid W=0 if a segment has no actives

                scaled_loss = loss_i * w_i
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                W += w_i

                for k in stat_keys:
                    v = float(stats_i.get(k, 0.0))
                    if k in active_weighted_keys:
                        step_sum[k] += w_i * v
                    else:
                        step_sum[k] += 1.0 * v

            invW = 1.0 / max(W, 1e-9)

            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)

            # normalize grads to weighted-mean objective
            for g in optimizer.param_groups:
                for p in g["params"]:
                    if p.grad is not None:
                        p.grad.mul_(invW)

            # ---- grad norm diagnostics + clipping (ONCE) ----
            all_params = [p for g in optimizer.param_groups for p in g["params"] if p.grad is not None]

            grad_norm_pre = torch.sqrt(sum((p.grad.detach().float() ** 2).sum() for p in all_params)).item()

            # clip_grad_norm_ returns *pre-clip* norm; we also compute true post-clip norm
            _ = torch.nn.utils.clip_grad_norm_(all_params, opts.max_grad_norm)
            grad_norm_post = torch.sqrt(sum((p.grad.detach().float() ** 2).sum() for p in all_params)).item()

            # ---- accumulate stats correctly ----
            invS = 1.0 / float(max(1, len(seg_list)))
            for k in stat_keys:
                if k in ("train/grad_norm_pre_clip", "train/grad_norm_post_clip"):
                    continue
                if k in active_weighted_keys:
                    acc[k] += float(step_sum[k] * invW)
                else:
                    acc[k] += float(step_sum[k] * invS)

            # grad norms: one scalar per PPO step (do NOT scale by invW)
            acc["train/grad_norm_pre_clip"] += float(grad_norm_pre)
            acc["train/grad_norm_post_clip"] += float(grad_norm_post)

            # ---- optimizer step ----
            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            n_ppo_steps += 1
        n_updates += 1
        # refresh behavior policy
        if getattr(opts, "update_old_model_every_batch", False):
            freq = int(getattr(opts, "update_old_model_freq", 1))
            if freq > 0 and (n_updates % freq == 0):
                with torch.no_grad():
                    model_old.load_state_dict(get_inner_model(model).state_dict(), strict=True)

    pbar.close()

    # Save Checkpoint
    total_epochs = int(getattr(opts, "total_epochs", opts.num_il_epochs + opts.num_rl_epochs))
    if getattr(opts, "is_main_process", True) and opts.save_dir and (epoch == total_epochs - 1 or (opts.checkpoint_epochs and epoch % opts.checkpoint_epochs == 0)):
        if opts.verbose:
            print("Saving model and state...")
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.save(
            {
                "model": get_inner_model(model).state_dict(),
                "model_args": build_model_args(opts),
                "optimizer": optimizer.state_dict(),
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": cuda_rng_state,
            },
            os.path.join(opts.save_dir, f"rl_epoch-{epoch}.pt"),
        )

    denom_updates = max(1, n_ppo_steps)
    train_metrics = {k: float(v / denom_updates) for k, v in acc.items()}
    train_metrics["train/size_low"] = float(size_low)
    train_metrics["train/size_high"] = float(size_high)
    train_metrics["train/size_sampling_power"] = float(size_sampling_power)
    if sampled_sizes_epoch:
        train_metrics["train/sampled_size_mean"] = float(sum(sampled_sizes_epoch) / len(sampled_sizes_epoch))
        train_metrics["train/sampled_size_min"] = float(min(sampled_sizes_epoch))
        train_metrics["train/sampled_size_max"] = float(max(sampled_sizes_epoch))
    else:
        train_metrics["train/sampled_size_mean"] = float(size_low)
        train_metrics["train/sampled_size_min"] = float(size_low)
        train_metrics["train/sampled_size_max"] = float(size_low)

    val_metrics = {}
    if not getattr(opts, "skip_train_validation", False):
        model.eval()
        val_metrics = validate(problem, get_inner_model(model), val_datasets, opts)

    epoch_results = {
        "epoch": int(epoch),
        "lr": float(optimizer.param_groups[0]['lr']),
        **train_metrics,
        **val_metrics,
    }
    return epoch_results

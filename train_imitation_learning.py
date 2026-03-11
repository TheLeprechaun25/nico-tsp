import math
import copy
import os
import random
from typing import Dict, Optional, Tuple, List

import torch
from tqdm import tqdm

from validation import (
    validate,
    _init_tabu_state,
    _apply_tabu_mask,
    _select_actions_from_logits,
    _update_tabu_state,
)
from utils import clip_grad_norms, get_inner_model, move_to, build_model_args, autocast_context


# ============================================================
# Numerics / safety
# ============================================================
def _sanitize_and_normalize_logp_all(
    logp_all: torch.Tensor, *, step: int, N: int, tag: str = "logp_all"
) -> torch.Tensor:
    """
    Accepts:
      - logits with -inf on masked actions, OR
      - log-probs with -inf on masked actions.
    Returns:
      - log-probs normalized per row (sum exp = 1 over non-masked actions),
        still keeping -inf on masked actions.

    Raises if:
      - any NaN / +inf appears
      - any row has all entries masked (-inf everywhere)
    """
    if torch.isnan(logp_all).any():
        raise RuntimeError(f"{tag} has NaNs at step={step}, N={N}")
    if torch.isposinf(logp_all).any():
        raise RuntimeError(f"{tag} has +inf at step={step}, N={N}")

    logZ = torch.logsumexp(logp_all, dim=-1, keepdim=True)  # (B,1)
    all_masked = torch.isneginf(logZ.squeeze(-1))           # (B,)

    if all_masked.any():
        b0 = int(all_masked.nonzero(as_tuple=False)[0].item())
        mx = logp_all[b0].max().item()
        mn = logp_all[b0].min().item()
        raise RuntimeError(
            f"{tag}: all actions masked for {all_masked.sum().item()}/{logp_all.size(0)} rows "
            f"at step={step}, N={N}. Example row={b0}, min={mn}, max={mx}."
        )

    logp_all = logp_all - logZ

    if torch.isnan(logp_all).any():
        raise RuntimeError(f"{tag} became NaN after normalization at step={step}, N={N}")

    return logp_all


# ============================================================
# Tie handling / set-valued teachers (vectorized)
# ============================================================
def _tie_mask(vals: torch.Tensor, minv: torch.Tensor, atol: float, rtol: float) -> torch.Tensor:
    """
    vals: (...), minv: broadcastable to vals
    returns boolean mask for vals within atol + rtol*|minv| of minv.
    """
    minv_t = minv if torch.is_tensor(minv) else torch.as_tensor(minv, device=vals.device, dtype=vals.dtype)
    eps = minv_t.abs() * float(rtol) + float(atol)
    return vals <= (minv_t + eps)


def _topk_from_mask_gumbel(mask: torch.Tensor, K: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given mask (B,M) bool, returns:
      topk_idx: (B,K) long indices into M (padded if K > M)
      valid:    (B,K) bool whether that entry is truly in mask

    Unbiased random subset ordering among ties via Gumbel trick.
    """
    B, M = mask.shape
    K = int(K)

    if K <= 0:
        topk_idx = torch.empty((B, 0), device=mask.device, dtype=torch.long)
        valid = torch.empty((B, 0), device=mask.device, dtype=torch.bool)
        return topk_idx, valid

    # K cannot exceed M for topk; pad afterwards to keep fixed output shape
    K_eff = min(K, M)

    # gumbel noise
    u = torch.rand((B, M), device=mask.device, dtype=torch.float32).clamp_min(1e-12)
    g = -torch.log(-torch.log(u))
    scores = g.masked_fill(~mask, -1e9)  # non-ties -> very low

    topk_eff = scores.topk(K_eff, dim=1).indices  # (B,K_eff)
    valid_eff = mask.gather(1, topk_eff)          # (B,K_eff)

    if K_eff == K:
        return topk_eff, valid_eff

    # pad to (B,K)
    pad_idx = torch.zeros((B, K - K_eff), device=mask.device, dtype=torch.long)
    pad_valid = torch.zeros((B, K - K_eff), device=mask.device, dtype=torch.bool)

    topk_idx = torch.cat([topk_eff, pad_idx], dim=1)
    valid = torch.cat([valid_eff, pad_valid], dim=1)
    return topk_idx, valid


def _sample_one_from_mask(mask: torch.Tensor, fallback_argmin_vals: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Sample one index per row uniformly among True entries of mask (B,M) bool.
    """
    B, M = mask.shape
    w = mask.float()
    row_sum = w.sum(dim=1)
    ok = row_sum > 0

    out = torch.empty((B,), device=mask.device, dtype=torch.long)

    if ok.any():
        out[ok] = torch.multinomial(w[ok], num_samples=1).squeeze(1)

    if (~ok).any():
        if fallback_argmin_vals is None:
            # last resort: pick 0
            out[~ok] = 0
        else:
            out[~ok] = fallback_argmin_vals[~ok].argmin(dim=1)

    return out


# ============================================================
# Exact k-step teacher (optimal DP) for 2-opt in tour-position space
# ============================================================
_TEACHER_CACHE: Dict[Tuple[str, int], Dict[str, torch.Tensor]] = {}


def get_teacher_cache_2opt(N: int, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Cache per (device,N):
      - pairs: (M,2) valid (i,j) 2-opt position moves
      - maps:  (M,N) position mapping for applying each move to a tour in position space
      - pi,pj,pin1,pjn1: (M,) position indices for fast delta on mapped tours
      - lut: (N*N,) maps key=i*N+j -> m index (or -1)
    """
    key = (str(device), int(N))
    if key in _TEACHER_CACHE:
        return _TEACHER_CACHE[key]

    ar = torch.arange(N, device=device)
    mask = torch.zeros((N, N), dtype=torch.bool, device=device)

    # invalid: i==j, adjacent swaps, and enforce i<j (upper triangle)
    mask[ar, ar] = True
    mask[ar, (ar + 1) % N] = True
    mask[ar, (ar - 1) % N] = True
    ii = ar.view(N, 1)
    jj = ar.view(1, N)
    mask |= (ii >= jj)

    pairs = torch.nonzero(~mask, as_tuple=False).long()  # (M,2)
    M = pairs.size(0)

    # maps[m, p] = new position index (in old tour) for element at position p after applying move m
    arN = torch.arange(N, device=device).view(1, N).expand(M, N)  # (M,N)
    lo = (pairs[:, 0] + 1).view(M, 1)  # (M,1)
    hi = (pairs[:, 1]).view(M, 1)      # (M,1) inclusive
    in_seg = (arN >= lo) & (arN <= hi)
    maps = torch.where(in_seg, lo + hi - arN, arN).long()  # (M,N)

    pi = pairs[:, 0]
    pj = pairs[:, 1]
    pin1 = (pi + 1) % N
    pjn1 = (pj + 1) % N

    key_pairs = pi * N + pj  # (M,)
    lut = torch.full((N * N,), -1, device=device, dtype=torch.long)
    lut[key_pairs] = torch.arange(M, device=device, dtype=torch.long)

    _TEACHER_CACHE[key] = {
        "pairs": pairs,
        "maps": maps,
        "pi": pi,
        "pj": pj,
        "pin1": pin1,
        "pjn1": pjn1,
        "key_pairs": key_pairs,
        "lut": lut,
    }
    return _TEACHER_CACHE[key]


@torch.no_grad()
def _delta_for_pairs_batch_general(
    tours: torch.Tensor,    # (B,N) long (node ids)
    dists: torch.Tensor,    # (B,N,N) float
    pairs: torch.Tensor,    # (M,2) long positions
) -> torch.Tensor:
    """
    Returns (B,M) delta for applying each (i,j) 2-opt (positions) to each tour:
      delta = d(a_i,a_j)+d(b_i,b_j) - d(a_i,b_i) - d(a_j,b_j)
    where b = roll(a,-1).
    """
    B, N = tours.shape
    a = tours
    b = torch.roll(a, shifts=-1, dims=1)

    i = pairs[:, 0]  # (M,)
    j = pairs[:, 1]  # (M,)

    ai = a[:, i]     # (B,M)
    aj = a[:, j]
    bi = b[:, i]
    bj = b[:, j]

    bidx = torch.arange(B, device=tours.device)[:, None]
    return (
        dists[bidx, ai, aj] + dists[bidx, bi, bj]
        - dists[bidx, ai, bi] - dists[bidx, aj, bj]
    )  # (B,M)


@torch.no_grad()
def _compose_maps(map_a: torch.Tensor, map_b: torch.Tensor) -> torch.Tensor:
    """
    map_a: (A,N), map_b: (B,N)
    returns map_ab: (A,B,N) where applying a then b equals gather by map_ab.
    """
    A, N = map_a.shape
    B = map_b.shape[0]
    map_a_exp = map_a[:, None, :].expand(A, B, N)   # (A,B,N)
    map_b_exp = map_b[None, :, :].expand(A, B, N)   # (A,B,N)
    return map_a_exp.gather(2, map_b_exp)           # (A,B,N)


@torch.no_grad()
def _k2_min2_streaming(
    tours0: torch.Tensor,            # (B,N)
    dists: torch.Tensor,             # (B,N,N)
    maps: torch.Tensor,              # (M,N)
    pi: torch.Tensor, pj: torch.Tensor, pin1: torch.Tensor, pjn1: torch.Tensor,
    *,
    block_m1: int,
    block_m2: int,
) -> torch.Tensor:
    device = tours0.device
    B, N = tours0.shape
    M = maps.size(0)
    block_m1 = max(1, int(block_m1))
    block_m2 = max(1, int(block_m2))

    bidx = torch.arange(B, device=device)[:, None, None]  # (B,1,1)
    min2 = torch.full((B, M), float("inf"), device=device, dtype=torch.float32)

    for m1_lo in range(0, M, block_m1):
        m1_hi = min(M, m1_lo + block_m1)
        map1 = maps[m1_lo:m1_hi]  # (B1,N)
        B1 = m1_hi - m1_lo

        # IMPORTANT: match dim1 for gather (B,B1,N)
        tours0_exp = tours0[:, None, :].expand(B, B1, N)

        best_blk = torch.full((B, B1), float("inf"), device=device, dtype=torch.float32)

        for m2_lo in range(0, M, block_m2):
            m2_hi = min(M, m2_lo + block_m2)

            i = pi[m2_lo:m2_hi]
            j = pj[m2_lo:m2_hi]
            i1 = pin1[m2_lo:m2_hi]
            j1 = pjn1[m2_lo:m2_hi]

            # (B1,B2)
            mi = map1[:, i]
            mj = map1[:, j]
            mi1 = map1[:, i1]
            mj1 = map1[:, j1]

            # (B,B1,B2)
            mi3 = mi[None, :, :].expand(B, -1, -1)
            mj3 = mj[None, :, :].expand(B, -1, -1)
            mi13 = mi1[None, :, :].expand(B, -1, -1)
            mj13 = mj1[None, :, :].expand(B, -1, -1)

            # gather nodes from tours0 along last dim
            a_i = tours0_exp.gather(2, mi3)
            a_j = tours0_exp.gather(2, mj3)
            b_i = tours0_exp.gather(2, mi13)
            b_j = tours0_exp.gather(2, mj13)

            d = (
                dists[bidx, a_i, a_j] + dists[bidx, b_i, b_j]
                - dists[bidx, a_i, b_i] - dists[bidx, a_j, b_j]
            )  # (B,B1,B2)
            d = torch.nan_to_num(d, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))

            best_blk = torch.minimum(best_blk, d.amin(dim=2))  # (B,B1)

        min2[:, m1_lo:m1_hi] = best_blk

    return min2


@torch.no_grad()
def _k2_d2_for_chosen_m1(
    tours0: torch.Tensor,            # (B,N)
    dists: torch.Tensor,             # (B,N,N)
    maps: torch.Tensor,              # (M,N)
    pi: torch.Tensor, pj: torch.Tensor, pin1: torch.Tensor, pjn1: torch.Tensor,
    m1_idx: torch.Tensor,            # (B,)
    *,
    block_m2: int,
) -> torch.Tensor:
    """
    Compute d2_chosen[b, m2] = delta of move m2 on tour after applying chosen m1[b], exactly.
    Returns: (B,M) float32
    """
    device = tours0.device
    B, N = tours0.shape
    M = maps.size(0)
    block_m2 = max(1, int(block_m2))

    map1_b = maps.index_select(0, m1_idx)  # (B,N)
    d2 = torch.empty((B, M), device=device, dtype=torch.float32)

    bidx = torch.arange(B, device=device)[:, None]

    for m2_lo in range(0, M, block_m2):
        m2_hi = min(M, m2_lo + block_m2)

        i = pi[m2_lo:m2_hi]
        j = pj[m2_lo:m2_hi]
        i1 = pin1[m2_lo:m2_hi]
        j1 = pjn1[m2_lo:m2_hi]

        mi = map1_b[:, i]    # (B,B2)
        mj = map1_b[:, j]
        mi1 = map1_b[:, i1]
        mj1 = map1_b[:, j1]

        a_i = tours0.gather(1, mi)
        a_j = tours0.gather(1, mj)
        b_i = tours0.gather(1, mi1)
        b_j = tours0.gather(1, mj1)

        dblk = (
            dists[bidx, a_i, a_j] + dists[bidx, b_i, b_j]
            - dists[bidx, a_i, b_i] - dists[bidx, a_j, b_j]
        )  # (B,B2)
        dblk = torch.nan_to_num(dblk, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
        d2[:, m2_lo:m2_hi] = dblk.to(torch.float32)

    return d2


@torch.no_grad()
def exact_optimal_kstep_teacher_sets_batch(
    tours0: torch.Tensor,        # (B,N) long
    dists: torch.Tensor,         # (B,N,N) float
    k_steps: int,
    *,
    max_ties: int = 128,
    tie_atol: float = 1e-9,
    tie_rtol: float = 1e-9,
    block_m1: Optional[int] = None,
    block_m2: Optional[int] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """
    Finite-horizon DP teacher producing set-valued optimal actions per step.

    Returns:
      path_actions:   (B,k,2)  sampled from optimal sets (unbiased tie-breaking)
      opt_actions:    list length k; each is (B,K,2) padded
      opt_valid:      list length k; each is (B,K) bool
      best_total:     (B,) optimal k-step total delta (global optimum from tours0)
    """
    device = tours0.device
    B, N = tours0.shape
    assert k_steps in (1, 2, 3)

    if block_m1 is None or block_m1 <= 0 or block_m2 is None or block_m2 <= 0:
        bm1, bm2 = 128, 8192
        block_m1 = bm1 if (block_m1 is None or block_m1 <= 0) else int(block_m1)
        block_m2 = bm2 if (block_m2 is None or block_m2 <= 0) else int(block_m2)

    cache = get_teacher_cache_2opt(N, device)
    pairs = cache["pairs"]  # (M,2)
    maps = cache["maps"]    # (M,N)
    pi, pj, pin1, pjn1 = cache["pi"], cache["pj"], cache["pin1"], cache["pjn1"]
    M = pairs.size(0)

    # ---- STEP 1 deltas ----
    d1 = _delta_for_pairs_batch_general(tours0, dists, pairs)  # (B,M)
    d1 = torch.nan_to_num(d1, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))

    # ============================================================
    # k = 1
    # ============================================================
    if k_steps == 1:
        best1 = d1.amin(dim=1)  # (B,)
        mask1 = _tie_mask(d1, best1[:, None], tie_atol, tie_rtol)  # (B,M)

        # path sample
        m1_idx = _sample_one_from_mask(mask1, fallback_argmin_vals=d1)
        a1 = pairs.index_select(0, m1_idx)  # (B,2)

        # set pack (vectorized)
        topk1, v1 = _topk_from_mask_gumbel(mask1, int(max_ties))
        a1_set = pairs[topk1]  # (B,K,2)

        path = a1.view(B, 1, 2)
        return path, [a1_set], [v1], best1

    # ============================================================
    # k = 2 (exact streaming DP)
    #   Q0(m1) = d1(m1) + min_m2 d2(m1,m2)
    # ============================================================
    if k_steps == 2:
        # min2[b,m1] exactly, streamed over blocks
        min2 = _k2_min2_streaming(
            tours0, dists, maps, pi, pj, pin1, pjn1, block_m1=block_m1, block_m2=block_m2
        )  # (B,M)

        q0 = d1.to(torch.float32) + min2  # (B,M)
        best2 = q0.amin(dim=1)            # (B,)

        # step-1 optimal set + sample path m1
        mask1 = _tie_mask(q0, best2[:, None], tie_atol, tie_rtol)  # (B,M)
        m1_idx = _sample_one_from_mask(mask1, fallback_argmin_vals=q0)
        a1 = pairs.index_select(0, m1_idx)  # (B,2)

        topk1, v1 = _topk_from_mask_gumbel(mask1, int(max_ties))
        a1_set = pairs[topk1]  # (B,K,2)

        # step-2 optimal set depends on chosen m1
        d2_chosen = _k2_d2_for_chosen_m1(
            tours0, dists, maps, pi, pj, pin1, pjn1, m1_idx, block_m2=block_m2
        )  # (B,M)

        min2_chosen = d2_chosen.amin(dim=1)  # (B,)
        mask2 = _tie_mask(d2_chosen, min2_chosen[:, None], tie_atol, tie_rtol)  # (B,M)

        m2_idx = _sample_one_from_mask(mask2, fallback_argmin_vals=d2_chosen)
        a2 = pairs.index_select(0, m2_idx)  # (B,2)

        topk2, v2 = _topk_from_mask_gumbel(mask2, int(max_ties))
        a2_set = pairs[topk2]  # (B,K,2)

        path = torch.stack([a1, a2], dim=1)  # (B,2,2)
        return path, [a1_set, a2_set], [v1, v2], best2

    # ============================================================
    # k = 3
    # ============================================================
    # NOTE: k=3 exact scales ~O(M^3) and is not practical at large N.
    block_m1 = max(1, int(block_m1))
    block_m2 = max(1, int(block_m2))

    path_out = torch.empty((B, 3, 2), device=device, dtype=torch.long)
    best_out = torch.empty((B,), device=device, dtype=torch.float32)

    K = int(max_ties)
    a1_set_out = torch.zeros((B, K, 2), device=device, dtype=torch.long)
    v1_out = torch.zeros((B, K), device=device, dtype=torch.bool)
    a2_set_out = torch.zeros((B, K, 2), device=device, dtype=torch.long)
    v2_out = torch.zeros((B, K), device=device, dtype=torch.bool)
    a3_set_out = torch.zeros((B, K, 2), device=device, dtype=torch.long)
    v3_out = torch.zeros((B, K), device=device, dtype=torch.bool)

    def _apply_maps_to_single(tour0_1: torch.Tensor, maps_x: torch.Tensor) -> torch.Tensor:
        N_ = tour0_1.size(0)
        flat = maps_x.reshape(-1, N_)
        src = tour0_1.view(1, N_).expand(flat.size(0), N_)
        out = src.gather(1, flat)
        return out.view(*maps_x.shape[:-1], N_)

    for b in range(B):
        tour0_1 = tours0[b]  # (N,)
        dist_1 = dists[b]    # (N,N)

        d1_1 = _delta_for_pairs_batch_general(
            tour0_1.view(1, N), dist_1.view(1, N, N), pairs
        ).view(-1)
        d1_1 = torch.nan_to_num(d1_1, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))

        tours1_1 = _apply_maps_to_single(tour0_1, maps)  # (M,N)
        d2_1 = _delta_for_pairs_batch_general(
            tours1_1,
            dist_1.view(1, N, N).expand(M, N, N),
            pairs
        )  # (M,M)
        d2_1 = torch.nan_to_num(d2_1, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))

        V1 = torch.full((M,), float("inf"), device=device, dtype=torch.float32)

        for m1_lo in range(0, M, block_m1):
            m1_hi = min(M, m1_lo + block_m1)
            map1 = maps[m1_lo:m1_hi]        # (B1,N)
            d2_rows = d2_1[m1_lo:m1_hi, :]  # (B1,M)

            V1_blk = torch.full((m1_hi - m1_lo,), float("inf"), device=device, dtype=torch.float32)

            for m2_lo in range(0, M, block_m2):
                m2_hi = min(M, m2_lo + block_m2)
                map2 = maps[m2_lo:m2_hi]             # (B2,N)
                d2_blk = d2_rows[:, m2_lo:m2_hi]     # (B1,B2)

                map12 = _compose_maps(map1, map2)    # (B1,B2,N)
                tours2 = _apply_maps_to_single(tour0_1, map12).reshape(-1, N)

                d3 = _delta_for_pairs_batch_general(
                    tours2,
                    dist_1.view(1, N, N).expand(tours2.size(0), N, N),
                    pairs,
                )  # (C,M)
                d3 = torch.nan_to_num(d3, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
                min3 = d3.amin(dim=1).view(m1_hi - m1_lo, m2_hi - m2_lo)

                q1_blk = d2_blk + min3
                V1_blk = torch.minimum(V1_blk, q1_blk.amin(dim=1))

            V1[m1_lo:m1_hi] = V1_blk

        q0 = d1_1.to(torch.float32) + V1
        best3 = q0.min()
        best_out[b] = best3

        mask1 = _tie_mask(q0, best3, tie_atol, tie_rtol)  # (M,)
        # pack + sample
        idx1 = torch.nonzero(mask1, as_tuple=False).view(-1)
        if idx1.numel() == 0:
            idx1 = q0.argmin().view(1)
        # pack
        if idx1.numel() > K:
            idx1 = idx1.index_select(0, torch.randperm(idx1.numel(), device=device)[:K])
        a1_set_out[b, :idx1.numel()] = pairs.index_select(0, idx1)
        v1_out[b, :idx1.numel()] = True
        m1 = idx1[torch.randint(0, idx1.numel(), (1,), device=device)].item()

        map1 = maps[m1].view(1, N)
        Q1 = torch.full((M,), float("inf"), device=device, dtype=torch.float32)

        for m2_lo in range(0, M, block_m2):
            m2_hi = min(M, m2_lo + block_m2)
            map2 = maps[m2_lo:m2_hi]
            map12 = _compose_maps(map1, map2).squeeze(0)
            tours2 = _apply_maps_to_single(tour0_1, map12)

            d3 = _delta_for_pairs_batch_general(
                tours2,
                dist_1.view(1, N, N).expand(tours2.size(0), N, N),
                pairs,
            )
            d3 = torch.nan_to_num(d3, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
            min3 = d3.amin(dim=1)
            Q1[m2_lo:m2_hi] = d2_1[m1, m2_lo:m2_hi].to(torch.float32) + min3.to(torch.float32)

        V1_m1 = Q1.min()
        mask2 = _tie_mask(Q1, V1_m1, tie_atol, tie_rtol)
        idx2 = torch.nonzero(mask2, as_tuple=False).view(-1)
        if idx2.numel() == 0:
            idx2 = Q1.argmin().view(1)
        if idx2.numel() > K:
            idx2 = idx2.index_select(0, torch.randperm(idx2.numel(), device=device)[:K])
        a2_set_out[b, :idx2.numel()] = pairs.index_select(0, idx2)
        v2_out[b, :idx2.numel()] = True
        m2 = idx2[torch.randint(0, idx2.numel(), (1,), device=device)].item()

        map12 = _compose_maps(maps[m1].view(1, N), maps[m2].view(1, N)).view(N)
        tour2 = _apply_maps_to_single(tour0_1, map12).view(1, N)

        d3v = _delta_for_pairs_batch_general(
            tour2,
            dist_1.view(1, N, N),
            pairs,
        ).view(-1)
        d3v = torch.nan_to_num(d3v, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
        min3 = d3v.min()
        mask3 = _tie_mask(d3v, min3, tie_atol, tie_rtol)
        idx3 = torch.nonzero(mask3, as_tuple=False).view(-1)
        if idx3.numel() == 0:
            idx3 = d3v.argmin().view(1)
        if idx3.numel() > K:
            idx3 = idx3.index_select(0, torch.randperm(idx3.numel(), device=device)[:K])
        a3_set_out[b, :idx3.numel()] = pairs.index_select(0, idx3)
        v3_out[b, :idx3.numel()] = True
        m3 = idx3[torch.randint(0, idx3.numel(), (1,), device=device)].item()

        path_out[b, 0] = pairs[m1]
        path_out[b, 1] = pairs[m2]
        path_out[b, 2] = pairs[m3]

    opt_actions = [a1_set_out, a2_set_out, a3_set_out]
    opt_valid = [v1_out, v2_out, v3_out]
    return path_out, opt_actions, opt_valid, best_out


# ============================================================
# Warmup
# ============================================================
@torch.no_grad()
def warmup(
    problem,
    rollout_model,               # inner model (frozen) for warmup sampling
    coords_base: torch.Tensor,    # (B,N,2)
    solution_base: torch.Tensor,  # (B,N)
    last_k_actions_base: torch.Tensor,  # (B,K,2)
    opts,
    *,
    T_warm_mult: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    device = opts.device
    B, N, _ = coords_base.size()
    problem.size = N

    rollout_model.eval()
    tabu_state_base, tabu_cfg = _init_tabu_state(B, device, opts)

    mult = float(getattr(opts, "T_warm_mult", 0.0)) if T_warm_mult is None else float(T_warm_mult)
    T_warm_max = int(mult * N)
    if T_warm_max <= 0:
        return solution_base, last_k_actions_base, tabu_state_base

    mode = str(getattr(opts, "warmup_mode", "uniform")).lower()

    if mode == "uniform":
        t0 = torch.randint(low=0, high=T_warm_max + 1, size=(B,), device=device)
    elif mode == "fixed":
        tfix = int(torch.randint(low=0, high=T_warm_max + 1, size=(1,), device=device).item())
        t0 = torch.full((B,), tfix, device=device, dtype=torch.long)
    else:
        i = torch.arange(B, device=device)
        lo = (i * (T_warm_max + 1)) // B
        hi = ((i + 1) * (T_warm_max + 1)) // B - 1
        hi = torch.maximum(hi, lo)
        u = torch.rand((B,), device=device)
        t0 = lo + (u * (hi - lo + 1).to(torch.float32)).floor().long()
        t0 = t0[torch.randperm(B, device=device)]

    t0_max = int(t0.max().item())
    K_hist = int(last_k_actions_base.size(1))  # action_hist_len
    for step in range(t0_max):
        active = (t0 > step)
        if not active.any():
            break
        idx = active.nonzero(as_tuple=True)[0]

        coords_a = coords_base.index_select(0, idx)
        sol_a = solution_base.index_select(0, idx)
        lastk_a = last_k_actions_base.index_select(0, idx)
        tabu_a = {
            "action_hist": tabu_state_base["action_hist"].index_select(0, idx),
            "edge_hist": tabu_state_base["edge_hist"].index_select(0, idx),
        }

        with autocast_context(opts):
            logits_all_a = rollout_model(
                coords_a, sol_a,
                last_k_actions=lastk_a,
                tabu_edge_hist=tabu_a["edge_hist"],
            )
        if logits_all_a.dim() == 3:
            logits_all_a = logits_all_a.view(logits_all_a.size(0), -1)
        masked_logits_a = _apply_tabu_mask(logits_all_a, sol_a, tabu_a, tabu_cfg)
        exchange_a = _select_actions_from_logits(masked_logits_a, N, do_sample=True)

        prev_sol_a = sol_a
        sol_a = move_to(problem.step(sol_a, exchange_a), device)
        _update_tabu_state(tabu_a, tabu_cfg, prev_sol_a, exchange_a)

        if K_hist > 0:
            lastk_a = torch.roll(lastk_a, shifts=-1, dims=1)
            lastk_a[:, -1, :] = exchange_a

        solution_base[idx] = sol_a
        last_k_actions_base[idx] = lastk_a
        tabu_state_base["action_hist"][idx] = tabu_a["action_hist"]
        tabu_state_base["edge_hist"][idx] = tabu_a["edge_hist"]

    return solution_base, last_k_actions_base, tabu_state_base


# ============================================================
# IL loss
# ============================================================
def il_loss_exact_kstep_batch(
    problem,
    model,
    rollout_model,                 # inner frozen snapshot for warmup sampling
    batch: Dict[str, torch.Tensor],
    opts,
    *,
    cur_T_warm_mult: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Multi-label exact k-step teacher IL with set-valued optimal actions.

    Loss per step:  -log mean_{a in A*} pi(a|s)  (implemented via logsumexp - log|A*|)
    State advanced along one sampled optimal path (unbiased tie-breaking).
    """
    device = opts.device
    batch = move_to(batch, device)

    coords = batch["coords"]                          # (B,N,2)

    B, N, _ = coords.size()
    problem.size = N

    sum_logmass = torch.zeros((B,), device=device, dtype=torch.float32)

    solution0 = move_to(problem.get_initial_solutions(B)[0], device).long()
    K_hist = int(getattr(opts, "action_hist_len", 0))
    lastk0 = torch.full((B, K_hist, 2), -1, device=device, dtype=torch.long)

    solution0, lastk0, tabu_state0 = warmup(
        problem,
        rollout_model,
        coords,
        solution0, lastk0,
        opts,
        T_warm_mult=cur_T_warm_mult,
    )

    k_steps = int(getattr(opts, "il_horizon", 2))
    assert k_steps in (1, 2, 3), "Exact teacher supports il_horizon in {1,2,3}."

    with torch.no_grad():
        c = coords.to(torch.float32)
        dists = torch.cdist(c, c, p=2).contiguous()

    max_ties = int(getattr(opts, "il_max_ties", 128))
    tie_atol = float(getattr(opts, "il_tie_atol", 1e-9))
    tie_rtol = float(getattr(opts, "il_tie_rtol", 1e-9))

    # allow opts to override blocks; otherwise heuristic defaults kick in
    bm1 = getattr(opts, "teacher_block_m1", None)
    bm2 = getattr(opts, "teacher_block_m2", None)
    bm1 = None if bm1 is None else int(bm1)
    bm2 = None if bm2 is None else int(bm2)

    path_acts, opt_acts_list, opt_valid_list, best_total_delta = exact_optimal_kstep_teacher_sets_batch(
        tours0=solution0,
        dists=dists,
        k_steps=k_steps,
        max_ties=max_ties,
        tie_atol=tie_atol,
        tie_rtol=tie_rtol,
        block_m1=bm1,
        block_m2=bm2,
    )

    inner = get_inner_model(model)
    inner.train()

    sol = solution0
    lastk = lastk0
    tabu_state = {
        "action_hist": tabu_state0["action_hist"].clone(),
        "edge_hist": tabu_state0["edge_hist"].clone(),
    }
    _, tabu_cfg = _init_tabu_state(B, device, opts)

    greedy_in_set: List[torch.Tensor] = []

    for t in range(k_steps):
        cand = opt_acts_list[t].long()       # (B,K,2)
        valid = opt_valid_list[t]            # (B,K) bool
        a_path = path_acts[:, t, :].long()   # (B,2)

        with autocast_context(opts):
            logits_all = inner(
                coords, sol,
                last_k_actions=lastk,
                tabu_edge_hist=tabu_state["edge_hist"],
            )
        if logits_all.dim() == 3:
            logits_all = logits_all.view(B, -1)

        masked_logits = _apply_tabu_mask(logits_all, sol, tabu_state, tabu_cfg)
        a_pred = _select_actions_from_logits(masked_logits, N, do_sample=False)

        logp_all = _sanitize_and_normalize_logp_all(masked_logits.to(torch.float32), step=t, N=N)
        base_logp_all = _sanitize_and_normalize_logp_all(
            logits_all.to(torch.float32),
            step=t,
            N=N,
            tag="base_logits_all",
        )

        a_pred = a_pred.long()
        match = (cand == a_pred[:, None, :]).all(dim=-1) & valid
        greedy_in_set.append(match.any(dim=1))

        row = cand[..., 0].clamp(0, N - 1)
        col = cand[..., 1].clamp(0, N - 1)
        idx = (row * N + col).long()  # (B,K)

        logp_cand = logp_all.gather(1, idx)  # (B,K)
        logp_cand = logp_cand.masked_fill(~valid, -float("inf"))
        dead_teacher = (~torch.isfinite(logp_cand)).all(dim=1)
        if bool(dead_teacher.any()):
            base_logp_cand = base_logp_all.gather(1, idx).masked_fill(~valid, -float("inf"))
            logp_cand = logp_cand.clone()
            logp_cand[dead_teacher] = base_logp_cand[dead_teacher]

        logmass = torch.logsumexp(logp_cand, dim=1)  # (B,)
        denom = valid.sum(dim=1).clamp_min(1).to(logmass.dtype)
        logmass = logmass - denom.log()  # log-mean-exp
        sum_logmass = sum_logmass + logmass

        prev_sol = sol
        sol = move_to(problem.step(sol, a_path), device)
        _update_tabu_state(tabu_state, tabu_cfg, prev_sol, a_path)
        if K_hist > 0:
            lastk = torch.roll(lastk, shifts=-1, dims=1)
            lastk[:, -1, :] = a_path

    logmass_mean = sum_logmass / float(k_steps)
    loss = -logmass_mean.mean()

    with torch.no_grad():
        greedy_in_set_t = torch.stack(greedy_in_set, dim=0)  # (T,B)
        acc_step1_any = greedy_in_set_t[0].float().mean()
        acc_traj_any = greedy_in_set_t.all(dim=0).float().mean()

        init_cost = problem.get_costs(coords, solution0)
        final_cost = problem.get_costs(coords, sol)
        rel_impr = ((init_cost - final_cost).clamp_min(0.0) / init_cost.clamp_min(1e-6)).mean()

    stats = {
        "train/il_loss": float(loss.detach().item()),
        "train/il_logmass_mean": float(logmass_mean.mean().detach().item()),
        "train/il_acc_step1": float(acc_step1_any.item()),
        "train/il_acc_traj": float(acc_traj_any.item()),
        "train/il_rel_impr_mean": float(rel_impr.item()),
        "train/il_teacher_best_total_delta_mean": float(best_total_delta.mean().item()),
        "train/teacher_block_m1": float(getattr(opts, "teacher_block_m1", -1) if getattr(opts, "teacher_block_m1", None) is not None else -1),
        "train/teacher_block_m2": float(getattr(opts, "teacher_block_m2", -1) if getattr(opts, "teacher_block_m2", None) is not None else -1),
    }
    return loss, stats


# ============================================================
# Training epoch
# ============================================================
def train_il_epoch(problem, model, optimizer, epoch: int, val_datasets, opts):
    """
    Variable-size IL training epoch.

    - sizes = range(opts.il_size_range[0], opts.il_size_range[1] + 1)
    - each update consumes up to opts.sizes_per_update batches (possibly different N)
    - warmup via frozen rollout snapshot (copied at epoch start if off-policy)
    - exact k-step teacher
    """
    if opts.verbose:
        print("\n")
        print("|", format(f" IL training epoch {epoch} ", "*^60"), "|")
        print(f"Training with lr={optimizer.param_groups[0]['lr']:.3e} ", flush=True)

    device = opts.device
    model.train()
    scaler = getattr(opts, "_grad_scaler", None)

    on_policy = bool(getattr(opts, "on_policy_IL", False))

    if on_policy:
        rollout_policy = model
    else:
        rollout_policy = copy.deepcopy(get_inner_model(model)).to(device)
        rollout_policy.eval()
        for p in rollout_policy.parameters():
            p.requires_grad_(False)

    size_low, size_high = map(int, getattr(opts, "il_size_range", (opts.graph_size, opts.graph_size)))
    rng = max(1, size_high - size_low + 1)

    il_batch_size = int(getattr(opts, "il_batch_size", None) or getattr(opts, "batch_size"))
    total_num_batches = int(math.ceil(float(opts.epoch_size) / float(il_batch_size)))
    K = max(1, int(getattr(opts, "sizes_per_update", 1)))
    num_updates_target = int(math.ceil(total_num_batches / K))

    acc = {
        "train/il_loss": 0.0,
        "train/il_logmass_mean": 0.0,
        "train/il_acc_step1": 0.0,
        "train/il_acc_traj": 0.0,
        "train/il_rel_impr_mean": 0.0,
        "train/il_teacher_best_total_delta_mean": 0.0,
    }
    n_updates = 0
    n_batches = 0

    pbar = tqdm(
        total=total_num_batches,
        disable=getattr(opts, "no_progress_bar", False),
        desc=f"Training IL | epoch {epoch}",
        bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
    )

    for _ in range(num_updates_target):
        remaining = total_num_batches - n_batches
        K_eff = min(K, max(0, remaining))
        if K_eff <= 0:
            break

        sizes = []
        for i in range(K_eff):
            lo = size_low + (i * rng) // K_eff
            hi = size_low + ((i + 1) * rng) // K_eff - 1
            hi = max(lo, hi)
            sizes.append(random.randint(lo, hi))
        random.shuffle(sizes)

        optimizer.zero_grad(set_to_none=True)
        scale = 1.0 / float(K_eff)
        stats_sum = {k: 0.0 for k in acc}

        for N in sizes:
            problem.size = N
            batch = problem.sample_batch(il_batch_size, opts.graph_type, N, device=device)

            loss_b, stats_b = il_loss_exact_kstep_batch(
                problem, model, rollout_policy, batch, opts,
                cur_T_warm_mult=float(getattr(opts, "T_warm_mult", 0.0)),
            )

            loss_scaled = loss_b * scale
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()
            for k in stats_sum:
                stats_sum[k] += float(stats_b[k])

            n_batches += 1
            pbar.update(1)

        if scaler is not None and scaler.is_enabled():
            scaler.unscale_(optimizer)
        clip_grad_norms(optimizer.param_groups, getattr(opts, "max_grad_norm", 1.0))
        if scaler is not None and scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        for k in acc:
            acc[k] += stats_sum[k] / float(K_eff)

        n_updates += 1

        if (not on_policy) and getattr(opts, "update_old_model_every_batch", False):
            freq = int(getattr(opts, "update_old_model_freq", 1))
            if freq > 0 and (n_updates % freq == 0):
                with torch.no_grad():
                    rollout_policy.load_state_dict(get_inner_model(model).state_dict(), strict=True)

    pbar.close()
    total_epochs = int(getattr(opts, "total_epochs", opts.num_il_epochs + opts.num_rl_epochs))
    save_on_last_il = False
    if not getattr(opts, "epoch_schedule_is_custom", False):
        save_on_last_il = (epoch == opts.num_il_epochs - 1)
    if getattr(opts, "is_main_process", True) and opts.save_dir and (epoch == total_epochs - 1 or save_on_last_il or (opts.checkpoint_epochs and epoch % opts.checkpoint_epochs == 0)):
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
            os.path.join(opts.save_dir, f"il_epoch-{epoch}.pt"),
        )

    denom = max(1, n_updates)
    train_metrics = {k: float(v / denom) for k, v in acc.items()}

    model.eval()
    val_metrics = validate(problem, get_inner_model(model), val_datasets, opts)

    epoch_results = {
        "epoch": int(epoch),
        "lr": float(optimizer.param_groups[0]["lr"]),
        **train_metrics,
        **val_metrics,
    }
    return epoch_results

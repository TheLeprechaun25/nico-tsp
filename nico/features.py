"""Edge-token features of the incumbent solution (Section 5.1 and Appendix A.1).

Token k describes the solution edge e_k = (u_k -> v_k) = (sol[k], sol[k+1]),
with w_k = sol[k-1] and s_k = sol[k+2] its neighbours along the solution.
Feature blocks, concatenated in this order:

geo  (Euclidean coordinates)   endpoint coordinates (embedded separately), edge
     length, unit direction, cos/sin of the turning angles at u_k and v_k,
     neighbour-relative length, per-solution length z-score.            2D + 9
dist (cost matrix only)        edge cost, per-solution cost z-score,
     neighbour-relative cost, normalized rank of v_k in row u_k and of
     u_k in row v_k.                                                         5
cvrp                           demands q_u/Q, q_v/Q, depot flags of u and v,
     prefix load P/Q, suffix load (Lambda-P)/Q, residual capacity
     (Q-Lambda)/Q.                                                           7
hist                           frequency of position k among the endpoints of
     the recent moves.                                                       1
"""

from __future__ import annotations

from typing import Optional

import torch

from .problems.base import Instance, edge_costs
from .problems.cvrp import CVRP

EPS = 1e-6
GEO_DIM = 9
DIST_DIM = 5
CVRP_DIM = 7


def rank_matrix(cost_matrix: torch.Tensor) -> torch.Tensor:
    """rank[b, u, v] in [0, 1]: position of v among the other nodes sorted by d(u, .)."""
    B, V, _ = cost_matrix.shape
    d = cost_matrix.clone()
    d.diagonal(dim1=1, dim2=2).fill_(float("inf"))
    ranks = d.argsort(dim=-1).argsort(dim=-1).float()
    return (ranks / max(V - 2, 1)).clamp(max=1.0)


def ensure_rank_matrix(instance: Instance) -> Instance:
    """Adds ``rank_matrix`` once per instance (it does not depend on the solution)."""
    if "rank_matrix" not in instance:
        instance["rank_matrix"] = rank_matrix(instance["cost_matrix"])
    return instance


def _cos_sin(a: torch.Tensor, b: torch.Tensor):
    a = a / a.norm(dim=-1, keepdim=True).clamp_min(EPS)
    b = b / b.norm(dim=-1, keepdim=True).clamp_min(EPS)
    cos = (a * b).sum(dim=-1, keepdim=True)
    sin = a[..., 0:1] * b[..., 1:2] - a[..., 1:2] * b[..., 0:1]
    return cos, sin


def _masked_zscore(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """z-score of x (B, L, 1) over the tokens with weight 1."""
    count = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
    mu = (x * weight).sum(dim=1, keepdim=True) / count
    var = ((x - mu) ** 2 * weight).sum(dim=1, keepdim=True) / count
    return (x - mu) / var.sqrt().clamp_min(EPS)


def _neighbour_ratio(x: torch.Tensor, floor: Optional[torch.Tensor] = None) -> torch.Tensor:
    """x over the mean of its two tour neighbours. ``floor`` (B, 1, 1) bounds the denominator for costs with
    zero entries (road networks: distinct locations at the same road node), where the ratio would explode."""
    prev = torch.roll(x, shifts=1, dims=1).clamp_min(EPS)
    nxt = torch.roll(x, shifts=-1, dims=1).clamp_min(EPS)
    den = (0.5 * (prev + nxt)).clamp_min(EPS)
    return x / (den if floor is None else den.clamp_min(floor))


def geometric_features(coords: torch.Tensor, solution: torch.Tensor, weight: torch.Tensor):
    """Returns (endpoint coords u, endpoint coords v, (B, L, 9) scalars)."""
    B, L = solution.shape
    idx = lambda s: s.unsqueeze(-1).expand(B, L, 2)
    cw = coords.gather(1, idx(torch.roll(solution, 1, 1)))
    cu = coords.gather(1, idx(solution))
    cv = coords.gather(1, idx(torch.roll(solution, -1, 1)))
    cs = coords.gather(1, idx(torch.roll(solution, -2, 1)))
    disp = cv - cu
    length = disp.norm(dim=-1, keepdim=True)
    direction = disp / length.clamp_min(EPS)
    cos_u, sin_u = _cos_sin(cu - cw, disp)
    cos_v, sin_v = _cos_sin(disp, cs - cv)
    scalars = torch.cat(
        [length, direction, cos_u, sin_u, cos_v, sin_v,
         _neighbour_ratio(length), _masked_zscore(length, weight)],
        dim=-1,
    )
    return cu, cv, scalars


def distance_features(instance: Instance, solution: torch.Tensor, weight: torch.Tensor):
    ensure_rank_matrix(instance)
    u, v = solution, torch.roll(solution, -1, 1)
    cost = edge_costs(instance["cost_matrix"], u, v).unsqueeze(-1)
    rank_uv = edge_costs(instance["rank_matrix"], u, v).unsqueeze(-1)
    rank_vu = edge_costs(instance["rank_matrix"], v, u).unsqueeze(-1)
    return torch.cat(
        [cost, _masked_zscore(cost, weight), _neighbour_ratio(cost, 0.01 * cost.mean(1, keepdim=True)),
         rank_uv, rank_vu], dim=-1
    )


def cvrp_features(instance: Instance, solution: torch.Tensor):
    B, L = solution.shape
    Q = instance["capacity"].view(B, 1, 1)
    demand = instance["demand"]
    u, v = solution, torch.roll(solution, -1, 1)
    info = CVRP.route_info(instance, solution)
    P, Lam = info["prefix"].unsqueeze(-1), info["load"].unsqueeze(-1)
    return torch.cat(
        [
            demand.gather(1, u).unsqueeze(-1) / Q,
            demand.gather(1, v).unsqueeze(-1) / Q,
            (u == 0).float().unsqueeze(-1),
            (v == 0).float().unsqueeze(-1),
            P / Q,
            (Lam - P) / Q,
            (Q - Lam) / Q,
        ],
        dim=-1,
    )


def history_feature(move_hist: Optional[torch.Tensor], B: int, L: int, device) -> torch.Tensor:
    """(B, L, 1): how often position k appears among the endpoints of the recent moves."""
    out = torch.zeros(B, L, 1, device=device)
    if move_hist is None or move_hist.numel() == 0:
        return out
    valid = (move_hist >= 0).all(dim=-1)
    idx = (move_hist.clamp_min(0) % L).long().reshape(B, -1)   # first entry is op * L + i
    val = valid.unsqueeze(-1).expand_as(move_hist).reshape(B, -1).float()
    out.scatter_add_(1, idx.unsqueeze(-1), val.unsqueeze(-1))
    return out / val.sum(dim=1).clamp_min(1.0).view(B, 1, 1)


def knn_stats(instance: Instance, k: int):
    """Cached per instance: neighbour ranks (B, V, V) (0 = nearest other node) and the mean and standard
    deviation of each node's distances to its k nearest nodes (B, V)."""
    # stored as separate tensors so that batch slicing / repetition of the instance dict keeps them aligned
    keys = (f"_knn_rank_{k}", f"_knn_mu_{k}", f"_knn_sd_{k}")
    if keys[0] not in instance:
        d = instance["cost_matrix"].clone()
        d.diagonal(dim1=1, dim2=2).fill_(float("inf"))
        order = d.argsort(dim=-1)
        kk = min(k, d.size(-1) - 1)
        near = d.gather(2, order[..., :kk])
        instance[keys[0]] = order.argsort(dim=-1)
        instance[keys[1]] = near.mean(-1)
        instance[keys[2]] = near.std(-1, unbiased=False)
    return tuple(instance[key] for key in keys)


def knn_index(instance: Instance, k: int) -> torch.Tensor:
    """Cached (B, V, k) indices of each node's k nearest other nodes (nearest first)."""
    key = f"_knn_idx_{k}"
    if key not in instance:
        d = instance["cost_matrix"].clone()
        d.diagonal(dim1=1, dim2=2).fill_(float("inf"))
        instance[key] = d.topk(min(k, d.size(-1) - 1), dim=-1, largest=False).indices
    return instance[key]


def polar_neighbourhoods(instance: Instance, solution: torch.Tensor, k: int) -> torch.Tensor:
    """(B, L, 2, k, 3): for the tail u and head v of every edge, its k nearest nodes in the edge's frame:
    radius / largest radius of the k, and cos / sin of the angle to the edge direction (u -> v).
    Invariant to translation, rotation and scale of the instance."""
    coords = instance["coords"]
    B, L = solution.shape
    idx = knn_index(instance, k)                                           # (B, V, k)
    K = idx.size(-1)
    u, v = solution, torch.roll(solution, -1, 1)
    cu = coords.gather(1, u.unsqueeze(-1).expand(B, L, 2))
    cv = coords.gather(1, v.unsqueeze(-1).expand(B, L, 2))
    e = cv - cu
    e = e / e.norm(dim=-1, keepdim=True).clamp_min(EPS)                     # edge direction
    out = []
    for w, cw in ((u, cu), (v, cv)):
        nb = idx.gather(1, w.unsqueeze(-1).expand(B, L, K))                   # (B, L, K)
        pts = coords.gather(1, nb.reshape(B, L * K, 1).expand(B, L * K, 2)).view(B, L, K, 2)
        r = pts - cw.unsqueeze(2)
        rho = r.norm(dim=-1)
        along = (r * e.unsqueeze(2)).sum(-1)
        across = e[..., 0:1] * r[..., 1] - e[..., 1:2] * r[..., 0]
        cos, sin = along / rho.clamp_min(EPS), across / rho.clamp_min(EPS)
        rho_n = rho / rho.amax(dim=-1, keepdim=True).clamp_min(EPS)
        out.append(torch.stack([rho_n, cos, sin], dim=-1))
    return torch.stack(out, dim=2)


KNN_DIM = 4


def knn_edge_features(instance: Instance, solution: torch.Tensor, k: int) -> torch.Tensor:
    """(B, L, 4): v in kNN(u), u in kNN(v), and the two neighbour ranks clipped at 2k and scaled to [0, 1]."""
    ranks, _, _ = knn_stats(instance, k)
    u, v = solution, torch.roll(solution, -1, 1)
    r_uv = edge_costs(ranks, u, v).float()
    r_vu = edge_costs(ranks, v, u).float()
    cap = float(2 * k)
    return torch.stack([(r_uv < k).float(), (r_vu < k).float(),
                        r_uv.clamp(max=cap) / cap, r_vu.clamp(max=cap) / cap], dim=-1)


def local_length_stats(instance: Instance, solution: torch.Tensor, k: int):
    """(B, L) mean and std of the k-NN distances of each edge's tail u_k and head v_k:
    (mu_u, sd_u, mu_v, sd_v)."""
    _, mu, sd = knn_stats(instance, k)
    u, v = solution, torch.roll(solution, -1, 1)
    return mu.gather(1, u), sd.gather(1, u), mu.gather(1, v), sd.gather(1, v)


def active_edges(problem_name: str, solution: torch.Tensor) -> torch.Tensor:
    """(B, L) float: 0 for CVRP depot-to-depot padding edges, 1 otherwise."""
    if problem_name != "cvrp":
        return torch.ones_like(solution, dtype=torch.float32)
    return 1.0 - ((solution == 0) & (torch.roll(solution, -1, 1) == 0)).float()

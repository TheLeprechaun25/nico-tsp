"""NICO policy: solution-edge tokens -> edge encoder -> pairwise 2-opt decoder.

For the controlled representation study the edge tokens can be replaced by node tokens
(``model.tokens``); everything after the tokenizer is shared."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from .config import ModelConfig
from .features import (
    CVRP_DIM, DIST_DIM, GEO_DIM, KNN_DIM,
    active_edges, cvrp_features, distance_features, geometric_features, history_feature,
    knn_edge_features, knn_stats, local_length_stats, polar_neighbourhoods, _masked_zscore, _neighbour_ratio,
)
from .nn import CycleMixing, EncoderLayer
from .problems.base import Instance, edge_costs

FEATURE_SETS = ("geo", "dist", "full")
TOKEN_TYPES = ("edge", "node_pe", "node")
GAIN_MLP_DIM = 6


class EdgeTokenizer(nn.Module):
    """Builds one D-dimensional token per solution edge (Eq. 4)."""

    def __init__(self, cfg: ModelConfig, problem_name: str):
        super().__init__()
        if cfg.features not in FEATURE_SETS:
            raise ValueError(f"model.features must be one of {FEATURE_SETS}")
        D = cfg.embedding_dim
        self.problem_name = problem_name
        self.use_geo = cfg.features in ("geo", "full")
        self.use_dist = cfg.features in ("dist", "full")
        self.use_cvrp = problem_name == "cvrp"
        self.use_hist = cfg.history_feature
        self.use_geo_scalars = cfg.geo_scalars
        if cfg.length_norm not in ("none", "density"):
            raise ValueError(f"Unknown model.length_norm {cfg.length_norm!r}")
        self.density_norm = cfg.length_norm == "density"
        if cfg.zscore not in ("global", "local"):
            raise ValueError(f"Unknown model.zscore {cfg.zscore!r}")
        self.coord_embed = cfg.coord_embed
        self.edge_direction = cfg.edge_direction
        self.local_z = cfg.zscore == "local"
        self.len_local_scale = cfg.len_local_scale
        self.length_source = cfg.length_source
        if self.length_source not in ("coords", "cost"):
            raise ValueError("model.length_source must be 'coords' or 'cost'")
        self.knn_features = cfg.knn_features
        self.knn_k = cfg.knn_k
        self.abs_length = cfg.abs_length
        if cfg.length_scale not in ("ratio", "raw"):
            raise ValueError(f"Unknown model.length_scale {cfg.length_scale!r}")
        self.raw_scale = cfg.length_scale == "raw"
        if self.raw_scale and not (cfg.abs_length and cfg.len_local_scale):
            raise ValueError("length_scale=raw requires abs_length and len_local_scale")
        self.use_progress = cfg.progress_features
        self.polar_k = cfg.polar_k
        self.sym = cfg.endpoint_symmetric
        self.reflect = cfg.reflect_invariant or cfg.endpoint_symmetric
        if self.sym and cfg.polar_k > 0:
            raise ValueError("endpoint_symmetric is not implemented for polar features")
        if self.sym and self.coord_embed:
            raise ValueError("endpoint_symmetric requires coord_embed=false (ordered endpoint embeddings)")
        if self.reflect and self.edge_direction:
            raise ValueError("reflect_invariant requires edge_direction=false")

        in_dim = 0
        if self.use_geo:
            self.node_embed = nn.Linear(2, D)    # kept (unused) without coord_embed: same parameter layout
            n_scalars = (GEO_DIM - (0 if self.edge_direction else 2) + (1 if self.len_local_scale else 0)
                         - (0 if self.abs_length else 1) + (1 if self.reflect else 0) + (1 if self.raw_scale else 0))
            in_dim += 2 * D * self.coord_embed + n_scalars * self.use_geo_scalars
        in_dim += KNN_DIM * self.knn_features + 3 * self.use_progress
        if self.polar_k > 0:
            P = cfg.polar_dim     # DeepSets over each endpoint's neighbours: phi per point, mean, rho
            self.polar_phi = nn.Sequential(nn.Linear(3, P), nn.ReLU(), nn.Linear(P, P))
            self.polar_rho = nn.Sequential(nn.ReLU(), nn.Linear(P, P))
            in_dim += 2 * P
        in_dim += DIST_DIM * self.use_dist + CVRP_DIM * self.use_cvrp + 1 * self.use_hist
        self.proj = nn.Linear(in_dim, D)
        if self.use_cvrp:
            # Replaces depot-to-depot (empty-route) edges, which carry no geometry.
            self.null_token = nn.Parameter(torch.randn(D) * 0.02)

    def forward(self, instance: Instance, solution: torch.Tensor,
                move_hist: Optional[torch.Tensor], progress: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L = solution.shape
        active = active_edges(self.problem_name, solution)
        weight = active.unsqueeze(-1)
        parts = []
        if self.use_geo:
            cu, cv, geo = geometric_features(instance["coords"], solution, weight)
            if self.length_source == "cost":    # lengths from the objective, angles from the coordinates
                c = edge_costs(instance["cost_matrix"], solution, torch.roll(solution, -1, 1)).unsqueeze(-1)
                geo = torch.cat([c, geo[..., 1:7], _neighbour_ratio(c, 0.01 * c.mean(1, keepdim=True)),
                                 _masked_zscore(c, weight)], dim=-1)
            # named columns: length, direction (2), cos_u, sin_u, cos_v, sin_v, rel_len, z
            length, direc = geo[..., 0:1], geo[..., 1:3]
            cos_u, sin_u, cos_v, sin_v = (geo[..., i:i + 1] for i in (3, 4, 5, 6))
            rel_len, z = geo[..., 7:8], geo[..., 8:9]
            if self.density_norm:   # edge length in units of the typical spacing 1 / sqrt(V)
                length = length * math.sqrt(instance["coords"].size(1))
            if self.local_z or self.len_local_scale:
                mu_u, sd_u, mu_v, sd_v = local_length_stats(instance, solution, self.knn_k)
                d = geo[..., 0]
            if self.local_z:        # z-score of the edge length against the endpoints' k-NN distances
                z_u = (d - mu_u) / sd_u.clamp_min(1e-6)
                z = (0.5 * (z_u + (d - mu_v) / sd_v.clamp_min(1e-6)) if self.sym else z_u).unsqueeze(-1)
            angles = [cos_u, sin_u, cos_v, sin_v]
            if self.reflect:        # mirror-invariant: |sin| per endpoint and the product of the signed sines
                sin_prod = sin_u * sin_v
                angles = [cos_u, sin_u.abs(), cos_v, sin_v.abs()]
            if self.sym:            # reversal-invariant: unordered pairs -> sum and product
                angles = [cos_u + cos_v, cos_u * cos_v, sin_u.abs() + sin_v.abs(), sin_u.abs() * sin_v.abs()]
            if self.raw_scale:      # the same length quantities without normalization (feature-scale control)
                d0 = geo[..., 0:1]
                nw = weight.sum(1, keepdim=True).clamp_min(1.0)
                mean = (d0 * weight).sum(1, keepdim=True) / nw
                sd = (((d0 - mean) ** 2) * weight).sum(1, keepdim=True).div(nw).sqrt()
                rel_len = 0.5 * (torch.roll(d0, 1, 1) + torch.roll(d0, -1, 1))
                z = torch.cat([mean.expand_as(d0), sd.expand_as(d0)], dim=-1)
            cols = ([length] if self.abs_length else []) + ([direc] if self.edge_direction else []) + angles + [rel_len, z]
            if self.len_local_scale:  # length in units of the local k-NN spacing of both endpoints
                # floor at 1% of the mean k-NN spacing: zero road distances give nodes with zero spacing
                floor = 0.01 * knn_stats(instance, self.knn_k)[1].mean(1, keepdim=True)
                if self.raw_scale:
                    cols.append((0.5 * (mu_u + mu_v)).unsqueeze(-1))
                else:
                    cols.append((d / (0.5 * (mu_u + mu_v)).clamp_min(1e-6).clamp_min(floor)).unsqueeze(-1))
            if self.reflect:
                cols.append(sin_prod)
            geo = torch.cat(cols, dim=-1)
            if self.coord_embed:
                parts += [self.node_embed(cu), self.node_embed(cv)]
            if self.use_geo_scalars:
                parts.append(geo)
        if self.knn_features:
            kf = knn_edge_features(instance, solution, self.knn_k)        # (u->v, v->u, rank u->v, rank v->u)
            if self.sym:
                kf = torch.cat([kf[..., 0:1] + kf[..., 1:2], kf[..., 0:1] * kf[..., 1:2],
                                kf[..., 2:3] + kf[..., 3:4], kf[..., 2:3] * kf[..., 3:4]], dim=-1)
            parts.append(kf)
        if self.polar_k > 0:
            pn = polar_neighbourhoods(instance, solution, self.polar_k)        # (B, L, 2, K, 3)
            pooled = self.polar_rho(self.polar_phi(pn).mean(dim=3))            # (B, L, 2, P)
            parts.append(pooled.flatten(2))
        if self.use_dist:
            parts.append(distance_features(instance, solution, weight))
        if self.use_cvrp:
            parts.append(cvrp_features(instance, solution))
        if self.use_hist:
            parts.append(history_feature(move_hist, B, L, solution.device))
        if self.use_progress:
            prog = progress if progress is not None else torch.zeros(B, 3, device=solution.device)
            parts.append(prog.float().unsqueeze(1).expand(B, L, 3))
        tokens = self.proj(torch.cat(parts, dim=-1))
        if self.use_cvrp:
            tokens = tokens * weight + self.null_token * (1.0 - weight)
        return tokens


def positional_encoding(kind: str, L: int, D: int, device) -> torch.Tensor:
    """(L, D) encoding of tour positions 0..L-1.

    "cyclic": sin/cos of 2*pi*m*k/L for harmonics m = 1..D/2, periodic in L (cyclic PE as in DACT).
    "sin":    the absolute sinusoidal encoding of Vaswani et al. (as in Wu et al., 2021)."""
    k = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(1)
    if kind == "cyclic":
        m = torch.arange(1, D // 2 + 1, device=device, dtype=torch.float32)
        angle = 2.0 * math.pi * k * m / L
    elif kind == "sin":
        freq = torch.exp(-math.log(10000.0) * torch.arange(0, D, 2, device=device, dtype=torch.float32) / D)
        angle = k * freq
    else:
        raise ValueError(f"Unknown positional encoding {kind!r}")
    pe = torch.zeros(L, D, device=device)
    pe[:, 0::2] = torch.sin(angle)
    pe[:, 1::2] = torch.cos(angle)
    return pe


class NodeTokenizer(nn.Module):
    """One token per node, ordered by tour position: token k describes node sol[k].

    ``node_pe``: the solution enters only through a positional encoding of k (added after the
    projection). ``node``: no solution information at all (apart from the move history)."""

    def __init__(self, cfg: ModelConfig, problem_name: str):
        super().__init__()
        if problem_name != "tsp" or cfg.features != "geo":
            raise ValueError("node tokens are implemented for the Euclidean TSP (features=geo) only")
        D = cfg.embedding_dim
        self.dim = D
        self.use_pe = cfg.tokens == "node_pe"
        self.pe_kind = cfg.positional_encoding
        self.use_hist = cfg.history_feature
        self.node_embed = nn.Linear(2, D)
        self.proj = nn.Linear(D + 1 * self.use_hist, D)

    def forward(self, instance: Instance, solution: torch.Tensor,
                move_hist: Optional[torch.Tensor]) -> torch.Tensor:
        B, L = solution.shape
        cu = instance["coords"].gather(1, solution.unsqueeze(-1).expand(B, L, 2))
        parts = [self.node_embed(cu)]
        if self.use_hist:
            parts.append(history_feature(move_hist, B, L, solution.device))
        tokens = self.proj(torch.cat(parts, dim=-1))
        if self.use_pe:
            tokens = tokens + positional_encoding(self.pe_kind, L, self.dim, solution.device)
        return tokens


class NICOPolicy(nn.Module):
    def __init__(self, cfg: ModelConfig, problem_name: str):
        super().__init__()
        self.cfg = cfg
        D = cfg.embedding_dim
        if cfg.tokens not in TOKEN_TYPES:
            raise ValueError(f"model.tokens must be one of {TOKEN_TYPES}")
        self.tokenizer = (EdgeTokenizer(cfg, problem_name) if cfg.tokens == "edge"
                          else NodeTokenizer(cfg, problem_name))
        self.cycle_mixing = (
            CycleMixing(D, cfg.activation, cfg.cycle_alpha_init, cfg.cycle_mixing_symmetric)
            if cfg.cycle_mixing else nn.Identity()
        )
        self.layers = nn.ModuleList(
            EncoderLayer(D, cfg.n_heads, cfg.ffn_dim, cfg.normalization, cfg.activation, cfg.attention,
                         cfg.ref_len, cfg.dropout, cfg.dropout_place, cfg.prenorm, cfg.dropout_kind,
                         cfg.attn_dist_bias)
            for _ in range(cfg.n_layers)
        )
        self.loc_proj = nn.Linear(D, D, bias=False)   # W_loc
        self.glob_proj = nn.Linear(D, D, bias=False)  # W_glob
        self.W_q = nn.Parameter(torch.empty(D, D))
        self.W_k = nn.Parameter(torch.empty(D, D))
        bound = 1.0 / math.sqrt(D)
        nn.init.uniform_(self.W_q, -bound, bound)
        nn.init.uniform_(self.W_k, -bound, bound)
        # One extra bilinear head per additional operator (relocate, swap, 2-opt*).
        self.operators = tuple(cfg.operators)
        n_extra = len(self.operators) - 1
        self.W_q_ops = nn.Parameter(torch.empty(n_extra, D, D).uniform_(-bound, bound)) if n_extra else None
        self.W_k_ops = nn.Parameter(torch.empty(n_extra, D, D).uniform_(-bound, bound)) if n_extra else None
        self.delta_beta = (nn.Parameter(torch.full((len(self.operators),), float(cfg.delta_bias_init)))
                           if cfg.delta_bias else None)
        if cfg.decoder == "gain_mlp":
            if not cfg.delta_bias or len(self.operators) > 1:
                raise ValueError("the gain_mlp decoder needs delta_bias and a single operator")
            self.gain_mlp = nn.Sequential(nn.Linear(GAIN_MLP_DIM, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                                          nn.Linear(64, 1))
        if cfg.decoder not in ("pairwise", "two_stage", "gain_mlp"):
            raise ValueError(f"Unknown model.decoder {cfg.decoder!r}")
        if cfg.decoder == "two_stage":
            if len(self.operators) > 1:
                raise ValueError("the two-stage decoder is implemented for a single operator")
            # first stage: which edge to cut; with gains, biased by the best gain available from that edge
            self.first_head = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, 1))
            self.first_beta = nn.Parameter(torch.tensor(float(cfg.delta_bias_init))) if cfg.delta_bias else None

    def neighbours(self, instance: Instance, solution: torch.Tensor) -> torch.Tensor:
        """(B, L, 2w+1+K) token indices for local attention: the +-w tour neighbours (incl. itself) and
        the K nearest other tokens by position (edge midpoint, or node coordinate for node tokens)."""
        B, L = solution.shape
        cfg = self.cfg
        c = instance["coords"].gather(1, solution.unsqueeze(-1).expand(B, L, 2))
        pos = 0.5 * (c + torch.roll(c, -1, 1)) if cfg.tokens == "edge" else c
        ar = torch.arange(L, device=solution.device)
        offs = torch.arange(-cfg.local_window, cfg.local_window + 1, device=solution.device)
        window = (ar.view(L, 1) + offs.view(1, -1)) % L                               # (L, 2w+1)
        K = min(cfg.local_k, L - window.size(1))
        if K <= 0:
            return window.unsqueeze(0).expand(B, L, -1)
        d = torch.cdist(pos, pos)
        d.scatter_(2, window.unsqueeze(0).expand(B, L, -1), float("inf"))
        knn = d.topk(K, dim=-1, largest=False).indices
        return torch.cat([window.unsqueeze(0).expand(B, L, -1), knn], dim=-1)

    def token_logdist(self, instance: Instance, solution: torch.Tensor) -> torch.Tensor:
        """(B, L, L) log(1 + distance between token positions / mean local k-NN spacing of the instance);
        token positions are edge midpoints (node coordinates for node tokens). Scale-invariant."""
        B, L = solution.shape
        c = instance["coords"].gather(1, solution.unsqueeze(-1).expand(B, L, 2))
        pos = 0.5 * (c + torch.roll(c, -1, 1)) if self.cfg.tokens == "edge" else c
        _, mu, _ = knn_stats(instance, self.cfg.knn_k)
        spacing = mu.mean(dim=1).clamp_min(1e-6).view(B, 1, 1)
        return torch.log1p(torch.cdist(pos, pos) / spacing)

    def encode(self, instance: Instance, solution: torch.Tensor,
               move_hist: Optional[torch.Tensor] = None, progress: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Under bf16 autocast only the encoder layers run in bf16: the tokenizer (raw geometric
        # features) and the pairwise decoder (fine logit differences) stay in fp32.
        with torch.autocast(solution.device.type, enabled=False):
            tok = (self.tokenizer(instance, solution, move_hist, progress) if self.cfg.tokens == "edge"
                   else self.tokenizer(instance, solution, move_hist))
            h = self.cycle_mixing(tok)
            nbrs = self.neighbours(instance, solution) if self.cfg.attention == "local" else None
            logdist = self.token_logdist(instance, solution) if self.cfg.attn_dist_bias else None
        for layer in self.layers:
            h = layer(h, nbrs, logdist)
        with torch.autocast(solution.device.type, enabled=False):
            h = h.float()
            out = self.loc_proj(h)
            if self.cfg.global_context:
                out = out + self.glob_proj(h.mean(dim=1)).unsqueeze(1)
            return out

    def forward(self, instance: Instance, solution: torch.Tensor,
                move_hist: Optional[torch.Tensor], invalid: torch.Tensor,
                gain: Optional[torch.Tensor] = None, progress: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Logits (B, O*L*L) over moves (op, i, j); -inf where ``invalid`` ((B, L, L) or (B, O, L, L)).

        ``gain`` (B, O*L*L): normalised cost decrease of each move, used when ``delta_bias``."""
        if self.cfg.decoder == "gain_mlp":
            return self._gain_mlp(solution, move_hist, invalid, gain)
        h = self.encode(instance, solution, move_hist, progress)
        return self._decode(h, invalid, gain)

    @torch.autocast("cuda", enabled=False)
    def _gain_mlp(self, solution: torch.Tensor, move_hist: Optional[torch.Tensor], invalid: torch.Tensor,
                  gain: torch.Tensor) -> torch.Tensor:
        """Control without an encoder: z_ij = beta * omega_ij + C * tanh(f(x_ij)), where x_ij are statistics of
        the immediate gains only: omega_ij, omega_ij - max omega, log(1 + gain rank), log(1 + number of
        improving moves), the gain standardized over the admissible moves, and the history frequency of the
        two cut positions. f can adapt the selectivity to the state but sees nothing beyond the gains."""
        B, L = solution.shape
        inv = invalid.reshape(B, -1)
        valid = ~inv
        g = gain.view(B, -1).float()
        gm = g.masked_fill(inv, -1e9)
        gmax = gm.amax(1, keepdim=True)
        order = gm.argsort(1, descending=True)
        rank = torch.empty_like(order).scatter_(1, order, torch.arange(g.size(1), device=g.device).expand(B, -1))
        nv = valid.sum(1, keepdim=True).clamp_min(1).float()
        n_imp = ((g > 1e-9) & valid).sum(1, keepdim=True).float()
        mu = (g * valid).sum(1, keepdim=True) / nv
        sd = (((g - mu) ** 2) * valid).sum(1, keepdim=True).div(nv).sqrt().clamp_min(1e-6)
        hist = history_feature(move_hist, B, L, solution.device).view(B, L)
        hs = (hist.unsqueeze(2) + hist.unsqueeze(1)).view(B, -1)
        x = torch.stack([g, (g - gmax).clamp_min(-20.0), torch.log1p(rank.float()),
                         torch.log1p(n_imp).expand_as(g), (g - mu) / sd, hs], dim=-1)
        learned = self.cfg.logit_clip * torch.tanh(self.gain_mlp(x).squeeze(-1))
        return (learned + self.delta_beta[0] * g).masked_fill(inv, float("-inf"))

    def _two_stage(self, h: torch.Tensor, scores: torch.Tensor, invalid: torch.Tensor,
                   gain: Optional[torch.Tensor]) -> torch.Tensor:
        """log p(i, j) = log(p(i) p(j | i) + p(j) p(i | j)) on admissible (i < j) moves, -inf elsewhere.
        p(i) is a softmax over edges with at least one admissible partner, p(j | i) a softmax over the
        admissible partners of i; both are over at most L options."""
        B, L, _ = scores.shape
        neg = -1e4                     # finite inside, so masked entries never produce NaN gradients
        valid = ~invalid
        vsym = valid | valid.transpose(1, 2)
        pair = scores
        first = self.first_head(h).squeeze(-1)                                  # (B, L)
        if self.delta_beta is not None:
            g = gain.view(B, L, L)
            g = torch.where(valid, g, g.transpose(1, 2))                         # symmetric gains
            pair = pair + self.delta_beta[0] * g
            first = first + self.first_beta * g.masked_fill(~vsym, neg).amax(dim=-1).clamp_min(-10.0)
        row_ok = vsym.any(dim=-1)
        cond = torch.log_softmax(pair.masked_fill(~vsym, neg), dim=-1).masked_fill(~vsym, neg)
        lp_first = torch.log_softmax(first.masked_fill(~row_ok, neg), dim=-1)
        joint = lp_first.unsqueeze(-1) + cond
        out = torch.logaddexp(joint, joint.transpose(1, 2))
        return out.masked_fill(invalid, float("-inf")).reshape(B, L * L)

    @torch.autocast("cuda", enabled=False)
    def _decode(self, h: torch.Tensor, invalid: torch.Tensor, gain: Optional[torch.Tensor]) -> torch.Tensor:
        B, L, D = h.shape
        scores = (h @ self.W_q) @ (h @ self.W_k).transpose(1, 2) / math.sqrt(D)
        scores = self.cfg.logit_clip * torch.tanh(scores)
        scores = 0.5 * (scores + scores.transpose(1, 2))
        if self.cfg.decoder == "two_stage":
            return self._two_stage(h, scores, invalid.view(B, L, L), gain)
        if self.cfg.logit_size_scale:
            scores = scores * (math.log(max(L * L, 2)) / math.log(max(self.cfg.ref_len ** 2, 2)))
        if self.W_q_ops is None:
            scores = scores.unsqueeze(1)
            if self.delta_beta is not None:
                scores = scores + self.delta_beta.view(1, -1, 1, 1) * gain.view(B, 1, L, L)
            return scores.masked_fill(invalid.view(B, 1, L, L), float("-inf")).reshape(B, L * L)
        q = torch.einsum("bld,ode->bole", h, self.W_q_ops)
        k = torch.einsum("bld,ode->bole", h, self.W_k_ops)
        extra = self.cfg.logit_clip * torch.tanh(q @ k.transpose(-1, -2) / math.sqrt(D))   # (B, O-1, L, L)
        sym = torch.tensor([op != "relocate" for op in self.operators[1:]], device=h.device).view(1, -1, 1, 1)
        extra = torch.where(sym, 0.5 * (extra + extra.transpose(-1, -2)), extra)
        scores = torch.cat([scores.unsqueeze(1), extra], dim=1)
        if self.delta_beta is not None:
            scores = scores + self.delta_beta.view(1, -1, 1, 1) * gain.view(B, -1, L, L)
        return scores.masked_fill(invalid.view(B, -1, L, L), float("-inf")).reshape(B, -1)

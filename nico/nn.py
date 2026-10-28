"""Transformer building blocks used by the NICO encoder."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


def activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation {name!r}")


class RMSNorm(nn.Module):
    """Parameter-free RMS normalization."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.float()
        return (t * torch.rsqrt(t.pow(2).mean(dim=-1, keepdim=True) + self.eps)).to(x.dtype)


def normalization(name: str, dim: int) -> nn.Module:
    if name == "layer":
        return nn.LayerNorm(dim)
    if name == "rms":
        return RMSNorm()
    raise ValueError(f"Unknown normalization {name!r}")


class VPADropout(nn.Module):
    """Variance-Preserving Affine Dropout (Vasylenko et al., "Length Generalization Needs Proper
    Regularization", arXiv 2610.04518). Training: z = alpha * b * x + beta with b ~ Bernoulli(1 - p),
    alpha = sigma / sqrt((1 - p)(sigma^2 + p mu^2)), beta = mu (1 - alpha (1 - p)), where mu and sigma^2
    are the mean and variance of each token's features, so the per-token mean and variance are preserved.
    Inference: identity."""

    def __init__(self, p: float, eps: float = 1e-6):
        super().__init__()
        self.p, self.eps = float(p), eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0:
            return x
        q = 1.0 - self.p
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        alpha = torch.sqrt(var + self.eps) * torch.rsqrt(q * (var + self.p * mu * mu) + self.eps)
        beta = mu * (1.0 - alpha * q)
        b = torch.bernoulli(torch.full_like(x, q))
        return alpha * b * x + beta


def make_dropout(p: float, kind: str) -> nn.Module:
    if p <= 0:
        return nn.Identity()
    if kind == "standard":
        return nn.Dropout(p)
    if kind == "vpad":
        return VPADropout(p)
    raise ValueError(f"Unknown dropout kind {kind!r}")


class MultiHeadSelfAttention(nn.Module):
    """Self-attention without positional encodings or masking.

    The fused projection is laid out per head as [q_h, k_h, v_h], h = 1..H.

    ``scope``: "dense" (all tokens), "logn" (dense, logits scaled by a learned per-head factor times
    log(N) / log(ref_len), so attention does not flatten as N grows), or "local" (each token attends
    only to the neighbour indices passed to ``forward``, (B, N, K); O(N K)).
    ``dropout`` with ``dropout_place="pre_proj"`` acts on the context right before the output projection.
    """

    def __init__(self, dim: int, n_heads: int, scope: str = "dense", ref_len: int = 50,
                 dropout: float = 0.0, dropout_place: str = "standard", dist_bias: bool = False):
        super().__init__()
        if dim % n_heads:
            raise ValueError("embedding_dim must be divisible by n_heads")
        if scope not in ("dense", "logn", "local"):
            raise ValueError(f"Unknown attention scope {scope!r}")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scope = scope
        self.ref_len = ref_len
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        for p in self.parameters():
            bound = 1.0 / math.sqrt(p.size(-1))
            nn.init.uniform_(p, -bound, bound)
        if scope == "logn":
            self.log_scale = nn.Parameter(torch.ones(n_heads))
        self.pre_proj_drop = nn.Dropout(dropout) if dropout_place == "pre_proj" else nn.Identity()
        if dist_bias:
            if scope == "local":
                raise ValueError("attn_dist_bias is implemented for dense attention")
            # softplus(0.5413) = 1: the bias starts at -log(1 + d / s) for every head
            self.dist_gamma = nn.Parameter(torch.full((n_heads,), 0.5413))
        self.dist_bias = dist_bias

    def forward(self, x: torch.Tensor, neighbours: Optional[torch.Tensor] = None,
                logdist: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, self.n_heads, 3 * self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in qkv.split(self.head_dim, dim=-1))      # (B, H, N, d)
        if self.scope == "logn":
            factor = math.log(max(N, 2)) / math.log(max(self.ref_len, 2))
            q = q * (self.log_scale.view(1, -1, 1, 1) * factor)
        if self.scope == "local":
            if neighbours is None:
                raise ValueError("local attention needs neighbour indices")
            K = neighbours.size(-1)
            idx = neighbours.view(B, 1, N * K, 1).expand(B, self.n_heads, N * K, self.head_dim)
            kn = k.gather(2, idx).view(B, self.n_heads, N, K, self.head_dim)
            vn = v.gather(2, idx).view(B, self.n_heads, N, K, self.head_dim)
            att = torch.einsum("bhnd,bhnkd->bhnk", q, kn) / math.sqrt(self.head_dim)
            ctx = torch.einsum("bhnk,bhnkd->bhnd", att.softmax(dim=-1), vn)
        else:
            mask = None
            if self.dist_bias:      # soft locality: -gamma_h * log(1 + distance / local spacing)
                mask = -F.softplus(self.dist_gamma).view(1, -1, 1, 1) * logdist.unsqueeze(1).to(q.dtype)
            ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(self.pre_proj_drop(ctx.transpose(1, 2).reshape(B, N, D)))


class EncoderLayer(nn.Module):
    """Transformer block. Post-norm (default): x <- Norm(x + MHA(x)); x <- Norm(x + FFN(x)).
    Pre-norm: x <- x + MHA(Norm(x)); x <- x + FFN(Norm(x)).

    Dropout placement: "standard" drops each sublayer output before the residual addition (and hence
    before the post-norm); "pre_proj" drops the input of the last linear projection of each sublayer
    (attention context before W_O, FFN hidden activation before W_2); "pre_linear" drops the (normalized)
    sublayer input right before the QKV projection and the FFN up-projection, the residual path keeping
    the undropped input (Vasylenko et al., 2026). ``dropout_kind``: "standard" (inverted) or "vpad"."""

    def __init__(self, dim: int, n_heads: int, ffn_dim: int, norm: str, act: str, scope: str = "dense",
                 ref_len: int = 50, dropout: float = 0.0, dropout_place: str = "standard",
                 prenorm: bool = False, dropout_kind: str = "standard", dist_bias: bool = False):
        super().__init__()
        if dropout_place not in ("standard", "pre_proj", "pre_linear"):
            raise ValueError(f"Unknown dropout placement {dropout_place!r}")
        self.attn = MultiHeadSelfAttention(dim, n_heads, scope, ref_len, dropout, dropout_place, dist_bias)
        self.in_drop = make_dropout(dropout, dropout_kind) if dropout_place == "pre_linear" else nn.Identity()
        self.norm1 = normalization(norm, dim)
        # layout kept as Sequential(W_1, act, W_2) so existing checkpoints load unchanged
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), activation(act), nn.Linear(ffn_dim, dim))
        self.ffn_drop = nn.Dropout(dropout) if dropout_place == "pre_proj" else nn.Identity()
        self.norm2 = normalization(norm, dim)
        self.drop = nn.Dropout(dropout) if dropout_place == "standard" else nn.Identity()
        self.prenorm = prenorm

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn[2](self.ffn_drop(self.ffn[1](self.ffn[0](x))))

    def forward(self, x: torch.Tensor, neighbours: Optional[torch.Tensor] = None,
                logdist: Optional[torch.Tensor] = None) -> torch.Tensor:
        d = self.in_drop
        if self.prenorm:
            x = x + self.drop(self.attn(d(self.norm1(x)), neighbours, logdist))
            return x + self.drop(self._ffn(d(self.norm2(x))))
        x = self.norm1(x + self.drop(self.attn(d(x), neighbours, logdist)))
        return self.norm2(x + self.drop(self._ffn(d(x))))


class CycleMixing(nn.Module):
    """h_k <- h_k + alpha * phi([h_{k-1}, h_k, h_{k+1}]), indices mod L (Eq. 5).

    The output layer of phi is zero-initialised so the block starts as the identity.
    """

    def __init__(self, dim: int, act: str, alpha_init: float, symmetric: bool = False):
        super().__init__()
        # symmetric: phi([h_k, h_{k-1} + h_{k+1}, h_{k-1} * h_{k+1}]), unchanged when the cycle is reversed
        self.symmetric = symmetric
        self.mlp = nn.Sequential(nn.Linear(3 * dim, dim), activation(act), nn.Linear(dim, dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        prev = torch.roll(h, shifts=1, dims=1)
        nxt = torch.roll(h, shifts=-1, dims=1)
        if self.symmetric:
            return h + self.alpha * self.mlp(torch.cat([h, prev + nxt, prev * nxt], dim=-1))
        return h + self.alpha * self.mlp(torch.cat([prev, h, nxt], dim=-1))

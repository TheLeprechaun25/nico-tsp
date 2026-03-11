import math
import torch
import torch.nn.functional as F
from torch import nn


# ============================================================
# Utility modules: Activation and Normalization
# ============================================================
class Activation(nn.Module):
    def __init__(self, name: str = "relu"):
        super().__init__()
        self.kind = name

    def forward(self, x):
        if self.kind == "relu":
            return F.relu(x)
        if self.kind == "gelu":
            return F.gelu(x)
        if self.kind == "relu_squared":
            x = F.relu(x)
            return x * x
        raise NotImplementedError(f"Unknown activation: {self.kind}")


class Normalization(nn.Module):
    def __init__(self, embed_dim, normalization="batch"):
        super().__init__()
        self.normalization = normalization
        self.embed_dim = embed_dim

        if normalization == "layer":
            self.norm = nn.LayerNorm(embed_dim)
        elif normalization == "batch":
            self.norm = nn.BatchNorm1d(embed_dim, affine=True, track_running_stats=False)
        elif normalization == "rms":
            self.norm = None
        elif normalization == "instance":
            self.norm = nn.InstanceNorm1d(embed_dim, affine=True, track_running_stats=False)
        elif normalization == "none":
            self.norm = nn.Identity()
        else:
            raise NotImplementedError(f"Unknown normalization: {normalization}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalization == "rms":
            t = x.float()
            rms = torch.sqrt(torch.mean(t * t, dim=-1, keepdim=True) + 1e-6)
            return (t / rms).to(x.dtype)

        if self.normalization in ["instance", "batch"]:
            x = x.permute(0, 2, 1)
            x = self.norm(x)
            x = x.permute(0, 2, 1)
            return x

        return self.norm(x)


# ============================================================
# Multi-Head Attention
# ============================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, opts, n_heads, embed_dim):
        super().__init__()
        val_dim = embed_dim // n_heads

        self.opts = opts
        self.n_heads = int(n_heads)
        self.embed_dim = int(embed_dim)
        self.val_dim = int(val_dim)

        self.norm_factor = 1 / math.sqrt(self.val_dim)

        self.W_qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)
        self.W_out_lin = nn.Linear(self.n_heads * self.val_dim, self.embed_dim, bias=False)

        self.init_parameters()

    def init_parameters(self):
        with torch.no_grad():
            for p in self.parameters():
                stdv = 1.0 / math.sqrt(p.size(-1))
                p.uniform_(-stdv, stdv)

            if getattr(self.opts, "zero_init_residuals", False):
                self.W_out_lin.weight.zero_()

    def forward(self, x):
        B, N, C = x.size()
        assert C == self.embed_dim

        # ---- Fused QKV projection: (B,N,H*3Dv) -> (B,H,N,3Dv) ----
        qkv = self.W_qkv(x)  # (B,N,H*3Dv)
        qkv = qkv.view(B, N, self.n_heads, 3 * self.val_dim)
        q, k, v = torch.split(qkv, [self.val_dim, self.val_dim, self.val_dim], dim=-1)
        q = q.permute(0, 2, 1, 3).contiguous()  # (B,H,N,Dv)
        k = k.permute(0, 2, 1, 3).contiguous()  # (B,H,N,Dv)
        v = v.permute(0, 2, 1, 3).contiguous()  # (B,H,N,Dv)

        # ---- ScaledDotProductAttn ----
        if hasattr(F, "scaled_dot_product_attention"):
            ctx = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )  # (B,H,N,Dv)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.norm_factor
            attn = torch.softmax(scores, dim=-1)
            ctx = torch.matmul(attn, v)  # (B,H,N,Dv)

        out = self.W_out_lin(ctx.transpose(1, 2).contiguous().view(B, N, self.n_heads * self.val_dim))
        return out


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, opts, n_heads: int, embed_dim: int, feed_forward_hidden: int, normalization: str = "batch"):
        super().__init__()
        self.opts = opts
        self.embed_dim = embed_dim

        self.mha = MultiHeadAttention(opts, n_heads=n_heads, embed_dim=embed_dim)
        self.drop1 = nn.Dropout(opts.dropout) if opts.dropout > 0 else nn.Identity()
        self.norm1 = Normalization(embed_dim, normalization)
        self.act = Activation(opts.activation)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_hidden),
            self.act,
            nn.Linear(feed_forward_hidden, embed_dim),
        )
        self.drop2 = nn.Dropout(opts.dropout) if opts.dropout > 0 else nn.Identity()
        self.norm2 = Normalization(embed_dim, normalization)

        if getattr(opts, "zero_init_residuals", False):
            self.ffn[-1].weight.data.zero_()
            if self.ffn[-1].bias is not None:
                self.ffn[-1].bias.data.zero_()

    def forward(self, x):
        x = x + self.drop1(self.mha(x=x))
        x = self.norm1(x)
        x = x + self.drop2(self.ffn(x))
        x = self.norm2(x)
        return x


# ============================================================
# Edge embeddings
# ============================================================
class EdgeCycleInject(nn.Module):
    def __init__(self, D: int, hidden: int, activation: str = "relu", alpha_init: float = 0.1, zero_init_out: bool = True):
        super().__init__()
        self.D = int(D)
        self.mlp = nn.Sequential(
            nn.Linear(3 * self.D, hidden),
            Activation(activation),
            nn.Linear(hidden, self.D),
        )
        if zero_init_out:
            nn.init.zeros_(self.mlp[-1].weight)
            if self.mlp[-1].bias is not None:
                nn.init.zeros_(self.mlp[-1].bias)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, h_edge: torch.Tensor) -> torch.Tensor:
        prev = torch.roll(h_edge, shifts=1, dims=1)
        nxt = torch.roll(h_edge, shifts=-1, dims=1)
        delta = self.mlp(torch.cat([prev, h_edge, nxt], dim=-1))
        return self.alpha * delta


# ============================================================
# Edge-based decoder (scores (i,j) directly from edge tokens)
# ============================================================
class EdgePairDecoder(nn.Module):
    def __init__(self, opts, problem, embed_dim):
        super().__init__()
        self.problem = problem
        self.opts = opts
        self.embed_dim = int(embed_dim)

        self.norm_factor = 1.0 / math.sqrt(self.embed_dim)

        self.W_query = nn.Parameter(torch.Tensor(1, self.embed_dim, self.embed_dim))  # 1 decoder head
        self.W_key = nn.Parameter(torch.Tensor(1, self.embed_dim, self.embed_dim))  # 1 decoder head

        self.init_parameters()

    def init_parameters(self):
        for p in self.parameters():
            if p is None:
                continue
            stdv = 1.0 / math.sqrt(p.size(-1))
            p.data.uniform_(-stdv, stdv)

    def forward(self, h_hat, rec):
        B, N, D = h_hat.size()
        assert D == self.embed_dim

        hflat = h_hat.contiguous().view(-1, D)
        Q = torch.matmul(hflat, self.W_query).view(1, B, N, -1)
        K = torch.matmul(hflat, self.W_key).view(1, B, N, -1)

        compat = self.norm_factor * torch.matmul(Q, K.transpose(2, 3)).mean(dim=0)  # (B,N,N)
        C = float(getattr(self.opts, "logit_tanh_clip", 10.0))
        compat = C * torch.tanh(compat)

        # Make edge logits symmetric in the decoder
        compat = 0.5 * (compat + compat.transpose(1, 2))

        mask = self.problem.get_2opt_edge_pair_mask(rec)  # (B,N,N)
        logits = compat.masked_fill(mask, float("-inf")).view(B, -1)

        return logits


# ============================================================
# Edge-based body
# ============================================================
class EdgeAttentionModelBody(nn.Module):
    def __init__(self, opts, problem, embedding_dim, hidden_dim, n_heads, n_encode_layers, device):
        super().__init__()
        self.opts = opts
        self.problem = problem
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_encode_layers)
        self.device = device

        # Node tokens: Coords
        node_dim = 2
        self.node_embedder = nn.Linear(node_dim, self.embedding_dim)

        # Edge tokens: History features
        hist_dim = 0
        self.use_action_hist_feats = not bool(getattr(opts, "not_use_action_hist_feats", False))
        if self.use_action_hist_feats:
            hist_dim += 1

        raw_use_added = getattr(opts, "use_added_edge_hist_feats", False)
        self.use_added_edge_hist_feats = bool(raw_use_added)
        if self.use_added_edge_hist_feats:
            hist_dim += 1

        # Edge tokens: Geom features
        geom_dim = 9

        # Edge tokens:
        self.edge_in_dim = 2 * self.embedding_dim + geom_dim + hist_dim
        self.edge_token_proj = nn.Linear(self.edge_in_dim, self.embedding_dim)

        self.use_edge_inject = not bool(getattr(opts, "not_use_tour_inject", False))
        if self.use_edge_inject:
            alpha0 = float(getattr(opts, "tour_inject_alpha_init", 0.1))
            self.edge_inject = EdgeCycleInject(
                D=self.embedding_dim,
                hidden=self.embedding_dim,
                activation=getattr(opts, "activation", "relu"),
                alpha_init=alpha0,
                zero_init_out=True,
            )
        else:
            self.edge_inject = None

        norm = getattr(opts, "normalization", "batch")
        self.encoder = nn.ModuleList([
            MultiHeadAttentionLayer(opts, self.n_heads, self.embedding_dim, self.hidden_dim, normalization=norm)
            for _ in range(self.n_layers)
        ])

        self.project_graph = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.project_edge = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

    @staticmethod
    def _action_hist_edge_feats(last_k_actions: torch.Tensor, N: int, device) -> torch.Tensor:
        B = last_k_actions.size(0)
        i = last_k_actions[..., 0]
        j = last_k_actions[..., 1]
        valid = (i >= 0) & (j >= 0)
        i = i.clamp(0, N - 1).long()
        j = j.clamp(0, N - 1).long()
        idx = torch.cat([i, j], dim=1)
        v = torch.cat([valid, valid], dim=1).float()

        out = torch.zeros(B, N, 1, device=device)
        out.scatter_add_(1, idx.reshape(B, -1).unsqueeze(-1), v.reshape(B, -1).unsqueeze(-1))
        denom = v.sum(dim=1, keepdim=True).clamp_min(1.0)
        return out / denom.view(B, 1, 1)

    @staticmethod
    def _cos_sin(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6):
        na = torch.norm(a, dim=-1, keepdim=True).clamp_min(eps)
        nb = torch.norm(b, dim=-1, keepdim=True).clamp_min(eps)
        a_ = a / na
        b_ = b / nb
        cos = (a_ * b_).sum(dim=-1, keepdim=True)
        sin = (a_[..., 0:1] * b_[..., 1:2]) - (a_[..., 1:2] * b_[..., 0:1])  # 2D cross
        return cos, sin

    @staticmethod
    def _added_edge_hist_edge_feats(solutions: torch.Tensor, tabu_edge_hist: torch.Tensor) -> torch.Tensor:
        B, N = solutions.shape
        out = torch.zeros((B, N, 1), device=solutions.device, dtype=torch.float32)
        if tabu_edge_hist is None or tabu_edge_hist.numel() == 0:
            return out

        e0 = tabu_edge_hist[..., 0]
        e1 = tabu_edge_hist[..., 1]
        valid = (e0 >= 0) & (e1 >= 0)
        if not bool(valid.any()):
            return out

        eh_lo = torch.minimum(e0, e1)
        eh_hi = torch.maximum(e0, e1)

        u = solutions
        v = torch.roll(solutions, shifts=-1, dims=1)
        cur_lo = torch.minimum(u, v)
        cur_hi = torch.maximum(u, v)

        match = (
            (cur_lo.unsqueeze(-1) == eh_lo.unsqueeze(1))
            & (cur_hi.unsqueeze(-1) == eh_hi.unsqueeze(1))
            & valid.unsqueeze(1)
        )
        counts = match.sum(dim=-1, keepdim=True).to(torch.float32)
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1).to(torch.float32).unsqueeze(1)
        return counts / denom

    def forward(self, x, solutions, last_k_actions=None, tabu_edge_hist=None):
        B, N, _ = x.size()
        device = x.device
        eps = 1e-6

        node_h = self.node_embedder(x)  # (B,N,D) in node-id order

        tour = solutions.long()
        tour_next = torch.roll(tour, shifts=-1, dims=1)
        u = tour
        v = tour_next

        hu = node_h.gather(1, u.unsqueeze(-1).expand(B, N, self.embedding_dim))
        hv = node_h.gather(1, v.unsqueeze(-1).expand(B, N, self.embedding_dim))
        parts = [hu, hv]

        # geometric features
        coords = x[..., :2]
        cu = coords.gather(1, u.unsqueeze(-1).expand(B, N, 2))
        cv = coords.gather(1, v.unsqueeze(-1).expand(B, N, 2))
        dvec = cv - cu
        dist = torch.norm(dvec, dim=-1, keepdim=True)
        dunit = dvec / dist.clamp_min(1e-6)
        parts.extend([dist, dunit])

        # Extra geometric features
        tour_prev = torch.roll(tour, shifts=1, dims=1)  # previous node in tour
        tour_next2 = torch.roll(tour, shifts=-2, dims=1)  # node after v

        cw = coords.gather(1, tour_prev.unsqueeze(-1).expand(B, N, 2))
        cnext = coords.gather(1, tour_next2.unsqueeze(-1).expand(B, N, 2))

        # vectors
        vin_u = cu - cw  # incoming at u
        vout_u = cv - cu  # outgoing at u (= edge vector)
        vin_v = cv - cu  # incoming at v
        vout_v = cnext - cv  # outgoing at v
        cos_u, sin_u = self._cos_sin(vin_u, vout_u, eps=eps)
        cos_v, sin_v = self._cos_sin(vin_v, vout_v, eps=eps)

        parts.extend([cos_u, sin_u, cos_v, sin_v])

        dist_prev = torch.norm(cu - cw, dim=-1, keepdim=True).clamp_min(eps)
        dist_next = torch.norm(cnext - cv, dim=-1, keepdim=True).clamp_min(eps)
        rel_len = dist / (0.5 * (dist_prev + dist_next)).clamp_min(eps)  # (B,N,1)

        mu = dist.mean(dim=1, keepdim=True)
        sd = dist.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        zlen = (dist - mu) / sd

        parts.extend([rel_len, zlen])

        # Action history (unordered)
        if self.use_action_hist_feats and last_k_actions is not None:
            parts.append(self._action_hist_edge_feats(last_k_actions, N, device))
        if self.use_added_edge_hist_feats:
            parts.append(self._added_edge_hist_edge_feats(tour, tabu_edge_hist).to(node_h.dtype))

        # Concat features and projection
        edge_in = torch.cat(parts, dim=-1)
        edge_h = self.edge_token_proj(edge_in)

        # Edge tour injection
        if self.edge_inject is not None:
            edge_h = edge_h + self.edge_inject(edge_h)

        # MHA layers
        for layer in self.encoder:
            edge_h = layer(x=edge_h)

        # Linear projection
        edge_feature = self.project_edge(edge_h)

        # Graph-level context: mean pool over edge tokens
        pooled = edge_h.mean(dim=1)
        graph_feature = self.project_graph(pooled)

        # Add graph context
        h_hat = edge_feature + graph_feature[:, None, :].expand_as(edge_feature)
        return h_hat


# ============================================================
# Top-level model
# ============================================================
class EdgeAttentionModel(nn.Module):
    """
    Edge-attention NI model.
    """
    def __init__(self, opts, problem, embedding_dim, hidden_dim, n_heads, n_layers, normalization, device):
        super().__init__()
        self.opts = opts
        self.problem = problem
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.normalization = normalization
        self.device = device

        self.body = EdgeAttentionModelBody(
            opts=opts,
            problem=problem,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            n_heads=self.n_heads,
            n_encode_layers=self.n_layers,
            device=device,
        )

        self.decoder = EdgePairDecoder(
            opts=opts,
            problem=problem,
            embed_dim=self.embedding_dim,
        )

    def forward(self, x, solutions, last_k_actions=None, tabu_edge_hist=None):
        h_hat = self.body(
            x,
            solutions=solutions,
            last_k_actions=last_k_actions,
            tabu_edge_hist=tabu_edge_hist,
        )
        return self.decoder(h_hat, rec=solutions)

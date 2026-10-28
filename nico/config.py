"""Typed configuration for models, training and evaluation.

A run is fully described by one :class:`Config`. It is loaded from YAML, can be
overridden from the command line (``key.sub=value``), and is stored verbatim in
every checkpoint so that a checkpoint can always be re-instantiated exactly.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class ModelConfig:
    embedding_dim: int = 128
    ffn_dim: int = 512
    n_layers: int = 3
    n_heads: int = 8
    normalization: str = "layer"          # "layer" | "rms"
    activation: str = "relu"              # "relu" | "gelu"
    logit_clip: float = 10.0              # C in C * tanh(.)
    cycle_mixing: bool = True
    cycle_alpha_init: float = 0.1
    # Edge-token feature set:
    #   "geo"  - coordinates and Euclidean geometry
    #   "dist" - coordinate-free features read from the cost matrix
    #   "full" - union of "geo" and "dist"
    features: str = "geo"
    # Token type (controlled representation study; NICO = "edge"):
    #   "edge"    - one token per solution edge (sol[k] -> sol[k+1]), Eq. 4
    #   "node_pe" - one token per node, coordinates + positional encoding of its tour position
    #               (the representation of earlier improvement policies); no successor information
    #   "node"    - node coordinates only, no positional encoding (no solution information)
    # Node tokens are indexed by tour position, so the 2-opt action (i, j) and decoder are unchanged.
    tokens: str = "edge"
    positional_encoding: str = "cyclic"   # node_pe only: "cyclic" (period L) | "sin" (absolute position)
    geo_scalars: bool = True              # edge "geo" only: include the 9 engineered geometric scalars
    history_feature: bool = True          # per-token frequency of recently cut positions
    # Move operators, one pairwise head each (CVRP: "2opt", "relocate", "swap", "2opt*").
    operators: Tuple[str, ...] = ("2opt",)
    # Add beta_o * (-delta / mean edge length) to each move's logit, with the exact cost
    # change delta of the move and a learned per-operator beta_o (initialised to delta_bias_init).
    delta_bias: bool = False
    delta_bias_init: float = 10.0
    # ---- size-generalization factors (ACTION_PLAN.md, Phase 6); defaults = the released model
    # Attention scope: "dense" | "logn" (logits scaled by learned_h * log(L) / log(ref_len)) |
    # "local" (each token attends to its +-local_window tour neighbours and its local_k nearest tokens
    # by position: edge midpoints / node coordinates; O(L * K)).
    attention: str = "dense"
    ref_len: int = 50
    local_k: int = 16
    local_window: int = 2
    dropout: float = 0.0
    dropout_place: str = "standard"       # "standard" (sublayer output) | "pre_proj" (before the last projection)
                                          # | "pre_linear" (sublayer input, before QKV / FFN up-projection)
    dropout_kind: str = "standard"        # "standard" (inverted dropout) | "vpad" (variance-preserving affine)
    prenorm: bool = False
    # "none" | "density": multiply the absolute edge length feature by sqrt(V), so its scale does not
    # shrink as uniform instances get denser
    length_norm: str = "none"
    global_context: bool = True           # W_glob * mean-pooled summary in every token (Eq. 6)
    # Multiply the learned move scores by log(L^2) / log(ref_len^2) (sharper policy as moves grow)
    logit_size_scale: bool = False
    # >0: the action space is restricted to 2-opt moves that insert at least one edge between
    # k-nearest neighbours (LKH-style candidate set), in training and inference (TSP)
    candidates: int = 0
    # ---- size-generalization ideas, round 2 (ACTION_PLAN.md, Phase 6b)
    coord_embed: bool = True              # embed absolute endpoint coordinates (geo features)
    edge_direction: bool = True           # absolute unit direction of the edge (geo features)
    # "coords": edge lengths of the geometric scalars are Euclidean; "cost": they are read from the cost matrix
    # (road networks: angles still come from the coordinates, lengths from the objective)
    length_source: str = "coords"
    len_local_scale: bool = False         # add edge length / mean distance of its endpoints to their knn_k nearest nodes
    zscore: str = "global"                # "global": over the solution's edges | "local": vs the endpoint's k-NN distances
    knn_features: bool = False            # k-NN graph features: v in kNN(u), u in kNN(v), clipped ranks
    knn_k: int = 10
    # "pairwise": one softmax over all moves (Eq. 8) | "two_stage": p(i) * p(j | i), first the edge to cut,
    # then its partner (symmetrised over the two orders), so each decision is over L (not L^2) options |
    # "gain_mlp": no encoder; logit = beta * omega + C * tanh(MLP(per-move gain statistics)) (reviewer control)
    decoder: str = "pairwise"
    # ---- round 3 (Phase 6c)
    abs_length: bool = True               # keep the raw edge length (the one input that shrinks with n)
    # >0: polar description of each endpoint's polar_k nearest nodes in the edge's frame (angle to the edge
    # direction, radius / largest radius), pooled by a small DeepSets encoder (ELG / INViT-style local views)
    polar_k: int = 0
    polar_dim: int = 32
    # encoder attention biased by -gamma_h * log(1 + dist(midpoints) / local spacing), learned gamma_h >= 0
    # per head and layer (soft locality, cf. DAR / ELG distance penalties); dense attention only
    attn_dist_bias: bool = False
    # ---- symmetry ladder (Phase 6d)
    # turning angles as cos, |sin| and sin_u * sin_v (invariant to mirroring the instance)
    reflect_invariant: bool = False
    # every per-endpoint quantity enters as the sum and product of its tail and head values (token features
    # invariant to reversing the tour); implies reflect_invariant for the turning angles
    endpoint_symmetric: bool = False
    # cycle mixing phi([h_k, h_{k-1} + h_{k+1}, h_{k-1} * h_{k+1}]) instead of the ordered concatenation
    cycle_mixing_symmetric: bool = False
    # ---- RL-drift fixes (Phase 6e): search-progress inputs broadcast to every token: log(1 + moves / L),
    # log(1 + moves since the last new best / L), (current - best cost) / best cost
    progress_features: bool = False
    # ---- reviewer controls (2026-10-24)
    # "ratio": scale-free length features (length / neighbouring lengths, z-score, length / k-NN spacing) |
    # "raw": the same quantities without normalization (length, mean neighbouring length, mean and std of
    # the solution's edge lengths, k-NN spacing); requires abs_length and len_local_scale
    length_scale: str = "ratio"


@dataclass
class SearchConfig:
    history_len: int = 16                 # moves fed to the history feature
    tabu_tenure: int = 8                  # recent moves (and their swaps) that are masked
    temperature: float = 1.0              # inference only: moves are sampled from softmax(logits / T)
    temperature_end: Optional[float] = None   # if set, T decays geometrically to this value over the budget
    # inference only: apply up to this many sampled, non-overlapping 2-opt moves per policy evaluation
    # (0 = auto: one per 100 nodes); 1 = one move per step (default)
    moves_per_step: int = 1
    # inference only: average the policy's log-probabilities over the tour and its reverse, and over the
    # instance and its mirror image (4 evaluations per step; exact reversal / reflection invariance)
    symmetrize: bool = False
    # inference only: per-instance temperature T * (1 + adaptive_temp * log(1 + moves since last best / L)),
    # i.e. sample more broadly when the search stagnates (0 = off)
    adaptive_temp: float = 0.0


@dataclass
class ILConfig:
    size_range: Tuple[int, int] = (20, 50)
    batch_size: int = 256
    lookahead: int = 2                    # K of the exact oracle (TSP: 1 or 2, CVRP: 1)
    supervised_steps: int = 2             # states supervised along one oracle trajectory
    tie_atol: float = 1e-9
    tie_rtol: float = 1e-9
    oracle_block_m1: int = 128
    oracle_block_m2: int = 8192


@dataclass
class RLConfig:
    size_range: Tuple[int, int] = (20, 100)
    batch_size: int = 256
    group_size: int = 20                  # G rollouts per instance
    horizon: int = 32                     # T
    ppo_clip: float = 0.2
    normalize_advantages: bool = True
    ppo_microbatch: int = 1024            # rollouts per backward chunk (0 = whole batch); memory only
    # "best": group-relative improvement of the best-so-far cost with a shared cutoff (the paper);
    # "final": group-relative final cost of each rollout, credited to every step (never zero-signal).
    reward: str = "best"                  # also "best_rel": group-relative best cost reached
    ref: str = "warmup_best"              # C_ref: "warmup_best" (paper) | "current" (state at rollout start)
    entropy_coef: float = 0.0             # entropy bonus on the PPO steps
    # Optional rotation augmentation inside each group (Euclidean features only):
    # the G rollouts are split over R evenly spaced coordinate rotations, with a
    # Jensen-Shannon consistency penalty between the rotated policies.
    # >0: rollout length T = max(horizon_min, round(horizon_mult * n)) instead of the fixed horizon
    horizon_mult: float = 0.0
    horizon_min: int = 8
    # >0: add kl_anchor_coef * KL(p_anchor || p_theta) on the PPO states, p_anchor = frozen copy of the
    # policy at the start of RL (keeps RL close to the imitation policy)
    kl_anchor_coef: float = 0.0
    # share of RL batches that start from late-search states: a policy warmup of t0 ~ U{lo * L, ..., hi * L} moves
    # (late_start_mult = (lo, hi)) instead of U{0, ..., warmup_mult * L}, so the reward measures escaping local optima
    late_start_frac: float = 0.0
    late_start_mult: Tuple[float, float] = (1.0, 5.0)
    group_rotations: int = 1
    rotation_consistency_coef: float = 0.0


@dataclass
class TrainConfig:
    schedule: str = "IL:100,RL:200"
    epoch_size: int = 128_000             # instances per epoch
    lr: float = 1e-4
    lr_decay: float = 0.99                # per epoch
    weight_decay: float = 0.01            # AdamW (0.01 = the PyTorch default used by the released model)
    optimizer: str = "adamw"              # "adamw" | "muon_adamw" (Muon on 2-D matrices, AdamW on the rest; nico.optim)
    max_grad_norm: float = 1.0
    warmup_mult: float = 1.0              # warmup length t0 ~ U{0, ..., warmup_mult * n}
    init: str = "random"                  # training start solutions, see search.initial_solutions
    pool_size: int = 0                    # >0: persistent start states per size (nico.pool)
    pool_refresh: float = 0.1             # share of used pool entries replaced by fresh instances
    # Optional pool preloaded from a file (CVRP npz instances + zero-padded start solutions,
    # e.g. POMO outputs); its size is served from the file, other sizes are generated.
    pool_data: Optional[str] = None
    pool_solutions: Optional[str] = None
    init_checkpoint: Optional[str] = None # start from these model weights (fine-tuning)
    precision: str = "fp32"               # "fp32" (TF32 matmuls) | "bf16" (autocast in training epochs)
    # Safety net for unattended runs (0 = off): if the validation cost exceeds the best so far by
    # this fraction, restore the previous epoch's state and halve the learning rate (logged).
    rollback_tol: float = 0.0
    behavior_refresh: int = 20            # updates between refreshes of the frozen policy
    reset_optimizer_on_stage_change: bool = True
    checkpoint_every: int = 10
    seed: int = 42
    # Validation after every epoch on a fixed set of generated instances (0 = off).
    val_size: int = 0
    val_n: int = 100
    val_step_mult: float = 10.0
    # Optional fixed validation set instead of generated instances (CVRP npz, e.g. the POMO
    # test set), with initial solutions and reference costs; reports val_gap in percent.
    val_data: Optional[str] = None
    val_init: Optional[str] = None
    val_reference: Optional[str] = None
    val_greedy: bool = False              # also validate with argmax moves (reports val_greedy_*)
    # Size-extrapolation validation (TSP): fixed sets data/tsp_val/tsp_unif{n}_test_seed4321.pkl with LKH
    # references, extrap_step_mult * n moves from the stored order, every extrap_every epochs and at the end;
    # also one-step probes of the policy at the largest size. Logged as extrap/*.
    extrap_sizes: Tuple[int, ...] = ()
    extrap_num: int = 32
    extrap_every: int = 5
    extrap_step_mult: float = 10.0
    extrap_dir: str = "data/tsp_val"
    il: ILConfig = field(default_factory=ILConfig)
    rl: RLConfig = field(default_factory=RLConfig)


@dataclass
class DataConfig:
    # "uniform": Euclidean instances sampled on the fly.
    # "road":    RRNCO road-network cities (see nico.problems.road).
    source: str = "uniform"
    road_root: Optional[str] = None
    road_split: Optional[str] = None
    road_city_key: str = "train"
    road_val_key: str = "validation"       # split of the cities used for validation
    # CVRP only: empty routes added on top of the initial solution's routes.
    extra_routes: int = 2
    # TSP: share of training batches made of crops of large instances: the `size` nearest nodes to a random
    # centre in a uniform instance with crop_parent nodes (crop_parent ~ U[crop_parent_range]); keeps the
    # local density of large instances (coordinates are not rescaled)
    crop_frac: float = 0.0
    crop_parent_range: Tuple[int, int] = (200, 2000)
    # CVRP: per-batch capacity multiplier m ~ log-uniform[lo, hi] on the size-based capacity (Kool et al.),
    # so that training covers longer routes; (1, 1) = standard capacities
    cvrp_capacity_mult: Tuple[float, float] = (1.0, 1.0)


@dataclass
class EvalConfig:
    step_mult: float = 10.0               # budget = step_mult * n moves
    restarts: int = 1
    batch_size: int = 100
    sample: bool = True


@dataclass
class Config:
    problem: str = "tsp"                  # "tsp" | "cvrp"
    model: ModelConfig = field(default_factory=ModelConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    run_name: str = "nico"
    output_dir: str = "outputs"

    # ------------------------------------------------------------------ io
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Config":
        return _build(cls, raw or {})

    @classmethod
    def load(cls, path: str | Path, overrides: Optional[List[str]] = None) -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        for item in overrides or []:
            _apply_override(raw, item)
        return cls.from_dict(raw)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


def _build(cls, raw: Dict[str, Any]):
    names = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(raw) - set(names)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs = {}
    for name, value in raw.items():
        f = names[name]
        default = f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default
        if dataclasses.is_dataclass(default):
            kwargs[name] = _build(type(default), value or {})
        elif isinstance(default, tuple) and value is not None:
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _apply_override(raw: Dict[str, Any], item: str) -> None:
    key, _, value = item.partition("=")
    if not _:
        raise ValueError(f"Override must look like key.sub=value, got {item!r}")
    node = raw
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = yaml.safe_load(value)

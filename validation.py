import os
import time
import pickle
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import torch
from torch.utils.data import DataLoader
from utils import move_to, get_inner_model


# =========================
# History / Recording
# =========================
@dataclass
class RolloutHistory:
    best_costs: Optional[List[List[float]]] = None  # (T+1) rows, each len B
    times: Optional[List[float]] = None             # (T+1)
    events: List[List[Tuple[int, float, float]]] = field(default_factory=list)


class RolloutRecorder:
    def __init__(self, B: int, save: bool = False):
        self.save = bool(save)
        self.hist = RolloutHistory(
            best_costs=[] if self.save else None,
            times=[] if self.save else None,
            events=[[] for _ in range(B)],
        )

    def record_step(self, t: int, elapsed_s: float, best_val: torch.Tensor, improved_mask: torch.Tensor):
        if self.save:
            bv_cpu = best_val.detach().float().cpu().tolist()
            self.hist.best_costs.append(bv_cpu)
            self.hist.times.append(float(elapsed_s))

            if improved_mask.any():
                idx = improved_mask.nonzero(as_tuple=False).view(-1).detach().cpu().tolist()
                for b in idx:
                    self.hist.events[b].append((int(t), float(elapsed_s), float(bv_cpu[b])))

    def finalize(self):
        if self.save:
            return {"events": self.hist.events, "costs": self.hist.best_costs, "times": self.hist.times}
        else:
            return None


# =========================
# Small tensor helpers
# =========================
def envelope_min(v_big: torch.Tensor, K: int, B: int) -> torch.Tensor:
    """
    v_big: (K*B,) -> env: (B,) where env[b] = min_k v_big[k*B + b]
    """
    return v_big.view(K, B).min(dim=0).values


def _extract_optimal_costs(
    batch: Dict[str, Any],
    val_dataset: Any,
    batch_offset: int,
    batch_size: int,
    ref: torch.Tensor,
) -> Optional[torch.Tensor]:
    # Prefer batch-provided optimal costs if available (works with custom datasets/collate fns).
    for key in ("optimal_cost", "opt_cost", "best_known_cost", "target_cost"):
        if key in batch:
            vals = torch.as_tensor(batch[key], dtype=ref.dtype, device=ref.device).view(-1)
            return vals if vals.numel() == batch_size else None

    # Fallback: common dataset attribute names, sliced consistently with shuffle=False.
    for attr in ("optimal_costs", "opt_costs", "best_known_costs", "target_costs"):
        if not hasattr(val_dataset, attr):
            continue
        raw = getattr(val_dataset, attr)
        try:
            vals = torch.as_tensor(raw[batch_offset:batch_offset + batch_size], dtype=ref.dtype, device=ref.device).view(-1)
        except Exception:
            continue
        return vals if vals.numel() == batch_size else None

    return None


_CONCORDE_OPT_CACHE: Dict[str, Optional[torch.Tensor]] = {}


def _load_concorde_opt_costs(graph_type: str, graph_size: int, num_instances: int) -> Optional[torch.Tensor]:
    folder_name = f"tsp_{graph_type}{graph_size}_test_seed1234"
    rel_candidates = [
        os.path.join("..", "..", "data", "results"),
        os.path.join("..", "data", "results"),
    ]
    file_dir = os.path.dirname(os.path.abspath(__file__))
    rel_candidates.extend(
        [
            os.path.join(file_dir, "..", "..", "data", "results"),
            os.path.join(file_dir, "..", "data", "results"),
        ]
    )

    # Keep candidate order stable while removing duplicates.
    seen = set()
    roots = []
    for root in rel_candidates:
        root_abs = os.path.abspath(root)
        if root_abs in seen:
            continue
        seen.add(root_abs)
        roots.append(root_abs)

    path = None
    for root in roots:
        cand = os.path.join(root, folder_name, "concorde_costs.pkl")
        if os.path.exists(cand):
            path = cand
            break

    if path is None:
        return None

    if path not in _CONCORDE_OPT_CACHE:
        try:
            with open(path, "rb") as f:
                vals = pickle.load(f)
            _CONCORDE_OPT_CACHE[path] = torch.as_tensor(vals, dtype=torch.float32).view(-1).cpu()
        except Exception:
            _CONCORDE_OPT_CACHE[path] = None

    opt_all = _CONCORDE_OPT_CACHE[path]
    if opt_all is None:
        return None
    return opt_all[:num_instances]


def _get_tabu_cfg(opts) -> Dict[str, Any]:
    """
    Tabu configuration (used by training and evaluation):
      - tabu_mode:
          actions
          added_edges
      - tabu_action_tenure (int, default: mask_prev_actions)
      - tabu_edge_tenure   (int, default: action tenure)
    """
    mode = getattr(opts, "tabu_mode", getattr(opts, "eval_tabu_mode", "actions"))
    valid_modes = {"actions", "added_edges"}
    if mode not in valid_modes:
        raise ValueError(
            f"Unknown tabu_mode='{mode}'. Valid: {sorted(valid_modes)}"
        )

    def _ival(name: str, default: int) -> int:
        raw = getattr(opts, name, None)
        if raw is None:
            return max(0, int(default))
        return max(0, int(raw))

    action_tenure = _ival(
        "tabu_action_tenure",
        int(getattr(opts, "mask_prev_actions", 0)),
    )
    edge_tenure = _ival(
        "tabu_edge_tenure",
        action_tenure,
    )

    use_actions = mode == "actions"
    use_added_edges = mode == "added_edges"

    return {
        "mode": mode,
        "use_actions": bool(use_actions and action_tenure > 0),
        "use_added_edges": bool(use_added_edges and edge_tenure > 0),
        "action_tenure": int(action_tenure),
        "edge_tenure": int(edge_tenure),
    }


def _make_hist(B: int, K: int, device: torch.device) -> torch.Tensor:
    K = max(0, int(K))
    return torch.full((B, K, 2), -1, device=device, dtype=torch.long)


def _fifo_append_pairs(hist: torch.Tensor, new_pairs: torch.Tensor) -> torch.Tensor:
    """
    hist:      (B,K,2)
    new_pairs: (B,M,2)
    """
    K = int(hist.size(1))
    M = int(new_pairs.size(1))
    if K == 0 or M == 0:
        return hist
    if M >= K:
        return new_pairs[:, -K:, :].clone()

    out = torch.empty_like(hist)
    out[:, : K - M, :] = hist[:, M:, :]
    out[:, K - M :, :] = new_pairs
    return out


def _push_last_k_actions(last_k_actions: torch.Tensor, actions_ij: torch.Tensor) -> torch.Tensor:
    if last_k_actions.numel() == 0:
        return last_k_actions
    out = torch.roll(last_k_actions, shifts=-1, dims=1)
    out[:, -1, :] = actions_ij
    return out


def _canonicalize_undirected_pairs(pairs: torch.Tensor) -> torch.Tensor:
    a = pairs[..., 0]
    b = pairs[..., 1]
    lo = torch.minimum(a, b)
    hi = torch.maximum(a, b)
    return torch.stack([lo, hi], dim=-1)


def _extract_2opt_added_edges(tour: torch.Tensor, actions_ij: torch.Tensor) -> torch.Tensor:
    """
    tour:       (B,N) position -> node_id
    actions_ij: (B,2) with i<j, non-adjacent
    Returns:
      added_edges: (B,2,2) undirected node-edge pairs inserted by the move
    """
    B, N = tour.shape
    i = actions_ij[:, 0].long()
    j = actions_ij[:, 1].long()
    i1 = (i + 1) % N
    j1 = (j + 1) % N

    b = torch.arange(B, device=tour.device)
    a = tour[b, i]
    b1 = tour[b, i1]
    c = tour[b, j]
    d = tour[b, j1]

    added_edges = torch.stack(
        [torch.stack([a, c], dim=-1), torch.stack([b1, d], dim=-1)],
        dim=1,
    )
    return _canonicalize_undirected_pairs(added_edges)


def _mask_logits_with_action_history(logits_all: torch.Tensor, last_k_actions: torch.Tensor, N: int) -> torch.Tensor:
    """
    Masks exact action pairs (and their symmetric counterparts).
    logits_all: (B, N*N)
    """
    if last_k_actions is None or last_k_actions.numel() == 0:
        return logits_all

    i = last_k_actions[..., 0]
    j = last_k_actions[..., 1]
    valid = (i >= 0) & (j >= 0)
    if not bool(valid.any()):
        return logits_all

    i = i.clamp(0, N - 1).long()
    j = j.clamp(0, N - 1).long()
    idx1 = i * N + j
    idx2 = j * N + i
    idx = torch.cat([idx1, idx2], dim=1)
    valid2 = torch.cat([valid, valid], dim=1)

    out = logits_all.clone()
    b = torch.arange(logits_all.size(0), device=logits_all.device).view(-1, 1).expand_as(idx)
    b_flat = b.reshape(-1)
    idx_flat = idx.reshape(-1)
    v_flat = valid2.reshape(-1)
    out[b_flat[v_flat], idx_flat[v_flat]] = -float("inf")
    return out


def _tabu_positions_from_edge_hist(tour: torch.Tensor, edge_hist: torch.Tensor) -> torch.Tensor:
    """
    Returns position-level tabu flags:
      pos_tabu[b,p] == True if edge (tour[b,p], tour[b,p+1]) is tabu.
    """
    B, N = tour.shape
    if edge_hist is None or edge_hist.numel() == 0:
        return torch.zeros((B, N), device=tour.device, dtype=torch.bool)

    valid = (edge_hist[..., 0] >= 0) & (edge_hist[..., 1] >= 0)
    if not bool(valid.any()):
        return torch.zeros((B, N), device=tour.device, dtype=torch.bool)

    u = tour
    v = torch.roll(tour, shifts=-1, dims=1)
    cur_edges = _canonicalize_undirected_pairs(torch.stack([u, v], dim=-1))  # (B,N,2)

    match_u = cur_edges[..., 0:1] == edge_hist[:, None, :, 0]
    match_v = cur_edges[..., 1:2] == edge_hist[:, None, :, 1]
    match = match_u & match_v & valid[:, None, :]
    return match.any(dim=-1)


def _mask_logits_with_tabu_positions(logits_all: torch.Tensor, pos_tabu: torch.Tensor, N: int) -> torch.Tensor:
    if pos_tabu is None or not bool(pos_tabu.any()):
        return logits_all
    pair_mask = pos_tabu[:, :, None] | pos_tabu[:, None, :]
    out = logits_all.clone()
    out[pair_mask.view(logits_all.size(0), N * N)] = -float("inf")
    return out


def _select_actions_from_logits(logits_all: torch.Tensor, N: int, do_sample: bool) -> torch.Tensor:
    if do_sample:
        pair_index = torch.distributions.Categorical(logits=logits_all).sample().unsqueeze(-1)
    else:
        pair_index = logits_all.argmax(dim=-1, keepdim=True)

    col = pair_index % N
    row = torch.div(pair_index, N, rounding_mode="trunc")
    return torch.cat([row, col], dim=-1).long()


def _logp_of_actions_from_logits(logits_all: torch.Tensor, actions_ij: torch.Tensor, N: int) -> torch.Tensor:
    logp_all = torch.log_softmax(logits_all, dim=-1)
    row = actions_ij[:, 0].clamp(0, N - 1).long()
    col = actions_ij[:, 1].clamp(0, N - 1).long()
    idx = (row * N + col).view(-1, 1)
    return logp_all.gather(1, idx).squeeze(-1)


def _init_tabu_state(B: int, device: torch.device, opts) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cfg = _get_tabu_cfg(opts)
    state = {
        "action_hist": _make_hist(B, cfg["action_tenure"], device) if cfg["use_actions"] else _make_hist(B, 0, device),
        "edge_hist": _make_hist(
            B,
            2 * cfg["edge_tenure"] if cfg["use_added_edges"] else 0,
            device,
        ),
    }
    return state, cfg


def _apply_tabu_mask(logits_all: torch.Tensor, tour: torch.Tensor, tabu_state: Dict[str, Any], cfg: Dict[str, Any]) -> torch.Tensor:
    _, N = tour.shape
    masked = logits_all
    if cfg["use_actions"]:
        masked = _mask_logits_with_action_history(masked, tabu_state["action_hist"], N)
    if cfg["use_added_edges"]:
        pos_tabu = _tabu_positions_from_edge_hist(tour, tabu_state["edge_hist"])
        masked = _mask_logits_with_tabu_positions(masked, pos_tabu, N)

    # Safety fallback: if a row got fully masked, revert to base logits for that row.
    dead_rows = (~torch.isfinite(masked)).all(dim=-1)
    if bool(dead_rows.any()):
        masked = masked.clone()
        masked[dead_rows] = logits_all[dead_rows]
    return masked


def _update_tabu_state(tabu_state: Dict[str, Any], cfg: Dict[str, Any], prev_tour: torch.Tensor, actions_ij: torch.Tensor) -> None:
    if cfg["use_actions"]:
        tabu_state["action_hist"] = _fifo_append_pairs(tabu_state["action_hist"], actions_ij.unsqueeze(1))

    if not cfg["use_added_edges"]:
        return

    added_edges = _extract_2opt_added_edges(prev_tour, actions_ij)
    tabu_state["edge_hist"] = _fifo_append_pairs(tabu_state["edge_hist"], added_edges)


# =========================
# Plain rollout(s)
# =========================
@torch.no_grad()
def rollout_plain(
    problem,
    model,
    coords: torch.Tensor,     # (B,N,2)
    solution: torch.Tensor,   # (B,N)
    value: torch.Tensor,      # (B,)
    opts,
    T: int,
    do_sample: bool,
    save: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    B = solution.size(0)
    N = solution.size(1)

    sol = solution.clone()
    cost = value
    best_logic = cost.clone()
    best_report = cost.clone()

    # Kept for model action-history features.
    model_hist = _make_hist(B, int(getattr(opts, "action_hist_len", 0)), opts.device)
    tabu_state, tabu_cfg = _init_tabu_state(B, opts.device, opts)

    rec = RolloutRecorder(B=B, save=save)
    inner = get_inner_model(model)

    start_t = time.time()
    rec.record_step(0, time.time() - start_t, best_report, torch.zeros(B, device=opts.device, dtype=torch.bool))

    improvements, rewards = [], []
    for t in range(1, T + 1):
        logits_all = inner(
            x=coords,
            solutions=sol,
            last_k_actions=model_hist,
            tabu_edge_hist=tabu_state["edge_hist"],
        )
        if logits_all.dim() == 3:
            logits_all = logits_all.view(B, -1)

        masked_logits = _apply_tabu_mask(logits_all, sol, tabu_state, tabu_cfg)
        exchange = _select_actions_from_logits(masked_logits, N, do_sample=do_sample)

        prev_sol = sol
        sol = move_to(problem.step(sol, exchange), opts.device)
        obj = problem.get_costs(coords, sol)

        improvements.append(cost - obj)
        cost = obj

        new_best = torch.minimum(best_report, obj)
        rewards.append(best_report - new_best)

        improved = new_best < best_report
        best_logic = torch.minimum(best_logic, obj)
        best_report = new_best

        model_hist = _push_last_k_actions(model_hist, exchange)
        _update_tabu_state(tabu_state, tabu_cfg, prev_sol, exchange)

        rec.record_step(t, time.time() - start_t, best_report, improved)

    hist = rec.finalize()
    return best_report.view(-1, 1), torch.stack(improvements, 1), torch.stack(rewards, 1), hist


@torch.no_grad()
def rollout_multistart_parallel_envelope(
    problem,
    model,
    coords: torch.Tensor,               # (B,N,2)
    init_solutions: List[torch.Tensor], # K x (B,N)
    init_values: List[torch.Tensor],    # K x (B,)
    opts,
    T: int,
    do_sample: bool,
    save: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Runs K starts in parallel by expanding batch to (K*B).
    The trace/history stores the *envelope* curve per original instance:
      env_best[t,b] = min_k best_big[t, k*B + b]
    """
    K = len(init_solutions)
    assert K >= 1
    B, N = init_solutions[0].shape

    # Expand batch
    coords_big = coords.unsqueeze(0).expand(K, *coords.shape).reshape(K * coords.size(0), *coords.shape[1:])

    sol_big = torch.cat(init_solutions, dim=0)                         # (K*B,N)
    cost_big = torch.cat([v.view(-1) for v in init_values], dim=0)     # (K*B,)
    best_big = cost_big.clone()

    model_hist = _make_hist(K * B, int(getattr(opts, "action_hist_len", 0)), opts.device)
    tabu_state, tabu_cfg = _init_tabu_state(K * B, opts.device, opts)

    # Envelope state (per original instance)
    env_cost = envelope_min(cost_big, K, B)
    env_best = envelope_min(best_big, K, B)

    rec = RolloutRecorder(B=B, save=save)
    inner = get_inner_model(model)

    start_t = time.time()
    rec.record_step(0, time.time() - start_t, env_best, torch.zeros(B, device=opts.device, dtype=torch.bool))

    improvements, rewards = [], []

    for t in range(1, T + 1):
        logits_all = inner(
            x=coords_big,
            solutions=sol_big,
            last_k_actions=model_hist,
            tabu_edge_hist=tabu_state["edge_hist"],
        )
        if logits_all.dim() == 3:
            logits_all = logits_all.view(K * B, -1)

        masked_logits = _apply_tabu_mask(logits_all, sol_big, tabu_state, tabu_cfg)
        exchange = _select_actions_from_logits(masked_logits, N, do_sample=do_sample)

        prev_sol_big = sol_big
        sol_big = move_to(problem.step(sol_big, exchange), opts.device)
        obj_big = problem.get_costs(coords_big, sol_big)

        cost_big = obj_big
        best_big = torch.minimum(best_big, obj_big)

        env_cost_new = envelope_min(cost_big, K, B)
        env_best_new = envelope_min(best_big, K, B)

        improvements.append(env_cost - env_cost_new)
        rewards.append(env_best - env_best_new)

        improved = env_best_new < env_best
        env_cost, env_best = env_cost_new, env_best_new

        model_hist = _push_last_k_actions(model_hist, exchange)
        _update_tabu_state(tabu_state, tabu_cfg, prev_sol_big, exchange)

        rec.record_step(t, time.time() - start_t, env_best, improved)

    hist = rec.finalize()
    return env_best.view(-1, 1), torch.stack(improvements, 1), torch.stack(rewards, 1), hist


# =========================
# Validation
# =========================
def validate(problem, model, val_datasets, opts, save_full_trace=False):
    model.eval()
    val_metrics: Dict[str, float] = {}
    all_history: Dict[str, Any] = {}

    eval_bs = int(getattr(opts, "eval_batch_size", 1))

    for graph_type, graph_size, val_dataset in zip(opts.val_graph_types, opts.val_graph_sizes, val_datasets):
        tag = f"{graph_type}{graph_size}"
        dataset_concorde_opt = _load_concorde_opt_costs(str(graph_type), int(graph_size), len(val_dataset))

        problem.size = int(graph_size)
        problem.graph_type = str(graph_type)
        T_max = int(getattr(opts, "T_max_eval_mult", 1.0) * graph_size)

        restarts = max(1, int(getattr(opts, "eval_restarts", 1)))
        init_method = str(getattr(opts, "eval_init_method", "sequential")).lower()

        load_path = None
        if init_method == "load":
            load_path = getattr(opts, "eval_init_path", None)
            assert load_path is not None, "opts.eval_init_path must be set when eval_init_method='load'"

        batch_offset = 0
        batch_times: List[float] = []
        batch_sizes: List[int] = []

        best_value_chunks: List[torch.Tensor] = []
        improvement_chunks: List[torch.Tensor] = []
        reward_chunks: List[torch.Tensor] = []
        optimal_value_chunks: List[torch.Tensor] = []
        have_optimal_for_all = True
        batch_histories: List[Dict[str, Any]] = []

        loader = DataLoader(val_dataset, batch_size=eval_bs, shuffle=False, pin_memory=True)

        for batch in loader:
            coords = move_to(batch["coords"], opts.device)  # coords-only
            B = int(coords.size(0))
            batch_sizes.append(B)

            # --- inits per batch (K restarts) ---
            init_solutions = problem.get_initial_solutions(
                B,
                init_method=init_method,
                restarts=restarts,
                device=opts.device,
                batch_offset=batch_offset,
                load_path=load_path,
            )
            init_values = [problem.get_costs(coords, s) for s in init_solutions]  # list of (B,)
            opt_costs = _extract_optimal_costs(batch, val_dataset, batch_offset, B, init_values[0])
            if opt_costs is None and dataset_concorde_opt is not None:
                sl = dataset_concorde_opt[batch_offset:batch_offset + B]
                if int(sl.numel()) == B:
                    opt_costs = sl.to(device=init_values[0].device, dtype=init_values[0].dtype)

            if opt_costs is None:
                have_optimal_for_all = False
            else:
                optimal_value_chunks.append(opt_costs.detach())

            t0 = time.time()
            if restarts == 1:
                bv, imp, r, hist = rollout_plain(
                    problem, model,
                    coords, init_solutions[0], init_values[0],
                    opts, T_max, do_sample=True, save=save_full_trace,
                )
            else:
                bv, imp, r, hist = rollout_multistart_parallel_envelope(
                    problem, model,
                    coords, init_solutions, init_values,
                    opts, T_max, do_sample=True, save=save_full_trace,
                )
            batch_times.append(float(time.time() - t0))

            best_value_chunks.append(bv.detach())
            improvement_chunks.append(imp.detach())
            reward_chunks.append(r.detach())
            if save_full_trace:
                batch_histories.append(hist)

            batch_offset += B

        best_value = torch.cat(best_value_chunks, 0)  # (N,1)
        improvement = torch.cat(improvement_chunks, 0)  # (N,T)
        reward = torch.cat(reward_chunks, 0)  # (N,T)

        if save_full_trace:
            all_history[tag] = batch_histories

        total_time_s = float(sum(batch_times))
        total_instances = int(sum(batch_sizes))
        avg_time_s = total_time_s / max(1, total_instances)
        steps_per_instance = int(reward.size(1)) if reward.dim() >= 2 else 0

        best_flat = best_value.view(-1).float()
        best_mean = float(best_flat.mean().item())
        best_std = float(best_flat.std(unbiased=False).item()) if best_flat.numel() > 1 else 0.0

        gap_to_opt_pct: Optional[float] = None
        if have_optimal_for_all and optimal_value_chunks:
            opt_flat = torch.cat(optimal_value_chunks, 0).view(-1).float()
            if opt_flat.numel() == best_flat.numel():
                valid = torch.isfinite(opt_flat) & (opt_flat > 0)
                if bool(valid.all()):
                    gap_to_opt_pct = float((((best_flat - opt_flat) / opt_flat) * 100.0).mean().item())

        if getattr(opts, "verbose", False):
            gap_str = f"{gap_to_opt_pct:.4f}%" if gap_to_opt_pct is not None else "N/A"
            init_str = f"init: {init_method}"
            if restarts > 1:
                init_str += f" | parallel inits: {restarts}"
            print(
                f"[{tag}: {total_instances}] avg cost: {best_mean:.6f} +- {best_std:.6f} | "
                f"gap to optimal: {gap_str} | "
                f"{init_str} | "
                f"steps: {steps_per_instance}/inst | "
                f"total time: {total_time_s:.3f}s | "
                f"avg t: {avg_time_s:.6f}s/inst"
            )

        val_metrics[f"val_{tag}/best_cost_mean"] = best_mean
        val_metrics[f"val_{tag}/avg_reward"] = reward.mean().item()
        val_metrics[f"val_{tag}/avg_improvement_per_step"] = improvement.mean().item()
        val_metrics[f"val_{tag}/total_time_s"] = total_time_s
        val_metrics[f"val_{tag}/avg_time_s"] = avg_time_s
        if gap_to_opt_pct is not None:
            val_metrics[f"val_{tag}/gap_to_optimal_pct"] = gap_to_opt_pct

        if save_full_trace and (opts.save_dir is not None):
            os.makedirs(opts.save_dir, exist_ok=True)
            save_path = os.path.join(opts.save_dir, f"{tag}_{init_method}Init_seed{opts.seed}.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(all_history, f, protocol=pickle.HIGHEST_PROTOCOL)

    return val_metrics

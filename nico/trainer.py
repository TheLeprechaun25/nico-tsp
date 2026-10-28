"""Two-stage training driver: IL epochs followed by group-RL epochs."""

from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from .checkpoint import save_checkpoint, set_rng_state
from .config import Config
from .data import InstanceSource
from .evaluate import evaluate
from .model import NICOPolicy
from .pool import StatePool
from .problems import problem_from_config
from .train_il import train_il_epoch
from .train_rl import train_rl_epoch


def parse_schedule(schedule: str) -> List[str]:
    """'IL:100,RL:200' -> ['il'] * 100 + ['rl'] * 200."""
    stages: List[str] = []
    for part in schedule.split(","):
        name, _, count = part.strip().partition(":")
        name = name.strip().lower()
        if name not in ("il", "rl") or not count.strip().isdigit():
            raise ValueError(f"Bad schedule entry {part!r}")
        stages += [name] * int(count)
    return stages


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_validator(cfg: Config, problem, device) -> Optional[Callable[[NICOPolicy], dict]]:
    """Mean best cost (and gap, with a reference) after ``val_step_mult * n`` sampled moves
    on a fixed set: ``val_data`` if given, else generated instances from random starts."""
    if not cfg.train.val_size:
        return None
    ref = None
    if cfg.train.val_data:
        import numpy as np

        instance = problem.load_npz(cfg.train.val_data, cfg.train.val_size)
        with np.load(cfg.train.val_init) as z:
            init = problem.from_sequences(z["solutions"][:cfg.train.val_size])
        if cfg.train.val_reference:
            with np.load(cfg.train.val_reference) as z:
                ref = torch.tensor(z["reference_costs"][:cfg.train.val_size], dtype=torch.float32, device=device)
        instance = {k: v.to(device) for k, v in instance.items()}
        init = init.to(device)
    else:
        gen = torch.Generator(device="cpu").manual_seed(1234)
        if cfg.data.source == "road":     # held-out validation cities, never the test cities
            from .problems.road import RoadCitySampler
            instance = RoadCitySampler(cfg.data.road_root, cfg.data.road_split, cfg.data.road_val_key,
                                       seed=1234).sample(cfg.train.val_size, cfg.train.val_n, "cpu")
        else:
            instance = problem.sample_instances(cfg.train.val_size, cfg.train.val_n, "cpu", generator=gen)
        instance = {k: v.to(device) for k, v in instance.items()}
        init = problem.random_solutions(instance, generator=torch.Generator(device=device).manual_seed(1234))
    n = instance["cost_matrix"].size(1) - (1 if cfg.problem == "cvrp" else 0)
    steps = int(cfg.train.val_step_mult * n)

    def run(model: NICOPolicy, sample: bool, prefix: str) -> dict:
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(0)
            out = evaluate(model, problem, instance, cfg.search, steps=steps, init=init,
                           batch_size=cfg.eval.batch_size, sample=sample)
        metrics = {f"{prefix}_cost": out["cost"].mean().item(), f"{prefix}_seconds": out["seconds"].item()}
        if ref is not None:
            metrics[f"{prefix}_gap"] = (100 * (out["cost"] - ref) / ref).mean().item()
        return metrics

    def validate(model: NICOPolicy) -> dict:
        metrics = run(model, True, "val")
        if cfg.train.val_greedy:
            metrics.update(run(model, False, "val_greedy"))
        model.train()
        return metrics

    return validate


def make_extrapolator(cfg: Config, problem, device) -> Optional[Callable[[NICOPolicy], dict]]:
    """Size-extrapolation validation (TSP): gap to LKH after ``extrap_step_mult * n`` sampled moves from
    the stored order on fixed sets at every size in ``extrap_sizes``, plus one-step probes of the policy at
    the largest size (states half-way through a greedy 2-opt descent): probability mass on improving
    moves, probability and rank of the best move, normalized entropy."""
    t = cfg.train
    if not t.extrap_sizes:
        return None
    if cfg.problem == "cvrp":
        return _make_cvrp_extrapolator(cfg, problem, device)
    import pickle

    from .problems import TSP
    from .search import SearchState, policy_logits, two_opt_descent

    root = Path(t.extrap_dir)
    sets = {}
    for n in t.extrap_sizes:
        name = f"tsp_unif{n}_test_seed4321"
        with open(root / f"{name}.pkl", "rb") as f:
            coords = torch.as_tensor(np.asarray(pickle.load(f), dtype=np.float32)[: t.extrap_num])
        with open(root / "results" / name / "lkh_costs.pkl", "rb") as f:
            ref = torch.as_tensor(np.asarray(pickle.load(f), dtype=np.float64)[: t.extrap_num])
        sets[n] = ({k: v.to(device) for k, v in TSP.from_coords(coords).items()}, ref)
    n_max = max(t.extrap_sizes)
    inst_p = sets[n_max][0]
    start = problem.identity_solutions(inst_p)
    probe_sol = two_opt_descent(problem, inst_p, start, n_max // 2)     # ~ half-way through the descent

    @torch.no_grad()
    def extrapolate(model: NICOPolicy) -> dict:
        model.eval()
        out = {}
        for n, (inst, ref) in sets.items():
            with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                torch.manual_seed(0)
                res = evaluate(model, problem, inst, cfg.search, steps=int(t.extrap_step_mult * n),
                               init="identity", batch_size=max(1, min(t.extrap_num, 16000 // n)), sample=True)
            out[f"extrap/gap_n{n}"] = float((100 * (res["cost"].double().cpu() / ref - 1)).mean())
            out[f"extrap/seconds_n{n}"] = float(res["seconds"])
        # one-step probes at the largest size
        B, L = probe_sol.shape
        state = SearchState.initial(probe_sol, cfg.search)
        logits, _ = policy_logits(model, problem, inst_p, state)
        p = torch.softmax(logits.float(), dim=-1)
        values = problem.move_values(inst_p, probe_sol)                     # inf = invalid
        valid = torch.isfinite(values)
        best = values.masked_fill(~valid, float("inf")).argmin(dim=1)
        out[f"extrap/probe_n{n_max}_improving_mass"] = float((p * (values < -1e-9)).sum(1).mean())
        out[f"extrap/probe_n{n_max}_best_prob"] = float(p.gather(1, best.view(-1, 1)).mean())
        rank = (logits > logits.gather(1, best.view(-1, 1))).sum(1).float()
        out[f"extrap/probe_n{n_max}_best_rank"] = float(rank.median())
        ent = -(torch.xlogy(p, p)).sum(1) / valid.sum(1).float().log()
        out[f"extrap/probe_n{n_max}_norm_entropy"] = float(ent.mean())
        model.train()
        return out

    return extrapolate


def _make_cvrp_extrapolator(cfg: Config, problem, device) -> Callable[[NICOPolicy], dict]:
    """CVRP size-extrapolation validation: gap to HGS after ``extrap_step_mult * n`` sampled moves from the
    Clarke-Wright savings solutions, on the validation sets <extrap_dir>/cvrp{n}_val128_seed5678.npz
    (references hgs_cvrp{n}_val128_seed5678.npz). Logged as extrap/gap_n{n}."""
    t = cfg.train
    root = Path(t.extrap_dir)
    sets = {}
    for n in t.extrap_sizes:
        inst = problem.load_npz(root / f"cvrp{n}_val128_seed5678.npz", t.extrap_num)
        with np.load(root / f"hgs_cvrp{n}_val128_seed5678.npz") as z:
            ref = torch.as_tensor(np.asarray(z["reference_costs"], dtype=np.float64)[: t.extrap_num])
        start = problem.savings_solutions(inst)
        sets[n] = ({k: v.to(device) for k, v in inst.items()}, start.to(device), ref)

    @torch.no_grad()
    def extrapolate(model: NICOPolicy) -> dict:
        model.eval()
        out = {}
        for n, (inst, start, ref) in sets.items():
            with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                torch.manual_seed(0)
                res = evaluate(model, problem, inst, cfg.search, steps=int(t.extrap_step_mult * n), init=start,
                               batch_size=max(1, min(t.extrap_num, 16000 // n)), sample=True)
            out[f"extrap/gap_n{n}"] = float((100 * (res["cost"].double().cpu() / ref - 1)).mean())
        model.train()
        return out

    return extrapolate


def make_anchor(model: NICOPolicy, out_dir: Path, device) -> NICOPolicy:
    """Frozen copy of the imitation policy: the last saved IL checkpoint of this run if any (so a resumed
    run anchors to the same policy), otherwise the current model (the start of RL)."""
    anchor = copy.deepcopy(model)
    il = sorted(out_dir.glob("il_epoch-*.pt"), key=lambda p: int(p.stem.split("-")[1]))
    if il:
        anchor.load_state_dict(torch.load(il[-1], map_location=device, weights_only=False)["model"], strict=True)
        print(f"KL anchor: {il[-1]}", flush=True)
    else:
        print("KL anchor: current model", flush=True)
    anchor.eval()
    for p in anchor.parameters():
        p.requires_grad_(False)
    return anchor


def train(cfg: Config, device, init_model: Optional[NICOPolicy] = None,
          log: Optional[Callable[[dict], None]] = None, resume: bool = False) -> Tuple[NICOPolicy, Path]:
    """Run the schedule. With ``resume``, continue from ``<out_dir>/last.pt`` if it exists."""
    device = torch.device(device)
    seed_everything(cfg.train.seed)
    problem = problem_from_config(cfg)
    source = InstanceSource(cfg, problem)
    validate = make_validator(cfg, problem, device)
    extrapolate = make_extrapolator(cfg, problem, device)
    pool = StatePool(cfg, problem, source, device) if cfg.train.pool_size > 0 else None
    if cfg.train.precision not in ("fp32", "bf16", "fp16"):
        raise ValueError(f"Unknown train.precision {cfg.train.precision!r}")
    amp_dtype = ({"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.train.precision)
                 if device.type == "cuda" else None)
    # fp16 needs loss scaling; bf16/fp32 do not
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    model = init_model if init_model is not None else NICOPolicy(cfg.model, cfg.problem)
    model = model.to(device)
    if cfg.train.init_checkpoint and init_model is None:
        start = torch.load(cfg.train.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(start["model"], strict=True)
        print(f"Initialised from {cfg.train.init_checkpoint}")
    if cfg.train.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    elif cfg.train.optimizer == "muon_adamw":
        if amp_dtype == torch.float16:
            raise ValueError("muon_adamw does not support the fp16 GradScaler")
        from .optim import MuonAdamW
        optimizer = MuonAdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    else:
        raise ValueError(f"Unknown train.optimizer {cfg.train.optimizer!r}")
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.train.lr_decay)

    out_dir = Path(cfg.output_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(out_dir / "config.yaml")
    stages = parse_schedule(cfg.train.schedule)

    prev, first_epoch, best_val = None, 0, float("inf")
    anchor = None   # KL anchor of the RL stage (train.rl.kl_anchor_coef), created at the first RL epoch
    last_path = out_dir / "last.pt"
    if resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        set_rng_state(state["rng_state"])
        first_epoch, prev = state["epoch"] + 1, state["stage"]
        print(f"Resumed from {last_path} at epoch {first_epoch}")
    for epoch in range(first_epoch, len(stages)):
        stage = stages[epoch]
        if prev is not None and stage != prev and cfg.train.reset_optimizer_on_stage_change:
            optimizer.state.clear()
        frozen = copy.deepcopy(model).eval()
        for p in frozen.parameters():
            p.requires_grad_(False)
        start = time.time()
        run_epoch = train_il_epoch if stage == "il" else train_rl_epoch
        # cache_enabled=False: the context spans the whole epoch, and autocast's weight-cast cache
        # would otherwise keep using the low-precision weights from the epoch's first step while
        # the optimizer updates the fp32 weights (this silently froze the encoder and broke RL).
        extra = {}
        if stage == "rl" and cfg.train.rl.kl_anchor_coef > 0:
            if anchor is None:
                anchor = make_anchor(model, out_dir, device)
            extra["anchor"] = anchor
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None, cache_enabled=False):
            metrics = run_epoch(model, frozen, optimizer, problem, source, cfg, device, pool=pool,
                                scaler=scaler, **extra)
        if pool is not None:
            metrics["pool/mean_cost"] = pool.depth()
        scheduler.step()
        metrics.update(epoch=epoch, stage=stage, lr=scheduler.get_last_lr()[0], seconds=time.time() - start)
        if validate is not None:
            metrics.update(validate(model))
            val = metrics["val_cost"]
            if (cfg.train.rollback_tol > 0 and val > best_val * (1 + cfg.train.rollback_tol)
                    and last_path.exists()):
                good = torch.load(last_path, map_location=device, weights_only=False)
                model.load_state_dict(good["model"], strict=True)
                optimizer.load_state_dict(good["optimizer"])
                for group in optimizer.param_groups:
                    group["lr"] *= 0.5
                scheduler.base_lrs = [lr * 0.5 for lr in scheduler.base_lrs]
                metrics["rollback"] = 1.0
                metrics["rollback_from_val"] = val
                print(f"Rollback at epoch {epoch}: val {val:.4f} > best {best_val:.4f}; lr halved", flush=True)
            else:
                best_val = min(best_val, val)
        last = epoch == len(stages) - 1
        if extrapolate is not None and (last or (epoch + 1) % max(1, cfg.train.extrap_every) == 0):
            metrics.update(extrapolate(model))
        with open(out_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics), flush=True)
        if log is not None:
            log(metrics)
        last = epoch == len(stages) - 1
        stage_end = last or stages[epoch + 1] != stage
        if last or stage_end or (cfg.train.checkpoint_every and (epoch + 1) % cfg.train.checkpoint_every == 0):
            save_checkpoint(out_dir / f"{stage}_epoch-{epoch + 1}.pt", model, cfg, epoch=epoch,
                            stage=stage, optimizer=optimizer, scheduler=scheduler)
        save_checkpoint(last_path, model, cfg, epoch=epoch, stage=stage, optimizer=optimizer, scheduler=scheduler)
        prev = stage
    return model, out_dir

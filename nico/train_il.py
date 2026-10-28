"""Stage 1: oracle-guided imitation learning (Algorithm 1)."""

from __future__ import annotations

import math
from typing import Dict

import torch

from .config import Config
from .oracle import flat_optimal_moves, flat_optimal_moves_k2, move_table, optimal_first_moves
from .problems.base import Instance
from .search import SearchState, advance, initial_solutions, policy_logits, warmup


def il_loss(model, frozen, problem, instance: Instance, cfg: Config, solution=None):
    """Set-valued imitation loss on one batch.

    From a warmed-up random solution, the state is advanced along one
    oracle-optimal trajectory for ``supervised_steps`` steps. At step t the
    target is the set A* of optimal first moves with lookahead
    min(K, supervised_steps - t), and the loss is -log of the mean policy
    probability over A* (equal to -log p(A*) up to the constant log|A*|).
    """
    il = cfg.train.il
    if solution is None:
        solution = initial_solutions(problem, instance, cfg)
    L = solution.size(1)
    state = SearchState.initial(solution, cfg.search)
    state, best_cost, best_sol = warmup(frozen, problem, instance, state, int(cfg.train.warmup_mult * L),
                                        return_best=True)

    multi = getattr(problem, "n_ops", 1) > 1
    if multi:
        flat = torch.arange(problem.n_ops * L * L, device=solution.device)
        pairs = torch.stack([flat // L, flat % L], 1)          # encoded (op * L + i, j)
    else:
        table = move_table(L, solution.device)
        flat, pairs = table["pairs"][:, 0] * L + table["pairs"][:, 1], table["pairs"]
    losses, hits, cand_kept, fallback = [], [], [], []
    for t in range(il.supervised_steps):
        k = min(il.lookahead, il.supervised_steps - t)
        if multi and k == 2:
            target, _ = flat_optimal_moves_k2(problem, instance, state.solution, il.tie_atol, il.tie_rtol)
        elif multi:
            target, _ = flat_optimal_moves(problem, instance, state.solution, il.tie_atol, il.tie_rtol)
        else:
            target, _ = optimal_first_moves(
                problem, instance, state.solution, k, il.tie_atol, il.tie_rtol,
                il.oracle_block_m1, il.oracle_block_m2,
            )
        masked, base = policy_logits(model, problem, instance, state)
        keep, oracle_target = None, target
        if model.cfg.candidates > 0:
            # optimal moves outside the candidate set cannot be imitated: restrict the targets, and
            # drop states whose optimal moves are all outside it (the oracle itself is unrestricted)
            target = target & torch.isfinite(base.float())[:, flat]
            keep = target.any(dim=1)
            target = target | ~keep.unsqueeze(1) & torch.isfinite(base.float())[:, flat]
        logp = torch.log_softmax(masked.float(), dim=-1)[:, flat].masked_fill(~target, float("-inf"))
        dead = torch.isneginf(logp).all(dim=1)
        fallback.append(dead.float().mean().detach())
        if bool(dead.any()):  # every optimal move is tabu: fall back to the unmasked policy
            base_logp = torch.log_softmax(base.float(), dim=-1)[:, flat].masked_fill(~target, float("-inf"))
            logp = torch.where(dead.unsqueeze(1), base_logp, logp)
        log_mass = torch.logsumexp(logp, dim=1) - target.sum(dim=1).clamp_min(1).float().log()
        if keep is not None:
            log_mass = log_mass * keep.float()
            cand_kept.append(keep.float().mean())
        losses.append(-log_mass)

        with torch.no_grad():
            greedy = masked.argmax(dim=-1)
            hits.append((target & (flat.unsqueeze(0) == greedy.unsqueeze(1))).any(dim=1).float())
            chosen = torch.multinomial(oracle_target.float(), 1).squeeze(1)
            state = advance(problem, instance, state, pairs[chosen])

    loss = torch.stack(losses).mean()
    stats = {
        "il/loss": float(loss.detach()),
        "il/greedy_in_target_step1": float(hits[0].mean()),
        "il/greedy_in_target_all": float(torch.stack(hits).prod(dim=0).mean()),
        "il/tabu_fallback_frac": float(torch.stack(fallback).mean()),   # states whose optimal moves are all tabu
    }
    if cand_kept:
        stats["il/target_in_candidates"] = float(torch.stack(cand_kept).mean())
    stats["best_cost"], stats["best_solution"] = best_cost, best_sol
    return loss, stats


def train_il_epoch(model, frozen, optimizer, problem, source, cfg: Config, device,
                   pool=None, scaler=None) -> Dict[str, float]:
    il = cfg.train.il
    updates = math.ceil(cfg.train.epoch_size / il.batch_size)
    totals: Dict[str, float] = {}
    model.train()
    for u in range(updates):
        n = int(torch.randint(il.size_range[0], il.size_range[1] + 1, (1,)))
        if pool is not None:
            instance, start, idx = pool.sample(il.batch_size, n)
        else:
            instance, start = source.sample(il.batch_size, n, device), None
        loss, stats = il_loss(model, frozen, problem, instance, cfg, start)
        best_cost, best_sol = stats.pop("best_cost"), stats.pop("best_solution")
        if pool is not None:
            pool.update(n, idx, best_sol, best_cost)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
            optimizer.step()
        for k, v in stats.items():
            totals[k] = totals.get(k, 0.0) + v / updates
        if (u + 1) % cfg.train.behavior_refresh == 0:
            frozen.load_state_dict(model.state_dict())
    return totals

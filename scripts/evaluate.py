"""Evaluate a NICO checkpoint on a fixed test set.

    # uniform n=100, 10n sampled moves from the stored node order
    python scripts/evaluate.py --checkpoint checkpoints/nico_tsp.pt \
        --data data/tsp_unif100_test_seed1234.pkl \
        --reference data/results/tsp_unif100_test_seed1234/concorde_costs.pkl

    # 8 random restarts
    python scripts/evaluate.py --checkpoint checkpoints/nico_tsp.pt --data ... --init random --restarts 8

Test data: TSP pickles (list of (n,2) coordinates), CVRP pickles in the Kool et al.
format, or road-network npz files with ``points`` and ``distance`` arrays.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nico.checkpoint import load_checkpoint  # noqa: E402
from nico.evaluate import evaluate, gap_percent  # noqa: E402
from nico.problems import CVRP, TSP, problem_from_config  # noqa: E402
from nico.problems.base import to_device  # noqa: E402
from nico.problems.road import load_road_npz  # noqa: E402


def load_instances(problem_name: str, path: str, num):
    if problem_name == "cvrp" and path.endswith(".npz"):
        return CVRP.load_npz(path, num)
    if path.endswith(".npz"):
        return load_road_npz(path, num)
    if problem_name == "cvrp":
        return CVRP.load_pickle(path, num)
    return TSP.load_pickle(path, num)


def load_initial_solutions(path: str, num):
    """npz/pkl with an integer array of solutions, (B, L) or (B, R, L)."""
    if path.endswith(".npz"):
        with np.load(path) as data:
            key = "solutions" if "solutions" in data else data.files[0]
            arr = data[key]
    else:
        with open(path, "rb") as f:
            arr = pickle.load(f)
    return torch.as_tensor(np.asarray(arr)[:num], dtype=torch.long)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="NICO checkpoint")
    parser.add_argument("--data", required=True)
    parser.add_argument("--num", type=int, default=None, help="evaluate only the first NUM instances")
    parser.add_argument("--reference", default=None, help="pickle/npy of reference costs (e.g. Concorde, HGS)")
    parser.add_argument("--init", default="identity", help="identity | random | path to initial solutions")
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--step-mult", type=float, default=10.0, help="budget = step_mult * n moves")
    parser.add_argument("--time-budget", type=float, default=None, help="seconds per batch")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--greedy", action="store_true", help="argmax moves instead of sampling")
    parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
    parser.add_argument("--tabu", type=int, default=None, help="override the tabu tenure of the checkpoint")
    parser.add_argument("--temperature-end", type=float, default=None,
                        help="anneal the sampling temperature geometrically to this value")
    parser.add_argument("--moves-per-step", type=int, default=1,
                        help="sampled non-overlapping moves applied per policy evaluation (0 = one per 100 nodes)")
    parser.add_argument("--symmetrize", action="store_true",
                        help="average the policy over tour reversal and mirror image (4 evaluations per step)")
    parser.add_argument("--adaptive-temp", type=float, default=0.0,
                        help="temperature x (1 + a * log(1 + moves since last best / n)) (0 = off)")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None, help="write per-instance results to this json file")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, cfg, _ = load_checkpoint(args.checkpoint, device=device)
    problem = problem_from_config(cfg)
    cfg.search.temperature = args.temperature
    cfg.search.temperature_end = args.temperature_end
    cfg.search.moves_per_step = args.moves_per_step
    cfg.search.symmetrize = args.symmetrize
    cfg.search.adaptive_temp = args.adaptive_temp
    if args.tabu is not None:
        cfg.search.tabu_tenure = args.tabu
    instance = to_device(load_instances(cfg.problem, args.data, args.num), device)
    n = instance["cost_matrix"].size(1) - (1 if cfg.problem == "cvrp" else 0)
    init = args.init if args.init in ("identity", "random") else load_initial_solutions(args.init, args.num)
    if cfg.problem == "cvrp" and torch.is_tensor(init):
        init = problem.from_sequences(init)          # depot-separated, zero-padded giant tours
    if torch.is_tensor(init) and init.size(0) < instance["cost_matrix"].size(0):
        # evaluate only the instances that have initial solutions
        instance = {k: v[: init.size(0)] if torch.is_tensor(v) else v for k, v in instance.items()}

    res = evaluate(model, problem, instance, cfg.search, steps=int(args.step_mult * n), init=init,
                   restarts=args.restarts, batch_size=args.batch_size, sample=not args.greedy,
                   time_budget_s=args.time_budget)
    scale = instance.get("objective_scale")
    cost = res["cost"] * scale if scale is not None else res["cost"]
    if cfg.problem == "cvrp":
        assert bool(problem.is_feasible(instance, res["solution"]).all()), "infeasible CVRP solution"
    summary = {"instances": int(cost.numel()), "moves_per_step": args.moves_per_step, "mean_cost": float(cost.mean()),
               "seconds": float(res["seconds"])}
    if args.reference:
        if args.reference.endswith(".npz"):
            with np.load(args.reference) as z:
                ref = torch.as_tensor(z["reference_costs"][: cost.numel()]).float()
        else:
            with open(args.reference, "rb") as f:
                ref = torch.as_tensor(np.asarray(pickle.load(f), dtype=np.float64)[: cost.numel()]).float()
        summary["mean_reference"] = float(ref.mean())
        summary["gap_of_means_percent"] = float(100 * (cost.mean().cpu() / ref.mean() - 1))
        summary["mean_gap_percent"] = float(gap_percent(cost.cpu(), ref).mean())
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps({**summary, "cost": cost.cpu().tolist()}, indent=1))


if __name__ == "__main__":
    main()

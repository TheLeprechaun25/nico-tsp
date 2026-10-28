# NICO: operator-aligned neural improvement for routing

Clean implementation of **NICO** (*Neural Improvement for Combinatorial
Optimization*). The incumbent solution is represented by its **solution edges**
(one token per edge, no positional encodings) with scale-free features; a gain-aware
pairwise decoder scores 2-opt moves from the two edges they remove and their exact
gain; the policy is trained by exact-oracle imitation followed by critic-free
group-based RL.

One shared package, `nico/`, serves every problem:

| Problem | Instances | Config | Checkpoint |
|---|---|---|---|
| Euclidean TSP | uniform in [0,1]² | `tsp/configs/nico_default.yaml` | `checkpoints/nico_tsp.pt` |
| Road-network TSP | RRNCO cities, symmetrized road distance | `tsp/configs/nico_road.yaml` | `checkpoints/nico_road.pt` |
| CVRP | uniform, Kool et al. demands/capacities | `cvrp/configs/nico_cvrp_default.yaml` | `checkpoints/nico_cvrp.pt` |

The road model is fine-tuned from the TSP model; the CVRP model uses two operators
(2-opt and 2-opt*) on the giant tour.

## Install

```bash
pip install -e .            # torch, numpy, pyyaml, tqdm
pip install -e ".[wandb]"   # + Weights & Biases logging
```

## Layout

```
nico/
  problems/   base.py   instance dicts, tour costs, 2-opt move and delta
              tsp.py    symmetric TSP
              cvrp.py   giant tour with depot copies, Proposition-4 feasibility mask
              road.py   RRNCO road-network instances (symmetrized objective)
  features.py  edge-token features (geo, dist, cvrp, history)
  nn.py        attention blocks, cycle mixing
  model.py     NICOPolicy: tokenizer -> encoder -> pairwise decoder
  search.py    search state, tabu masking, warmup, policy-driven improvement
  oracle.py    exact K-step 2-opt oracle with set-valued targets
  train_il.py  stage 1 (imitation)       train_rl.py  stage 2 (group RL + PPO)
  trainer.py   two-stage driver           evaluate.py  batched evaluation with restarts
  checkpoint.py  config-carrying checkpoints
scripts/       train.py, evaluate.py
tsp/, cvrp/    training configs and problem-specific notes
checkpoints/   trained models (weights + config)
data/          test sets with Concorde / HGS references, TSP validation sets with LKH references
```

## Conventions

* An instance batch is a `dict` of tensors. `cost_matrix` (B, V, V) is always the
  objective; `coords`, `demand`, `capacity`, ... are optional inputs. The same code
  therefore handles Euclidean and road-network costs.
* A solution is a (B, L) cyclic sequence of node ids. Token k is the edge
  (sol[k], sol[k+1]); move (i, j), i < j, cuts edges i and j and reverses
  sol[i+1 .. j].
* Every checkpoint stores its full config; loading is strict.

## Train

```bash
python scripts/train.py --config tsp/configs/nico_default.yaml
python scripts/train.py --config cvrp/configs/nico_cvrp_default.yaml train.seed=43 run_name=cvrp_s43
python scripts/train.py --config tsp/configs/nico_road.yaml \
    data.road_root=<rrnco>/data/dataset data.road_split=<rrnco_city_splits.json>
```

Any config field can be overridden as `section.field=value`. Metrics are appended to
`outputs/<run_name>/metrics.jsonl` (add `--wandb-project NAME` for W&B).

## Evaluate

```bash
# TSP: 10n sampled moves from the stored node order, gap to Concorde
python scripts/evaluate.py --checkpoint checkpoints/nico_tsp.pt \
    --data data/tsp_unif100_test_seed1234.pkl \
    --reference data/results/tsp_unif100_test_seed1234/concorde_costs.pkl
# CVRP: refine POMO solutions, gap to HGS
python scripts/evaluate.py --checkpoint checkpoints/nico_cvrp.pt \
    --data data/cvrp/cvrp100_test.npz --init data/cvrp/pomo_test_seed1234_100.npz \
    --reference data/cvrp/hgs_test_seed1234_100.npz
# options: --init identity|random|<solutions.npz>  --restarts 8  --step-mult 10  --num 100
```

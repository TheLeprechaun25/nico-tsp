# NICO for the TSP

Configs for the Euclidean and road-network TSP experiments. The implementation is
the shared package in `../nico` (run the commands from the repository root).

| Config | Model | Data |
|---|---|---|
| `configs/nico_default.yaml` | **default NICO training run** (IL:100 + RL:20, ~12 h) | uniform Euclidean, n = 20-100 |
| `configs/nico_road.yaml` | NICO-Road: RL fine-tuning of the default model (RL:30) | RRNCO cities, n = 20-100 |

## Euclidean TSP

Default run (S2 invariant edge features, gain-aware decoder, IL:100 + RL:20 with late-search starts):

```bash
python scripts/train.py --config tsp/configs/nico_default.yaml
```

Evaluate:

```bash
python scripts/evaluate.py --checkpoint checkpoints/nico_tsp.pt \
    --data data/tsp_unif500_test_seed1234.pkl --num 100 \
    --reference data/results/tsp_unif500_test_seed1234/concorde_costs.pkl
```

By default: 10n sampled moves from the stored node order (`--init identity`);
`--init random --restarts 8` for 8 restarts from random tours.

## Road-network TSP

The RRNCO data (Son et al.) provides, per city, `points` (lat/lon) and directed
`distance` and `duration` matrices. NICO optimizes the symmetrized distance
(d_uv + d_vu)/2, for which 2-opt deltas are exact. NICO-Road keeps the weights and
features of the TSP model, with the edge lengths read from the road costs
(`model.length_source: cost`), and is fine-tuned with RL from `checkpoints/nico_tsp.pt`.

```bash
python scripts/train.py --config tsp/configs/nico_road.yaml \
    data.road_root=<rrnco>/data/dataset data.road_split=<rrnco_city_splits.json>
python scripts/evaluate.py --checkpoint checkpoints/nico_road.pt \
    --data <test_ood_city_n100.npz> --reference <lkh3_ood_city_n100.pkl>
```

Test sets are npz files with `points` (B, n, 2) and `distance` (B, n, n); reported
costs are in the original distance units.

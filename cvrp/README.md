# NICO for the CVRP

Config for the CVRP model (`checkpoints/nico_cvrp.pt`). The implementation is the shared package in
`../nico` (run the commands from the repository root).

## Representation

* **Solution:** a giant tour of length L = n + K, starting at the depot, with K depot
  copies. Consecutive depot copies are empty routes, so routes can be emptied or
  opened while L stays fixed (`data.extra_routes` empty routes are added to the
  initial solution).
* **Tokens:** the L solution edges, with the geometric features of the TSP plus demands
  of both endpoints, depot flags, prefix load P, suffix load Lambda - P and residual
  capacity Q - Lambda. Depot-to-depot edges use a learned null token.
* **Moves:** two operators on the giant tour, each with its own decoder head. 2-opt:
  with both cut edges in one route it is an intra-route 2-opt, otherwise the reversing
  2-opt*. 2-opt*: the exchange of the tails of two routes.
* **Feasibility:** Proposition 4. A move (i, j) is feasible iff both edges are in the
  same route, or P_i + P_j <= Q and (Lambda_i - P_i) + (Lambda_j - P_j) <= Q. It is
  computed as an L x L mask from three per-token scalars
  (`nico/problems/cvrp.py`).
* **Oracle:** exact two-step (K = 2), feasible moves only.

## Train and evaluate

```bash
python scripts/train.py --config cvrp/configs/nico_cvrp_default.yaml
# n = 100: refine POMO solutions, gap to HGS
python scripts/evaluate.py --checkpoint checkpoints/nico_cvrp.pt \
    --data data/cvrp/cvrp100_test.npz --init data/cvrp/pomo_test_seed1234_100.npz \
    --reference data/cvrp/hgs_test_seed1234_100.npz
# n = 200, from the stored order
python scripts/evaluate.py --checkpoint checkpoints/nico_cvrp.pt \
    --data data/cvrp/cvrp200_test_seed1234.npz --num 100 \
    --reference data/cvrp/hgs_cvrp200_test_seed1234_100.npz
```

Test data are npz files with depot, customer coordinates, demands and capacities
(the Kool et al. pickle format, a list of `(depot, customers, demand, capacity)`,
is also accepted).

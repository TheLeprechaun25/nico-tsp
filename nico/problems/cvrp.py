"""Capacitated VRP on a giant tour with depot copies.

A solution is a (B, L) sequence with L = n + K: every customer 1..n appears
once and the depot (node 0) appears K times. ``solution[:, 0]`` is always a
depot. Each depot copy closes one route and opens the next; two consecutive
depot copies form an empty route, so the number of non-empty routes can change
during search while L stays fixed.

The operator is 2-opt on the giant tour. With both cut edges in the same route
it is an intra-route 2-opt; otherwise it exchanges route heads and tails (a
2-opt* move with reversal). Its capacity feasibility is a function of the two
cut edges only (Proposition 4 of the paper): with P the prefix load and
Lambda the load of the edge's route,

    feasible(i, j)  <=>  r(i) == r(j)
                      or (P_i + P_j <= Q  and  (Lambda_i - P_i) + (Lambda_j - P_j) <= Q).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from .base import (
    Instance,
    admissible_pairs_mask,
    apply_two_opt,
    edge_costs,
    euclidean_cost_matrix,
    tour_cost,
    two_opt_delta,
)

# Vehicle capacity by number of customers: Kool et al. (2019) for n <= 100 and
# the values commonly used for larger instances (e.g. LEHD / BQ-NCO) above.
_CAPACITY_TABLE = {10: 20, 20: 30, 50: 40, 100: 50, 200: 80, 500: 100, 1000: 250}
_FEAS_EPS = 1e-6


def capacity_for_size(n: int) -> float:
    """Capacity for ``n`` customers, linearly interpolated between table entries."""
    keys = sorted(_CAPACITY_TABLE)
    if n <= keys[0]:
        return float(_CAPACITY_TABLE[keys[0]])
    if n >= keys[-1]:
        return float(_CAPACITY_TABLE[keys[-1]])
    return float(np.round(np.interp(n, keys, [_CAPACITY_TABLE[k] for k in keys])))


OPERATORS = ("2opt", "relocate", "swap", "2opt*")


class CVRP:
    """Operators (pairwise over solution positions, each with its own policy head):

    2opt      (i, j)  i < j: cut e_i and e_j and reverse sol[i+1..j] (intra-route 2-opt or
                      the reversing 2-opt*)
    relocate  (p, l)  move the customer at position p into edge e_l, l not in {p-1, p}
    swap      (p, q)  p < q, not adjacent: exchange the customers at positions p and q
    2opt*     (i, j)  i < j in different routes: exchange the route tails after e_i and e_j
    """

    name = "cvrp"

    def __init__(self, extra_routes: int = 2, operators=("2opt",)):
        self.extra_routes = int(extra_routes)
        unknown = set(operators) - set(OPERATORS)
        if unknown or not operators or operators[0] != "2opt":
            raise ValueError(f"operators must start with '2opt' and be among {OPERATORS}, got {operators}")
        self.operators = tuple(operators)
        self.n_ops = len(self.operators)

    # --------------------------------------------------------------- instances
    @staticmethod
    def from_arrays(depot: torch.Tensor, customers: torch.Tensor, demand: torch.Tensor,
                    capacity: torch.Tensor) -> Instance:
        """depot (B,2), customers (B,n,2), demand (B,n), capacity (B,) -> instance."""
        coords = torch.cat([depot.unsqueeze(1), customers], dim=1).float()
        demand = torch.cat([torch.zeros_like(demand[:, :1]), demand], dim=1).float()
        return {
            "coords": coords,
            "demand": demand,
            "capacity": capacity.float().view(-1),
            "cost_matrix": euclidean_cost_matrix(coords),
        }

    @staticmethod
    def load_pickle(path, num: Optional[int] = None) -> Instance:
        """Kool et al. format: list of (depot (2,), customers (n,2), demand (n,), capacity)."""
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)[:num]
        depot = torch.tensor(np.array([d[0] for d in data]), dtype=torch.float32)
        customers = torch.tensor(np.array([d[1] for d in data]), dtype=torch.float32)
        demand = torch.tensor(np.array([d[2] for d in data]), dtype=torch.float32)
        capacity = torch.tensor([float(d[3]) for d in data])
        return CVRP.from_arrays(depot, customers, demand, capacity)

    @staticmethod
    def load_npz(path, num: Optional[int] = None) -> Instance:
        """npz with coords (B, n+1, 2) (depot first), demand (B, n+1) and capacity (B,)."""
        with np.load(path) as z:
            coords, demand, cap = z["coords"][:num], z["demand"][:num], z["capacity"][:num]
        return CVRP.from_arrays(torch.tensor(coords[:, 0]), torch.tensor(coords[:, 1:]),
                                torch.tensor(demand[:, 1:]), torch.tensor(cap))

    def from_sequences(self, raw, device=None) -> torch.Tensor:
        """Giant tours from depot-separated sequences padded with zeros (e.g. POMO output),
        re-padded to the number of routes plus ``extra_routes`` depot copies."""
        routes = []
        for row in np.asarray(raw):
            rs, cur = [], []
            for x in list(row) + [0]:
                if x == 0:
                    if cur:
                        rs.append(cur)
                    cur = []
                else:
                    cur.append(int(x))
            routes.append(rs)
        return self.from_routes(routes, max(len(r) for r in routes) + self.extra_routes, device)

    def sample_instances(self, batch_size: int, size: int, device, generator=None,
                         capacity_mult: float = 1.0) -> Instance:
        """Uniform CVRP (Kool et al., 2019): n customers, demands U{1..9}; capacity scaled by capacity_mult."""
        depot = torch.rand(batch_size, 2, device=device, generator=generator)
        customers = torch.rand(batch_size, size, 2, device=device, generator=generator)
        demand = torch.randint(1, 10, (batch_size, size), device=device, generator=generator)
        capacity = torch.full((batch_size,), float(round(capacity_for_size(size) * capacity_mult)), device=device)
        return self.from_arrays(depot, customers, demand.float(), capacity)

    # --------------------------------------------------------------- solutions
    @staticmethod
    def num_customers(instance: Instance) -> int:
        return instance["cost_matrix"].size(1) - 1

    @torch.no_grad()
    def split_permutations(self, instance: Instance, perm: torch.Tensor,
                           num_routes: Optional[int] = None) -> torch.Tensor:
        """Greedy capacity split of customer permutations (B, n) into giant tours.

        Customers are appended to the current route until the next one would
        exceed the capacity; then a new route starts. The result is padded with
        empty routes to ``num_routes`` depots (default: the largest number of
        routes in the batch plus ``extra_routes``).
        """
        demand, Q = instance["demand"], instance["capacity"]
        B, n = perm.shape
        device = perm.device
        load = torch.zeros(B, device=device)
        rows = [torch.zeros(B, dtype=torch.long, device=device)]  # leading depot
        is_break = torch.zeros(B, n, dtype=torch.bool, device=device)
        for k in range(n):
            q = demand.gather(1, perm[:, k:k + 1]).squeeze(1)
            brk = load + q > Q + _FEAS_EPS
            is_break[:, k] = brk
            load = torch.where(brk, q, load + q)
        seqs, lengths = [], []
        for b in range(B):
            out = [0]
            for k in range(n):
                if is_break[b, k]:
                    out.append(0)
                out.append(int(perm[b, k]))
            seqs.append(out)
            lengths.append(len(out))
        routes = [s.count(0) for s in seqs]
        K = num_routes if num_routes is not None else max(routes) + self.extra_routes
        if K < max(routes):
            raise ValueError(f"num_routes={K} is smaller than the {max(routes)} routes needed")
        L = n + K
        out = torch.zeros(B, L, dtype=torch.long, device=device)
        for b, s in enumerate(seqs):
            out[b, : len(s)] = torch.as_tensor(s, device=device)
        return out

    def random_solutions(self, instance: Instance, generator=None, num_routes=None) -> torch.Tensor:
        B = instance["demand"].size(0)
        n = self.num_customers(instance)
        noise = torch.rand(B, n, device=instance["demand"].device, generator=generator)
        return self.split_permutations(instance, noise.argsort(dim=1) + 1, num_routes)

    def sweep_permutations(self, instance: Instance, generator=None) -> torch.Tensor:
        """Customers ordered by polar angle around the depot, from a random start angle."""
        coords = instance["coords"]
        rel = coords[:, 1:] - coords[:, :1]
        angle = torch.atan2(rel[..., 1], rel[..., 0])
        offset = torch.rand(coords.size(0), 1, device=coords.device, generator=generator) * 2 * np.pi
        return torch.remainder(angle - offset, 2 * np.pi).argsort(dim=1) + 1

    def mixed_solutions(self, instance: Instance, sweep_frac: float, generator=None,
                        num_routes=None) -> torch.Tensor:
        """Random-permutation split for a (1 - sweep_frac) share of the rows, sweep split otherwise."""
        B = instance["demand"].size(0)
        n = self.num_customers(instance)
        device = instance["demand"].device
        perm = torch.rand(B, n, device=device, generator=generator).argsort(dim=1) + 1
        use_sweep = torch.rand(B, device=device, generator=generator) < sweep_frac
        if bool(use_sweep.any()):
            perm[use_sweep] = self.sweep_permutations(instance, generator)[use_sweep]
        return self.split_permutations(instance, perm, num_routes)

    def identity_solutions(self, instance: Instance, num_routes=None) -> torch.Tensor:
        B = instance["demand"].size(0)
        n = self.num_customers(instance)
        perm = torch.arange(1, n + 1, device=instance["demand"].device).expand(B, n)
        return self.split_permutations(instance, perm, num_routes)

    def savings_solutions(self, instance: Instance, num_routes=None) -> torch.Tensor:
        """Clarke-Wright parallel savings (Clarke & Wright, 1964), deterministic: every customer starts
        on its own route; pairs are merged in decreasing order of s_ij = d_0i + d_0j - d_ij > 0 when i
        and j end two different routes and the merged load fits the capacity."""
        C = instance["cost_matrix"].double().cpu().numpy()
        demand = instance["demand"].double().cpu().numpy()
        Q = instance["capacity"].double().cpu().numpy()
        out = []
        for b in range(C.shape[0]):
            d, n = C[b], C.shape[1] - 1
            iu, ju = np.triu_indices(n, 1)
            iu, ju = iu + 1, ju + 1
            s = d[0, iu] + d[0, ju] - d[iu, ju]
            order = np.argsort(-s, kind="stable")
            routes = {c: [c] for c in range(1, n + 1)}       # route id -> customers
            owner = np.arange(n + 1)                            # customer -> route id
            load = {c: demand[b, c] for c in range(1, n + 1)}
            for k in order:
                if s[k] <= 0:
                    break
                i, j = int(iu[k]), int(ju[k])
                ri, rj = owner[i], owner[j]
                if ri == rj or load[ri] + load[rj] > Q[b] + _FEAS_EPS:
                    continue
                a, c = routes[ri], routes[rj]
                if a[-1] != i:                                  # orient: i at the end of a
                    if a[0] != i:
                        continue
                    a = a[::-1]
                if c[0] != j:                                   # and j at the start of c
                    if c[-1] != j:
                        continue
                    c = c[::-1]
                routes[ri] = a + c
                load[ri] += load.pop(rj)
                for x in routes.pop(rj):
                    owner[x] = ri
            out.append(list(routes.values()))
        K = num_routes if num_routes is not None else max(len(r) for r in out) + self.extra_routes
        return self.from_routes(out, K, instance["demand"].device)

    @staticmethod
    def from_routes(routes_per_instance, num_routes: int, device=None) -> torch.Tensor:
        """Build giant tours from Python lists of routes (lists of customer ids)."""
        seqs = []
        for routes in routes_per_instance:
            seq = []
            for r in routes:
                seq += [0] + [int(c) for c in r]
            seqs.append(seq)
        n = max(len(s) - s.count(0) for s in seqs)
        L = n + num_routes
        out = torch.zeros(len(seqs), L, dtype=torch.long, device=device)
        for b, s in enumerate(seqs):
            out[b, : len(s)] = torch.as_tensor(s)
        return out

    # ------------------------------------------------------------- route data
    @staticmethod
    def route_info(instance: Instance, solution: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Per solution edge e_k = (sol[k], sol[k+1]):

        route   (B, L) long  route of the edge (an edge leaving a depot belongs to the
                             route it opens; an edge entering a depot to the route it closes)
        prefix  (B, L) float P_k, demand served by the route up to and including sol[k]
        load    (B, L) float Lambda of the edge's route
        """
        demand = instance["demand"]
        is_depot = solution == 0
        route = is_depot.long().cumsum(dim=1) - 1
        dem = demand.gather(1, solution)
        cum = dem.cumsum(dim=1)
        start = torch.where(is_depot, cum, torch.zeros_like(cum)).cummax(dim=1).values
        prefix = cum - start
        K = int(is_depot.sum(dim=1).max())
        loads = torch.zeros(solution.size(0), K, device=solution.device, dtype=dem.dtype)
        loads.scatter_add_(1, route, dem)
        return {"route": route, "prefix": prefix, "load": loads.gather(1, route)}

    # ------------------------------------------------------------- operations
    @staticmethod
    def cost(instance: Instance, solution: torch.Tensor) -> torch.Tensor:
        return tour_cost(instance["cost_matrix"], solution)

    def apply_move(self, instance: Instance, solution: torch.Tensor, moves: torch.Tensor) -> torch.Tensor:
        """Apply one move per row; ``moves`` (B, 2) = (op * L + i, j)."""
        B, L = solution.shape
        op, i, j = moves[:, 0] // L, moves[:, 0] % L, moves[:, 1]
        if self.n_ops == 1:
            return apply_two_opt(solution, torch.stack([i, j], 1))
        out = solution.clone()
        for o, name in enumerate(self.operators):
            rows = (op == o).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            sol, a, b = solution[rows], i[rows], j[rows]
            if name == "2opt":
                out[rows] = apply_two_opt(sol, torch.stack([a, b], 1))
            else:
                out[rows] = sol.gather(1, _position_map(name, sol, a, b))
        return out

    def invalid_moves(self, instance: Instance, solution: torch.Tensor) -> torch.Tensor:
        """(B, L, L) bool for 2-opt only, else (B, O, L, L): inadmissible or capacity-infeasible."""
        masks = [self._invalid(name, instance, solution) for name in self.operators]
        return masks[0] if self.n_ops == 1 else torch.stack(masks, 1)

    def _invalid(self, name: str, instance: Instance, solution: torch.Tensor) -> torch.Tensor:
        B, L = solution.shape
        info = self.route_info(instance, solution)
        r, P, Lam = info["route"], info["prefix"], info["load"]
        Q = instance["capacity"].view(B, 1, 1) + _FEAS_EPS
        diff = r.unsqueeze(2) != r.unsqueeze(1)
        ar = torch.arange(L, device=solution.device)
        customer = solution != 0
        if name == "2opt":   # Proposition 4
            head = P.unsqueeze(2) + P.unsqueeze(1)
            tail = (Lam - P).unsqueeze(2) + (Lam - P).unsqueeze(1)
            base = admissible_pairs_mask(L, solution.device).expand(B, L, L)
            return base | (diff & ((head > Q) | (tail > Q)))
        if name == "2opt*":
            Pi, Pj, Ti, Tj = P.unsqueeze(2), P.unsqueeze(1), (Lam - P).unsqueeze(2), (Lam - P).unsqueeze(1)
            upper = (ar.view(L, 1) >= ar.view(1, L)).expand(B, L, L)
            return upper | ~diff | (Pi + Tj > Q) | (Pj + Ti > Q)
        q = instance["demand"].gather(1, solution)
        if name == "relocate":
            # customer at p; the route of the receiving edge e_l must absorb it
            touch = ((ar.view(1, L) - ar.view(L, 1)) % L >= L - 1) | (ar.view(1, L) == ar.view(L, 1))
            cap = diff & (Lam.unsqueeze(1) + q.unsqueeze(2) > Q)
            return touch.expand(B, L, L) | cap | ~customer.unsqueeze(2)
        if name == "swap":
            gap = (ar.view(1, L) - ar.view(L, 1)) % L
            bad = (ar.view(L, 1) >= ar.view(1, L)) | (gap < 2) | (gap > L - 2)
            qp, qq = q.unsqueeze(2), q.unsqueeze(1)
            cap = diff & ((Lam.unsqueeze(2) - qp + qq > Q) | (Lam.unsqueeze(1) - qq + qp > Q))
            return bad.expand(B, L, L) | cap | ~customer.unsqueeze(2) | ~customer.unsqueeze(1)
        raise ValueError(name)

    def move_values(self, instance: Instance, solution: torch.Tensor) -> torch.Tensor:
        """(B, O * L * L) exact cost change of every move, +inf where invalid."""
        B, L = solution.shape
        C = instance["cost_matrix"]
        invalid = self.invalid_moves(instance, solution).view(B, self.n_ops, L, L)
        vals = []
        a, nxt, prv = solution, torch.roll(solution, -1, 1), torch.roll(solution, 1, 1)
        E = lambda x, y: edge_costs(C, x, y)
        for o, name in enumerate(self.operators):
            if name == "2opt":
                ar = torch.arange(L, device=solution.device)
                pairs = torch.stack(torch.meshgrid(ar, ar, indexing="ij"), -1).reshape(-1, 2)
                d = two_opt_delta(C, solution, pairs).view(B, L, L)
            elif name == "2opt*":
                ai, bi = a.unsqueeze(2).expand(B, L, L), nxt.unsqueeze(2).expand(B, L, L)
                aj, bj = a.unsqueeze(1).expand(B, L, L), nxt.unsqueeze(1).expand(B, L, L)
                d = E(ai, bj) + E(aj, bi) - E(ai, bi) - E(aj, bj)
            elif name == "relocate":
                x = a.unsqueeze(2).expand(B, L, L)
                px, nx = prv.unsqueeze(2).expand(B, L, L), nxt.unsqueeze(2).expand(B, L, L)
                u, v = a.unsqueeze(1).expand(B, L, L), nxt.unsqueeze(1).expand(B, L, L)
                d = E(px, nx) - E(px, x) - E(x, nx) + E(u, x) + E(x, v) - E(u, v)
            elif name == "swap":
                x, y = a.unsqueeze(2).expand(B, L, L), a.unsqueeze(1).expand(B, L, L)
                px, nx = prv.unsqueeze(2).expand(B, L, L), nxt.unsqueeze(2).expand(B, L, L)
                py, ny = prv.unsqueeze(1).expand(B, L, L), nxt.unsqueeze(1).expand(B, L, L)
                d = (E(px, y) + E(y, nx) + E(py, x) + E(x, ny)
                     - E(px, x) - E(x, nx) - E(py, y) - E(y, ny))
            vals.append(d.masked_fill(invalid[:, o], float("inf")))
        return torch.stack(vals, 1).reshape(B, -1)

    @staticmethod
    def is_feasible(instance: Instance, solution: torch.Tensor) -> torch.Tensor:
        """Every customer exactly once, sequence starts at the depot, loads within capacity."""
        B, V, _ = instance["cost_matrix"].shape
        customers = torch.where(solution == 0, torch.full_like(solution, V), solution)
        visited = customers.sort(dim=1).values[:, : V - 1]
        once = (visited == torch.arange(1, V, device=solution.device).expand(B, V - 1)).all(dim=1)
        info = CVRP.route_info(instance, solution)
        within = (info["load"] <= instance["capacity"].view(B, 1) + _FEAS_EPS).all(dim=1)
        return once & within & (solution[:, 0] == 0)


def _position_map(name: str, solution: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(B, L) gather index of the sequence after a relocate / swap / 2-opt* move."""
    B, L = solution.shape
    t = torch.arange(L, device=solution.device).view(1, L).expand(B, L)
    a, b = a.view(B, 1), b.view(B, 1)
    if name == "swap":
        return torch.where(t == a, b, torch.where(t == b, a, t))
    if name == "relocate":   # element at p = a moves into edge e_l, l = b
        fwd = (b > a) & (t >= a) & (t < b)               # l > p: shift left
        bwd = (b < a) & (t > b + 1) & (t <= a)           # l < p: shift right
        idx = torch.where(fwd, t + 1, torch.where(bwd, t - 1, t))
        idx = torch.where((b > a) & (t == b), a.expand(B, L), idx)
        idx = torch.where((b < a) & (t == b + 1), a.expand(B, L), idx)
        return idx
    if name == "2opt*":      # [.. i | X = i+1..eA | Y = eA+1..j | Z = j+1..eB | ..] -> [.. i | Z | Y | X | ..]
        is_depot = solution == 0
        pos = torch.where(is_depot, t, torch.full_like(t, L))
        nxt_depot = torch.flip(torch.cummin(torch.flip(pos, [1]), 1).values, [1])   # first depot at >= t
        nd = torch.cat([nxt_depot[:, 1:], torch.full_like(nxt_depot[:, :1], L)], 1)  # first depot at > t
        eA = nd.gather(1, a) - 1
        eB = nd.gather(1, b) - 1
        x0, x1 = a + 1, eA            # X = [x0, x1]
        y0, y1 = eA + 1, b            # Y
        z0, z1 = b + 1, eB            # Z
        lz, ly = z1 - z0 + 1, y1 - y0 + 1
        s0 = a + 1
        idx = t
        inZ = (t >= s0) & (t < s0 + lz)
        inY = (t >= s0 + lz) & (t < s0 + lz + ly)
        inX = (t >= s0 + lz + ly) & (t <= eB)
        idx = torch.where(inZ, z0 + (t - s0), idx)
        idx = torch.where(inY, y0 + (t - s0 - lz), idx)
        idx = torch.where(inX, x0 + (t - s0 - lz - ly), idx)
        return idx
    raise ValueError(name)

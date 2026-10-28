"""Training instance sources."""

from __future__ import annotations

import torch

from .config import Config
from .problems.base import Instance


class InstanceSource:
    """Samples training instances of a given size: uniform Euclidean or road network."""

    def __init__(self, cfg: Config, problem):
        self.problem = problem
        self.road = None
        self.crop_frac = cfg.data.crop_frac
        self.crop_parent_range = tuple(cfg.data.crop_parent_range)
        self.cap_mult = tuple(float(x) for x in cfg.data.cvrp_capacity_mult)
        if self.cap_mult != (1.0, 1.0) and problem.name != "cvrp":
            raise ValueError("data.cvrp_capacity_mult is implemented for the CVRP")
        if self.crop_frac > 0 and problem.name != "tsp":
            raise ValueError("data.crop_frac is implemented for the TSP")
        if cfg.data.source == "road":
            if problem.name != "tsp":
                raise ValueError("Road-network instances are implemented for the TSP")
            from .problems.road import RoadCitySampler

            if not (cfg.data.road_root and cfg.data.road_split):
                raise ValueError("data.road_root and data.road_split are required for road data")
            self.road = RoadCitySampler(cfg.data.road_root, cfg.data.road_split,
                                        cfg.data.road_city_key, seed=cfg.train.seed)
        elif cfg.data.source != "uniform":
            raise ValueError(f"Unknown data.source {cfg.data.source!r}")

    @torch.no_grad()
    def sample(self, batch_size: int, size: int, device) -> Instance:
        if self.road is not None:
            return self.road.sample(batch_size, size, device)
        if self.crop_frac > 0 and float(torch.rand(())) < self.crop_frac:
            return self.sample_crops(batch_size, size, device)
        if self.cap_mult != (1.0, 1.0):
            import math
            lo, hi = (math.log(x) for x in self.cap_mult)
            m = math.exp(lo + (hi - lo) * float(torch.rand(())))
            return self.problem.sample_instances(batch_size, size, device, capacity_mult=m)
        return self.problem.sample_instances(batch_size, size, device)

    @torch.no_grad()
    def sample_crops(self, batch_size: int, size: int, device) -> Instance:
        """`size` nearest nodes to a random centre of uniform instances with N ~ U[crop_parent_range] nodes."""
        lo, hi = self.crop_parent_range
        N = int(torch.randint(max(lo, size), max(hi, size) + 1, (1,)))
        parent = torch.rand(batch_size, N, 2, device=device)
        centre = parent[torch.arange(batch_size, device=device), torch.randint(0, N, (batch_size,), device=device)]
        d = (parent - centre.unsqueeze(1)).norm(dim=-1)
        idx = d.topk(size, dim=1, largest=False).indices
        coords = parent.gather(1, idx.unsqueeze(-1).expand(batch_size, size, 2))
        return self.problem.from_coords(coords)

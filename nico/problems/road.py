"""Road-network TSP instances from the RRNCO city data (Son et al.).

Each city file ``<root>/<city>/<city>_data.npz`` (or ``<root>/<city>_data.npz``)
holds ``points`` (lat/lon, (P, 2)), and directed ``distance`` and ``duration``
matrices (P, P). An instance samples ``n`` locations of one city.

The objective is the symmetrized road distance (d_uv + d_vu) / 2, divided by
its off-diagonal mean so that all instances live on a comparable scale;
``objective_scale`` converts costs back to the original units. 2-opt deltas are
exact because the objective is symmetric (see the paper, Section 7.5).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .base import Instance


def normalize_points(points: torch.Tensor) -> torch.Tensor:
    """Per-instance, per-axis min-max scaling to [0, 1]."""
    lo = points.amin(dim=1, keepdim=True)
    hi = points.amax(dim=1, keepdim=True)
    return (points - lo) / (hi - lo).clamp_min(1e-8)


def offdiag_mean(matrix: torch.Tensor) -> torch.Tensor:
    n = matrix.size(-1)
    return matrix.sum(dim=(-2, -1)) / float(n * (n - 1))


def road_instance(points: torch.Tensor, distance: torch.Tensor) -> Instance:
    """points (B, n, 2) lat/lon, distance (B, n, n) directed road distance -> instance."""
    sym = 0.5 * (distance + distance.transpose(1, 2))
    sym.diagonal(dim1=1, dim2=2).zero_()
    scale = offdiag_mean(sym).clamp_min(1e-8)
    return {
        "coords": normalize_points(points.float()),
        "cost_matrix": (sym / scale[:, None, None]).float(),
        "objective_scale": scale.float(),
    }


def _city_path(root: Path, city: str) -> Path:
    for path in (root / city / f"{city}_data.npz", root / f"{city}_data.npz"):
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing RRNCO data for city {city!r} under {root}")


class RoadCitySampler:
    """Samples road-network TSP instances uniformly over the cities of one split.

    ``split_json`` maps split names (e.g. "train", "test") to lists of city names.
    All randomness comes from the sampler's own CPU generator, so a run is
    reproducible independently of the global RNG.
    """

    def __init__(self, root: str | Path, split_json: str | Path, city_key: str = "train", seed: int = 42):
        self.root = Path(root).expanduser()
        split = json.loads(Path(split_json).expanduser().read_text())
        self.cities = tuple(split[city_key])
        if not self.cities:
            raise ValueError(f"Split {city_key!r} of {split_json} is empty")
        self.generator = torch.Generator().manual_seed(int(seed))
        self._cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _load(self, city: str):
        if city not in self._cache:
            with np.load(_city_path(self.root, city), allow_pickle=False) as data:
                points = torch.from_numpy(np.asarray(data["points"], dtype=np.float32))
                distance = torch.from_numpy(np.asarray(data["distance"], dtype=np.float32))
            if not (torch.isfinite(points).all() and torch.isfinite(distance).all()):
                raise ValueError(f"{city}: non-finite values")
            self._cache[city] = (points, distance)
        return self._cache[city]

    @torch.no_grad()
    def sample(self, batch_size: int, size: int, device) -> Instance:
        cities = torch.randint(len(self.cities), (batch_size,), generator=self.generator)
        pts, dist = [], []
        for c in cities.tolist():
            points, distance = self._load(self.cities[c])
            if len(points) < size:
                raise ValueError(f"{self.cities[c]} has {len(points)} locations, need {size}")
            sel = torch.randperm(len(points), generator=self.generator)[:size]
            pts.append(points[sel])
            dist.append(distance[sel][:, sel])
        inst = road_instance(torch.stack(pts), torch.stack(dist))
        return {k: v.to(device) for k, v in inst.items()}


def load_road_npz(path: str | Path, num: Optional[int] = None) -> Instance:
    """Load a fixed road test set saved as npz with ``points`` (B,n,2) and ``distance`` (B,n,n)."""
    with np.load(Path(path).expanduser(), allow_pickle=False) as data:
        points = torch.from_numpy(np.asarray(data["points"][:num], dtype=np.float32))
        distance = torch.from_numpy(np.asarray(data["distance"][:num], dtype=np.float32))
    return road_instance(points, distance)

import math
import os

import numpy as np
from torch.utils.data import Dataset
import torch
import pickle
from utils import find_existing_path, iter_data_path_candidates


class TSP(object):
    def __init__(self, opts):
        self.opts = opts
        self.size = None  # the number of nodes in tsp
        self.graph_type = None  # 'unif' or 'tsplib'
        self._edge_pair_mask_cache = {}  # (device.type, device.index, N) -> (N,N) bool
        self._init_pool_cache = {}  # path -> np.ndarray (M,Kpool,N)

    @torch.no_grad()
    def step_2opt_edge_idx(self, rec: torch.Tensor, ij: torch.Tensor) -> torch.Tensor:
        """
        rec: (B,N) permutation (position -> node_id)
        ij:  (B,2) edge indices i,j in [0..N-1], interpreted as cut edges:
             (i,i+1) and (j,j+1) on the tour positions.
        Assumes i<j and non-adjacent (mask should enforce).
        Applies: reverse rec[i+1 : j+1].
        """
        B, N = rec.shape
        device = rec.device
        ij = ij.long()

        i = ij[:, 0]
        j = ij[:, 1]

        lo = (i + 1).view(B, 1)   # start of reversed segment
        hi = j.view(B, 1)         # end of reversed segment

        idx = torch.arange(N, device=device).view(1, N).expand(B, N)
        in_seg = (idx >= lo) & (idx <= hi)
        idx2 = torch.where(in_seg, lo + hi - idx, idx)
        return rec.gather(1, idx2)

    def step(self, rec, exchange):
        return self.step_2opt_edge_idx(rec, exchange)

    @staticmethod
    def get_costs(coords, rec):
        """
        :param coords: (batch_size, graph_size, 2) coordinates
        :param rec: (batch_size, graph_size) permutations representing tours
        :return: (batch_size) lengths of tours
        """
        # Gather coords in order of tour
        d = coords.gather(1, rec.long().unsqueeze(-1).expand_as(coords))
        length = (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1)

        return length

    @staticmethod
    def build_even_rotation_angles(num_rotations: int, device, dtype) -> torch.Tensor:
        num_rotations = int(num_rotations)
        if num_rotations <= 0:
            raise ValueError(f"num_rotations must be > 0, got {num_rotations}")
        step = (2.0 * math.pi) / float(num_rotations)
        return torch.arange(num_rotations, device=device, dtype=dtype) * step

    @staticmethod
    def rotate_coords_by_angles(
        coords: torch.Tensor,
        angles: torch.Tensor,
        center: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        coords:  (B, N, 2)
        angles:  (R,)
        center:  optional (B, 1, 2) or (B, 2)
        returns: (B, R, N, 2)
        """
        if coords.dim() != 3 or coords.size(-1) != 2:
            raise ValueError(f"Expected coords with shape (B, N, 2), got {tuple(coords.shape)}")
        if angles.dim() != 1:
            raise ValueError(f"Expected angles with shape (R,), got {tuple(angles.shape)}")

        if center is None:
            center = coords.mean(dim=1, keepdim=True)
        elif center.dim() == 2:
            center = center.unsqueeze(1)
        elif center.dim() != 3:
            raise ValueError(f"Expected center with shape (B, 2) or (B, 1, 2), got {tuple(center.shape)}")

        centered = coords.unsqueeze(1) - center.unsqueeze(1)
        cos = torch.cos(angles).view(1, -1, 1, 1)
        sin = torch.sin(angles).view(1, -1, 1, 1)

        x = centered[..., 0:1]
        y = centered[..., 1:2]
        x_rot = cos * x - sin * y
        y_rot = sin * x + cos * y
        rotated = torch.cat([x_rot, y_rot], dim=-1)
        return rotated + center.unsqueeze(1)

    @staticmethod
    def rotate_coords_evenly(coords: torch.Tensor, num_rotations: int, center: torch.Tensor = None) -> torch.Tensor:
        angles = TSP.build_even_rotation_angles(num_rotations, device=coords.device, dtype=coords.dtype)
        return TSP.rotate_coords_by_angles(coords, angles, center=center)

    @staticmethod
    def get_node_adjacency(rec: torch.Tensor) -> torch.Tensor:
        """
        rec: (B, N) permutation (tour) expressed as node IDs.
        returns adj: (B, N, N) bool, True if nodes (u,v) are consecutive on the tour.
        """
        B, N = rec.size()
        device = rec.device
        rec = rec.long()
        # rec: (B, N)
        # next node along the tour (wrap-around)
        next_rec = torch.roll(rec, shifts=-1, dims=1)  # (B, N)

        adj = torch.zeros(B, N, N, dtype=torch.bool, device=device)

        # Flatten for scatter
        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N).reshape(-1)
        u_flat = rec.reshape(-1)
        v_flat = next_rec.reshape(-1)

        # Mark edges (u,v) and (v,u) as adjacent
        adj[b_idx, u_flat, v_flat] = True
        adj[b_idx, v_flat, u_flat] = True

        return adj  # (B, N, N)

    def get_2opt_edge_pair_mask(self, rec: torch.Tensor) -> torch.Tensor:
        """
        Mask over edge indices (i,j), both in [0..N-1].
        True = invalid.
        """
        B, N = rec.shape
        device = rec.device
        key = (device.type, device.index, N)
        base = self._edge_pair_mask_cache.get(key)
        if base is None:
            base = torch.zeros(N, N, dtype=torch.bool, device=device)
            ar = torch.arange(N, device=device)

            # i == j
            base[ar, ar] = True

            # adjacent edges on the cycle: j = i+1 and j = i-1 (wrap)
            base[ar, (ar + 1) % N] = True
            base[ar, (ar - 1) % N] = True

            # enforce i < j (upper triangle only)
            ii = ar.view(N, 1)
            jj = ar.view(1, N)
            base |= (ii >= jj)

            self._edge_pair_mask_cache[key] = base

        return base.unsqueeze(0).expand(B, -1, -1)

    def _load_init_pool(self, load_path: str) -> np.ndarray:
        """
        Loads and caches init solutions.
        Accepts:
          - payload["first_k_solutions"]: list len M, each (K,N)
          - payload["first_solutions"]:   list len M, each (N,)
          - POMO pickle: payload["batches"][0]["first_sample_best_tours"] with shape (M,N)
          - LEHD trace: .npz file with payload["improved_tours"] with shape (M,K,N)
            where slot 0 is the initial constructive tour and later slots are
            accepted improvements, if any.
        Returns: pool_np (M, Kpool, N) int64
        """
        load_path = os.path.expanduser(load_path)
        if load_path in self._init_pool_cache:
            return self._init_pool_cache[load_path]

        suffix = os.path.splitext(load_path)[1].lower()

        if suffix == ".npz":
            with np.load(load_path, allow_pickle=True) as payload:
                if "improved_tours" not in payload:
                    raise KeyError(f"No improved_tours in LEHD trace {load_path}")

                pool = np.asarray(payload["improved_tours"], dtype=np.int64)
                if pool.ndim == 2:
                    pool = pool[:, None, :]
                elif pool.ndim != 3:
                    raise ValueError(f"Expected improved_tours to have 2 or 3 dims, got shape {pool.shape}")

                M, Kpool, _ = pool.shape
                if "improvement_counts" in payload:
                    counts = np.asarray(payload["improvement_counts"], dtype=np.int64).reshape(-1)
                    if counts.shape[0] != M:
                        raise ValueError(
                            f"improvement_counts length {counts.shape[0]} does not match improved_tours M={M}"
                        )
                    counts = np.clip(counts, 1, Kpool)
                    for m in range(M):
                        valid = int(counts[m])
                        pool[m, valid:, :] = pool[m, 0:1, :]
                else:
                    invalid = np.any(pool < 0, axis=-1)
                    if bool(invalid.any()):
                        for m in range(M):
                            valid_idx = np.flatnonzero(~invalid[m])
                            if valid_idx.size == 0:
                                raise ValueError(f"Instance {m} in {load_path} has no valid LEHD tours")
                            pool[m, invalid[m], :] = pool[m, valid_idx[0], :]
        else:
            with open(load_path, "rb") as f:
                payload = pickle.load(f)

            if isinstance(payload, dict) and payload.get("first_k_solutions", None) is not None:
                lst = payload["first_k_solutions"]
                pool = np.stack([np.asarray(x, dtype=np.int64) for x in lst], axis=0)  # (M,K,N)

            elif isinstance(payload, dict) and payload.get("first_solutions", None) is not None:
                lst = payload["first_solutions"]
                pool = np.stack([np.asarray(x, dtype=np.int64)[None, :] for x in lst], axis=0)  # (M,1,N)

            elif isinstance(payload, dict) and payload.get("batches", None) is not None:
                batches = payload["batches"]
                if len(batches) == 0 or "first_sample_best_tours" not in batches[0]:
                    raise KeyError(f"No first_sample_best_tours in POMO payload {load_path}")
                tours = np.asarray(batches[0]["first_sample_best_tours"], dtype=np.int64)
                if tours.ndim != 2:
                    raise ValueError(
                        f"Expected first_sample_best_tours to have shape (M,N), got {tours.shape}"
                    )
                pool = tours[:, None, :]

            else:
                raise KeyError(
                    f"Unsupported init file format for {load_path}. "
                    f"Expected generic first_solutions/first_k_solutions, POMO batches, or LEHD improved_tours."
                )

        self._init_pool_cache[load_path] = pool
        return pool

    def get_initial_solutions(self, batch_size, init_method="sequential", load_path=None, restarts=1, batch_offset=0, device=None):
        assert self.size is not None, "Set problem.size before calling get_initial_solutions"
        N = int(self.size)
        device = device if device is not None else torch.device("cpu")
        method = str(init_method).lower()
        if method == "random":
            sol = []
            for _ in range(restarts):
                noise = torch.rand((batch_size, N), device=device)
                sol.append(torch.argsort(noise, dim=-1).long())

        elif method == "sequential":
            assert restarts == 1, "Use random or load init for multiple inits"
            sol = [torch.arange(N, device=device).view(1, N).expand(batch_size, N).clone().long()]

        elif method == "load":
            assert load_path is not None, "load_path must be specified for init_method='load'"
            pool = self._load_init_pool(load_path)  # (M,Kpool,N)
            M, Kpool, Npool = pool.shape
            if Npool != N:
                raise ValueError(f"Loaded pool has N={Npool}, but problem.size={N}")
            if not (0 <= batch_offset and batch_offset + batch_size <= M):
                raise ValueError(f"Batch slice [{batch_offset}:{batch_offset+batch_size}] out of range for pool M={M}")

            sol = []
            for k in range(restarts):
                kk = k if k < Kpool else 0
                arr = pool[batch_offset:batch_offset + batch_size, kk, :]  # (B,N)
                sol.append(torch.as_tensor(arr, device=device, dtype=torch.long))

        else:
            raise ValueError(f"Unknown init_method: {method}")

        return sol

    @staticmethod
    def make_dataset(opts, graph_type, graph_size, num_samples=100, test=False):
        return TSPDataset(opts=opts, graph_type=graph_type, graph_size=graph_size, num_samples=num_samples, test=test)

    @torch.no_grad()
    def sample_batch(self, batch_size: int, graph_type: str, graph_size: int, device) -> dict:
        """
        Returns a single batch dict on CPU: coords, x_node, e_edge, dists.
        """
        assert graph_type == "unif", "training sampler currently supports only unif"
        coords = torch.rand(batch_size, graph_size, 2, dtype=torch.float32, device=device)
        return {"coords": coords}


class TSPDataset(Dataset):
    def __init__(self, opts, graph_type, graph_size, num_samples=100, test=False):
        super().__init__()
        self.graph_type = graph_type
        self.graph_size = graph_size
        self.test = test
        self.opts = opts

        if test:
            try:
                data_path = find_existing_path(
                    iter_data_path_candidates(f"tsp_{graph_type}{graph_size}_test_seed1234.pkl")
                )
                if data_path is None:
                    raise FileNotFoundError
                with open(data_path, "rb") as f:
                    data = pickle.load(f)
                self.coords = torch.as_tensor(data[:num_samples], dtype=torch.float32)
            except Exception:
                assert graph_type == "unif", "Currently only uniform graph type is supported for training data."
                self.coords = torch.empty(num_samples, graph_size, 2).uniform_(0, 1)
        else:
            assert graph_type == "unif", "Currently only uniform graph type is supported for training data."
            self.coords = torch.empty(num_samples, graph_size, 2).uniform_(0, 1)

    def __len__(self):
        return self.coords.size(0)

    def __getitem__(self, idx):
        return {"coords": self.coords[idx]}  # (N,2)

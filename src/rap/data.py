from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .toydata import ToySpec, generate_toy_sample


def _smooth_1d(x: np.ndarray, k: int = 3) -> np.ndarray:
    # simple moving average for smoothing along time axis
    if k <= 1:
        return x
    pad = k // 2
    xpad = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(x)
    for t in range(x.shape[0]):
        out[t] = xpad[t:t+k].mean(axis=0)
    return out


def perturb_ego_history(
    agent_hist: np.ndarray,
    prob: float,
    sigma_xy: float,
    sigma_yaw: float,
    rng: np.random.Generator,
) -> np.ndarray:
    '''
    Roughly mimics RAP's ego perturbation:
      - pick a random history timestep
      - add noise to ego x,y and yaw (sin/cos)
      - smooth the trajectory

    agent_hist: (N, H, F_hist), where ego is index 0 and features include
      [x, y, vx, vy, sin(yaw), cos(yaw), type_id]
    '''
    if rng.random() > prob:
        return agent_hist

    x = agent_hist.copy()
    H = x.shape[1]
    t0 = int(rng.integers(0, H))

    # position noise
    x[0, t0, 0:2] += rng.normal(0.0, sigma_xy, size=(2,)).astype(np.float32)

    # yaw noise: convert sin/cos to angle then perturb then back
    sin_y = x[0, t0, 4]
    cos_y = x[0, t0, 5]
    yaw = float(np.arctan2(sin_y, cos_y))
    yaw += float(rng.normal(0.0, sigma_yaw))
    x[0, t0, 4] = np.sin(yaw).astype(np.float32)
    x[0, t0, 5] = np.cos(yaw).astype(np.float32)

    # smooth x,y and yaw sin/cos across time
    x[0, :, 0:2] = _smooth_1d(x[0, :, 0:2], k=3)
    x[0, :, 4:6] = _smooth_1d(x[0, :, 4:6], k=3)

    # recompute velocity from positions
    dt = 0.2
    vxy = np.zeros((H, 2), dtype=np.float32)
    vxy[1:] = (x[0, 1:, 0:2] - x[0, :-1, 0:2]) / dt
    x[0, :, 2:4] = vxy
    return x


class ToyDataset(Dataset):
    def __init__(self, spec: ToySpec, steps: int, augment: Optional[Dict[str, Any]] = None):
        self.spec = spec
        self.steps = steps
        self.augment = augment or {}
        self.rng = np.random.default_rng(spec.seed)

    def __len__(self) -> int:
        return self.steps

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = generate_toy_sample(self.spec, self.rng)

        # augment ego history (feedback-asymmetry mitigation)
        sample["agent_hist"] = perturb_ego_history(
            sample["agent_hist"],
            prob=float(self.augment.get("ego_perturb_prob", 0.0)),
            sigma_xy=float(self.augment.get("ego_perturb_sigma_xy", 0.0)),
            sigma_yaw=float(self.augment.get("ego_perturb_sigma_yaw", 0.0)),
            rng=self.rng,
        )
        return sample


class NPZFolderDataset(Dataset):
    '''
    Loads samples from a folder of `.npz` files.
    You can create these via your own preprocessing pipeline (e.g., nuPlan -> npz).

    This class assumes each file contains keys described in README.
    '''
    def __init__(self, folder: str):
        self.files = sorted(glob.glob(os.path.join(folder, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in: {folder}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}


def collate_batch(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    '''
    Pads variable-length dimensions (agents, map elems, obstacles) to max in batch.
    Creates masks where needed.

    Returns tensors with batch dimension first.
    '''
    # Determine max sizes
    maxN = max(s["agent_hist"].shape[0] for s in samples)
    maxM = max(s["map_poly"].shape[0] for s in samples)
    maxO = max(s["obstacles"].shape[0] for s in samples)

    H = samples[0]["agent_hist"].shape[1]
    F_hist = samples[0]["agent_hist"].shape[2]
    T = samples[0]["agent_fut"].shape[1]
    P = samples[0]["map_poly"].shape[1]
    F_map = samples[0]["map_poly"].shape[2]
    S = samples[0]["drivable_poly"].shape[0]

    # Allocate
    agent_hist = np.zeros((len(samples), maxN, H, F_hist), dtype=np.float32)
    agent_hist_valid = np.zeros((len(samples), maxN, H), dtype=np.bool_)
    agent_fut = np.zeros((len(samples), maxN, T, 4), dtype=np.float32)
    agent_box = np.zeros((len(samples), maxN, 2), dtype=np.float32)

    map_poly = np.zeros((len(samples), maxM, P, F_map), dtype=np.float32)
    map_center = np.zeros((len(samples), maxM, 2), dtype=np.float32)
    map_on_route = np.zeros((len(samples), maxM), dtype=np.bool_)

    obstacles = np.zeros((len(samples), maxO, 5), dtype=np.float32)
    drivable_poly = np.zeros((len(samples), S, 2), dtype=np.float32)

    agent_mask = np.zeros((len(samples), maxN), dtype=np.bool_)
    map_mask = np.zeros((len(samples), maxM), dtype=np.bool_)
    obstacle_mask = np.zeros((len(samples), maxO), dtype=np.bool_)

    for b, s in enumerate(samples):
        N = s["agent_hist"].shape[0]
        M = s["map_poly"].shape[0]
        O = s["obstacles"].shape[0]
        agent_hist[b, :N] = s["agent_hist"]
        agent_hist_valid[b, :N] = s["agent_hist_valid"]
        agent_fut[b, :N] = s["agent_fut"]
        agent_box[b, :N] = s["agent_box"]
        agent_mask[b, :N] = True

        map_poly[b, :M] = s["map_poly"]
        map_center[b, :M] = s["map_poly_center"]
        map_on_route[b, :M] = s["map_on_route"]
        map_mask[b, :M] = True

        obstacles[b, :O] = s["obstacles"]
        obstacle_mask[b, :O] = True

        drivable_poly[b] = s["drivable_poly"]

    batch = {
        "agent_hist": torch.from_numpy(agent_hist),
        "agent_hist_valid": torch.from_numpy(agent_hist_valid),
        "agent_fut": torch.from_numpy(agent_fut),
        "agent_box": torch.from_numpy(agent_box),
        "agent_mask": torch.from_numpy(agent_mask),

        "map_poly": torch.from_numpy(map_poly),
        "map_center": torch.from_numpy(map_center),
        "map_on_route": torch.from_numpy(map_on_route),
        "map_mask": torch.from_numpy(map_mask),

        "obstacles": torch.from_numpy(obstacles),
        "obstacle_mask": torch.from_numpy(obstacle_mask),

        "drivable_poly": torch.from_numpy(drivable_poly),
    }
    return batch


def build_dataloader(cfg: Dict[str, Any], augment: Optional[Dict[str, Any]] = None) -> DataLoader:
    name = cfg.get("name", "toy")
    batch_size = int(cfg.get("batch_size", 16))
    num_workers = int(cfg.get("num_workers", 0))
    if name == "toy":
        spec = ToySpec(
            num_agents=int(cfg.get("num_agents", 8)),
            hist_len=int(cfg.get("hist_len", 8)),
            fut_len=int(cfg.get("fut_len", 16)),
            map_elems=int(cfg.get("map_elems", 24)),
            map_points=int(cfg.get("map_points", 10)),
            obstacles=int(cfg.get("obstacles", 6)),
            seed=int(cfg.get("seed", 0)),
        )
        steps = int(cfg.get("steps_per_epoch", 200)) * int(cfg.get("epochs", 5))
        ds = ToyDataset(spec, steps=steps, augment=augment)
    else:
        folder = cfg.get("folder", "")
        ds = NPZFolderDataset(folder)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_batch,
        drop_last=False,
    )

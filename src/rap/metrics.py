from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .model import RAPOutput
from .losses import select_best_mode


def compute_ade_fde(pred_xy: torch.Tensor, gt_xy: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    '''
    pred_xy: (B,N,T,2)
    gt_xy:   (B,N,T,2)
    mask:    (B,N) bool
    '''
    diff = (pred_xy - gt_xy).norm(dim=-1)  # (B,N,T)
    diff = diff * mask[:, :, None].float()
    ade = diff.mean(dim=-1).sum() / (mask.sum() + 1e-6)  # mean over time, avg over valid agents
    fde = diff[:, :, -1].sum() / (mask.sum() + 1e-6)
    return {"ADE": ade, "FDE": fde}


@torch.no_grad()
def evaluate_batch(out: RAPOutput, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    gt = batch["agent_fut"]
    agent_mask = batch["agent_mask"]
    traj = out.traj_refined

    k_hat = select_best_mode(traj, gt, agent_mask)
    B, N, K, T, _ = traj.shape
    idx = k_hat.view(B, 1, 1, 1, 1).expand(B, N, 1, T, 4)
    best = traj.gather(dim=2, index=idx).squeeze(2)  # (B,N,T,4)

    metrics_all = compute_ade_fde(best[..., 0:2], gt[..., 0:2], agent_mask)

    # Ego-only metrics
    ego_mask = agent_mask[:, :1]
    metrics_ego = compute_ade_fde(best[:, :1, :, 0:2], gt[:, :1, :, 0:2], ego_mask)
    metrics_all.update({f"ego_{k}": v for k, v in metrics_ego.items()})

    return metrics_all

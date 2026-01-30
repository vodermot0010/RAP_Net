from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .model import RAPOutput


def select_best_mode(traj: torch.Tensor, gt: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
    '''
    Implements the idea of Eq.(13) in RAP:
      k_hat = argmin_k sum_{i,t} || l_{t,i,k} - g_{t,i} ||_1

    traj: (B, N, K, T, 4)
    gt:   (B, N, T, 4)
    agent_mask: (B, N) bool (True=valid)
    returns:
      k_hat: (B,) long
    '''
    B, N, K, T, _ = traj.shape
    # L1 on positions only
    diff = (traj[..., 0:2] - gt[:, :, None, :, 0:2]).abs().sum(dim=-1)  # (B,N,K,T)
    # mask padded agents
    diff = diff * agent_mask[:, :, None, None].float()
    score = diff.sum(dim=(1, 3))  # (B,K) sum over agents & time
    k_hat = score.argmin(dim=1)
    return k_hat


def imitation_loss(
    out: RAPOutput,
    gt: torch.Tensor,
    agent_mask: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    '''
    Returns:
      total imitation loss,
      components dict,
      k_hat (B,)
    '''
    traj_prop = out.traj_proposal
    traj_ref = out.traj_refined
    logits = out.mode_logits

    B, N, K, T, _ = traj_ref.shape
    k_hat = select_best_mode(traj_ref, gt, agent_mask)  # (B,)

    # gather best-mode trajectories
    idx = k_hat.view(B, 1, 1, 1, 1).expand(B, N, 1, T, 4)
    best_prop = traj_prop.gather(dim=2, index=idx).squeeze(2)  # (B,N,T,4)
    best_ref = traj_ref.gather(dim=2, index=idx).squeeze(2)    # (B,N,T,4)

    # regression (Huber / smooth L1)
    # mask padded agents
    mask = agent_mask[:, :, None, None].float()
    lreg1 = F.smooth_l1_loss(best_prop * mask, gt * mask, reduction="sum") / (mask.sum() + 1e-6)
    lreg2 = F.smooth_l1_loss(best_ref * mask, gt * mask, reduction="sum") / (mask.sum() + 1e-6)

    # classification
    lcls = F.cross_entropy(logits, k_hat)

    loss = lreg1 + lreg2 + lcls
    return loss, {"lreg1": lreg1, "lreg2": lreg2, "lcls": lcls}, k_hat


# -----------------------
# Geometry helpers (torch)
# -----------------------

def rect_corners(center: torch.Tensor, yaw: torch.Tensor, length: torch.Tensor, width: torch.Tensor) -> torch.Tensor:
    '''
    center: (B,T,2)
    yaw:    (B,T)
    length,width: (B,1) or (B,T) broadcastable
    returns corners: (B,T,4,2) in CCW order
    '''
    B, T, _ = center.shape
    hx = (length / 2.0).view(B, -1)  # (B,1) or (B,T)
    hy = (width / 2.0).view(B, -1)
    # broadcast to (B,T)
    if hx.shape[1] == 1:
        hx = hx.expand(B, T)
    if hy.shape[1] == 1:
        hy = hy.expand(B, T)

    local = torch.stack([
        torch.stack([ hx,  hy], dim=-1),
        torch.stack([ hx, -hy], dim=-1),
        torch.stack([-hx, -hy], dim=-1),
        torch.stack([-hx,  hy], dim=-1),
    ], dim=2)  # (B,T,4,2)

    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    R = torch.stack([
        torch.stack([cy, -sy], dim=-1),
        torch.stack([sy,  cy], dim=-1),
    ], dim=-2)  # (B,T,2,2)

    corners = (local @ R.transpose(-1, -2)) + center.unsqueeze(2)
    return corners


def point_in_poly(points: torch.Tensor, poly: torch.Tensor) -> torch.Tensor:
    '''
    Ray casting (non-differentiable boolean) for inside polygon test.
    points: (B, ..., 2)
    poly:   (B, S, 2) polygon vertices (not necessarily closed)
    returns: bool mask (B, ...)
    '''
    B = points.shape[0]
    S = poly.shape[1]
    px = points[..., 0]
    py = points[..., 1]
    x1 = poly[:, :, 0]
    y1 = poly[:, :, 1]
    x2 = torch.roll(x1, shifts=-1, dims=1)
    y2 = torch.roll(y1, shifts=-1, dims=1)

    # Expand poly dims to broadcast over points
    # points dims: (B, Q); we want (B, Q, S)
    Q_shape = px.shape[1:]
    px_e = px.unsqueeze(-1).expand(*px.shape, S)
    py_e = py.unsqueeze(-1).expand(*py.shape, S)
    x1_e = x1.view(B, *([1] * len(Q_shape)), S).expand(B, *Q_shape, S)
    y1_e = y1.view(B, *([1] * len(Q_shape)), S).expand(B, *Q_shape, S)
    x2_e = x2.view(B, *([1] * len(Q_shape)), S).expand(B, *Q_shape, S)
    y2_e = y2.view(B, *([1] * len(Q_shape)), S).expand(B, *Q_shape, S)

    # Edge crosses horizontal ray
    cond1 = (y1_e > py_e) != (y2_e > py_e)
    # Compute x coordinate of intersection
    denom = (y2_e - y1_e).clamp(min=1e-6)
    x_int = (x2_e - x1_e) * (py_e - y1_e) / denom + x1_e
    cond2 = px_e < x_int
    crossings = (cond1 & cond2).to(torch.int32).sum(dim=-1)
    inside = (crossings % 2 == 1)
    return inside


def point_to_segments_distance(points: torch.Tensor, seg_a: torch.Tensor, seg_b: torch.Tensor) -> torch.Tensor:
    '''
    points: (B, Q, 2)
    seg_a:  (B, S, 2)
    seg_b:  (B, S, 2)
    returns min distance: (B, Q)
    '''
    B, Q, _ = points.shape
    S = seg_a.shape[1]

    p = points.unsqueeze(2)  # (B,Q,1,2)
    a = seg_a.unsqueeze(1)   # (B,1,S,2)
    b = seg_b.unsqueeze(1)   # (B,1,S,2)
    ab = b - a               # (B,1,S,2)
    ap = p - a               # (B,Q,S,2)

    ab2 = (ab ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-6)
    t = (ap * ab).sum(dim=-1, keepdim=True) / ab2
    t = t.clamp(0.0, 1.0)
    proj = a + t * ab
    d = ((p - proj) ** 2).sum(dim=-1).sqrt()  # (B,Q,S)
    return d.min(dim=-1).values  # (B,Q)


def signed_distance_to_polygon(points: torch.Tensor, poly: torch.Tensor) -> torch.Tensor:
    '''
    Signed distance convention:
      + outside, - inside (like RAP's road distance)
    points: (B, Q, 2)
    poly:   (B, S, 2)
    returns: (B, Q)
    '''
    seg_a = poly
    seg_b = torch.roll(poly, shifts=-1, dims=1)
    dist = point_to_segments_distance(points, seg_a, seg_b)  # (B,Q)
    inside = point_in_poly(points, poly)  # (B,Q) bool
    signed = torch.where(inside, -dist, dist)
    return signed


def signed_inside_distance_to_rect(points: torch.Tensor, rect: torch.Tensor) -> torch.Tensor:
    '''
    rect: (B, O, 5) [cx, cy, L, W, yaw]
    points: (B, Q, 2)
    returns signed_inside: (B, Q, O)
      + inside, - outside, magnitude is distance to boundary (approx via SDF)
    '''
    B, Q, _ = points.shape
    O = rect.shape[1]
    c = rect[:, :, 0:2]  # (B,O,2)
    L = rect[:, :, 2:3]
    W = rect[:, :, 3:4]
    yaw = rect[:, :, 4]

    # transform points to rect frame
    p = points.unsqueeze(2)  # (B,Q,1,2)
    d = p - c.unsqueeze(1)   # (B,Q,O,2)

    cy = torch.cos(-yaw).unsqueeze(1)  # (B,1,O)
    sy = torch.sin(-yaw).unsqueeze(1)
    # rotation for each obstacle
    x = d[..., 0]
    y = d[..., 1]
    xr = cy * x - sy * y
    yr = sy * x + cy * y

    hx = (L / 2.0).squeeze(-1).unsqueeze(1)  # (B,1,O)
    hy = (W / 2.0).squeeze(-1).unsqueeze(1)

    qx = xr.abs() - hx
    qy = yr.abs() - hy

    # SDF for axis-aligned rectangle (positive outside, negative inside)
    outside = torch.stack([qx.clamp(min=0.0), qy.clamp(min=0.0)], dim=-1).norm(dim=-1)
    inside = torch.minimum(torch.maximum(qx, qy), torch.zeros_like(qx))
    sdf = outside + inside  # positive outside, negative inside

    signed_inside = -sdf
    return signed_inside


def ego_auxiliary_losses(
    ego_traj: torch.Tensor,          # (B,T,4)
    gt_agents: torch.Tensor,         # (B,N,T,4) including ego at index 0
    agent_box: torch.Tensor,         # (B,N,2) length,width
    drivable_poly: torch.Tensor,     # (B,S,2)
    obstacles: torch.Tensor,         # (B,O,5)
    agent_mask: torch.Tensor,        # (B,N)
    obstacle_mask: torch.Tensor,     # (B,O)
    eps_road: float,
    eps_obstacle: float,
    eps_L: float,
    eps_W: float,
) -> Dict[str, torch.Tensor]:
    '''
    Ego-only vectorized auxiliary losses inspired by RAP (Eq. 18-21).
    This implementation is simplified but differentiable wrt ego trajectory.

    - L_road: keep ego bounding box corners inside drivable polygon with margin eps_road
    - L_obstacle: keep ego corners away from static obstacles with margin eps_obstacle
    - L_agent: keep ego center away from other agents with inflated boxes (eps_L, eps_W)
    '''
    B, T, _ = ego_traj.shape
    # ego pose from (sin, cos)
    yaw = torch.atan2(ego_traj[:, :, 2], ego_traj[:, :, 3])  # (B,T)
    center = ego_traj[:, :, 0:2]

    ego_L = agent_box[:, 0, 0:1]  # (B,1)
    ego_W = agent_box[:, 0, 1:2]

    corners = rect_corners(center, yaw, ego_L, ego_W)  # (B,T,4,2)
    corners_flat = corners.reshape(B, T * 4, 2)

    # --- road loss ---
    sd_road = signed_distance_to_polygon(corners_flat, drivable_poly)  # (B, T*4)
    l_road = F.relu(sd_road + float(eps_road)).mean()

    # --- obstacle loss ---
    if obstacles.numel() == 0:
        l_obst = torch.zeros((), device=ego_traj.device)
    else:
        signed_in = signed_inside_distance_to_rect(corners_flat, obstacles)  # (B, T*4, O)
        # mask pad obstacles
        signed_in = signed_in * obstacle_mask[:, None, :].float() + (-1e6) * (~obstacle_mask)[:, None, :].float()
        # For each corner, consider the "worst" (closest/inside-most) obstacle
        worst = (signed_in + float(eps_obstacle)).max(dim=-1).values  # (B,T*4)
        l_obst = F.relu(worst).mean()

    # --- agent collision loss (center-to-center with inflated boxes, axis-aligned in ego frame) ---
    # Use GT agent futures (excluding ego).
    if gt_agents.shape[1] <= 1:
        l_agent = torch.zeros((), device=ego_traj.device)
    else:
        other = gt_agents[:, 1:, :, 0:2]  # (B,N-1,T,2)
        other_mask = agent_mask[:, 1:]    # (B,N-1)
        other_L = agent_box[:, 1:, 0:1]   # (B,N-1,1)
        other_W = agent_box[:, 1:, 1:2]

        ego_c = center[:, None, :, :]  # (B,1,T,2)
        d = (ego_c - other).abs()      # (B,N-1,T,2)

        halfL = (ego_L[:, None, :] / 2.0) + (other_L / 2.0) + float(eps_L)  # (B,N-1,1)
        halfW = (ego_W[:, None, :] / 2.0) + (other_W / 2.0) + float(eps_W)

        dx = halfL - d[..., 0]  # (B,N-1,T)
        dy = halfW - d[..., 1]
        pen = torch.minimum(dx, dy)          # positive only if overlap in both dims
        pen = F.relu(pen)

        pen = pen * other_mask[:, :, None].float()
        l_agent = pen.sum() / (other_mask.sum() * T + 1e-6)

    return {"l_road": l_road, "l_obstacle": l_obst, "l_agent": l_agent}


def total_loss(
    out: RAPOutput,
    batch: Dict[str, torch.Tensor],
    lambda_aux: float,
    eps_road: float,
    eps_obstacle: float,
    eps_L: float,
    eps_W: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    gt = batch["agent_fut"]                 # (B,N,T,4)
    agent_mask = batch["agent_mask"]        # (B,N)
    obstacles = batch["obstacles"]          # (B,O,5)
    obstacle_mask = batch["obstacle_mask"]  # (B,O)
    agent_box = batch["agent_box"]          # (B,N,2)
    drivable_poly = batch["drivable_poly"]  # (B,S,2)

    lim, comps, k_hat = imitation_loss(out, gt, agent_mask)

    # pick ego trajectory from best mode (planning output)
    B, N, K, T, _ = out.traj_refined.shape
    idx = k_hat.view(B, 1, 1, 1).expand(B, 1, T, 4)  # for ego only
    ego_best = out.traj_refined[:, 0].gather(dim=1, index=idx).squeeze(1)  # (B,T,4)

    aux = ego_auxiliary_losses(
        ego_best,
        gt,
        agent_box,
        drivable_poly,
        obstacles,
        agent_mask,
        obstacle_mask,
        eps_road=eps_road,
        eps_obstacle=eps_obstacle,
        eps_L=eps_L,
        eps_W=eps_W,
    )
    laux = aux["l_road"] + aux["l_obstacle"] + aux["l_agent"]
    loss = lim + float(lambda_aux) * laux

    comps_all = {**comps, **aux, "l_imitation": lim, "l_aux": laux}
    return loss, comps_all
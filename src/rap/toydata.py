from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class ToySpec:
    num_agents: int
    hist_len: int
    fut_len: int
    map_elems: int
    map_points: int
    obstacles: int
    seed: int = 0


def _rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def generate_toy_sample(spec: ToySpec, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    '''
    Produce one synthetic sample in an ego-centered coordinate frame.

    Shapes (fixed):
      - N_total = 1 + num_agents
      - agent_hist: (N_total, H, F_hist)
      - agent_fut:  (N_total, T, 4)  [x, y, sin(yaw), cos(yaw)]
      - map_poly:   (M, P, F_map)
      - obstacles:  (O, 5) [x, y, length, width, yaw]
      - drivable_poly: (S, 2) polygon boundary
    '''
    N = 1 + spec.num_agents
    H = spec.hist_len
    T = spec.fut_len
    M = spec.map_elems
    P = spec.map_points
    O = spec.obstacles

    # ---------- Ego & agents kinematics ----------
    # agent types: 0=ego-car, 1=vehicle, 2=pedestrian
    agent_type = np.zeros((N,), dtype=np.int64)
    agent_type[0] = 0
    agent_type[1:] = rng.choice([1, 2], size=(N - 1,), p=[0.8, 0.2])

    # boxes (length, width)
    agent_box = np.zeros((N, 2), dtype=np.float32)
    agent_box[0] = np.array([4.8, 2.0], dtype=np.float32)
    for i in range(1, N):
        if agent_type[i] == 1:
            agent_box[i] = np.array([4.5, 1.9], dtype=np.float32) + rng.normal(0, 0.2, size=(2,))
        else:
            agent_box[i] = np.array([0.8, 0.6], dtype=np.float32) + rng.normal(0, 0.05, size=(2,))

    # initial pose (ego at origin)
    pos0 = rng.normal(0, 8.0, size=(N, 2)).astype(np.float32)
    pos0[0] = 0.0
    yaw0 = rng.uniform(-0.2, 0.2, size=(N,)).astype(np.float32)  # radians

    # speeds
    speed = np.zeros((N,), dtype=np.float32)
    speed[0] = rng.uniform(4.0, 10.0)
    for i in range(1, N):
        speed[i] = rng.uniform(0.8, 8.0) if agent_type[i] == 1 else rng.uniform(0.3, 2.0)

    # Ego follows a "route" mostly along +x
    ego_dir = np.array([1.0, 0.0], dtype=np.float32)

    # Simple constant-velocity rollouts for history and future
    def rollout(pos_init: np.ndarray, yaw_init: float, v: float, heading: np.ndarray, steps: int, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        pos = np.zeros((steps, 2), dtype=np.float32)
        yaw = np.zeros((steps,), dtype=np.float32)
        pos[0] = pos_init
        yaw[0] = yaw_init
        for t in range(1, steps):
            pos[t] = pos[t - 1] + heading * v * dt
            yaw[t] = yaw[t - 1]  # keep yaw constant
        return pos, yaw

    dt_hist = 0.2

    hist_pos = np.zeros((N, H, 2), dtype=np.float32)
    hist_yaw = np.zeros((N, H), dtype=np.float32)
    fut_pos = np.zeros((N, T, 2), dtype=np.float32)
    fut_yaw = np.zeros((N, T), dtype=np.float32)

    for i in range(N):
        if i == 0:
            heading = ego_dir
        else:
            # random heading for other agents
            th = rng.uniform(-np.pi, np.pi)
            heading = _rot(th) @ np.array([1.0, 0.0], dtype=np.float32)
        pos_fwd, yaw_fwd = rollout(pos0[i], yaw0[i], speed[i], heading, H + T, dt_hist)
        hist_pos[i] = pos_fwd[:H]
        hist_yaw[i] = yaw_fwd[:H]
        fut_pos[i] = pos_fwd[H:H+T]
        fut_yaw[i] = yaw_fwd[H:H+T]

    # Ego-centric transform: center at ego current pose (last history frame)
    ego_center = hist_pos[0, -1].copy()
    ego_yaw = hist_yaw[0, -1].copy()
    R = _rot(-ego_yaw)  # rotate world -> ego

    def to_ego(xy: np.ndarray) -> np.ndarray:
        return (xy - ego_center[None, ...]) @ R.T

    for i in range(N):
        hist_pos[i] = to_ego(hist_pos[i])
        fut_pos[i] = to_ego(fut_pos[i])
        hist_yaw[i] = hist_yaw[i] - ego_yaw
        fut_yaw[i] = fut_yaw[i] - ego_yaw

    # Construct agent_hist features: [x, y, vx, vy, sin(yaw), cos(yaw), type_id]
    F_hist = 7
    agent_hist = np.zeros((N, H, F_hist), dtype=np.float32)
    agent_hist_valid = np.ones((N, H), dtype=np.bool_)
    for i in range(N):
        agent_hist[i, :, 0:2] = hist_pos[i]
        # velocity as finite difference
        vxy = np.zeros((H, 2), dtype=np.float32)
        vxy[1:] = (hist_pos[i, 1:] - hist_pos[i, :-1]) / dt_hist
        agent_hist[i, :, 2:4] = vxy
        agent_hist[i, :, 4] = np.sin(hist_yaw[i])
        agent_hist[i, :, 5] = np.cos(hist_yaw[i])
        agent_hist[i, :, 6] = float(agent_type[i])

    # Future ground truth: [x,y,sin(yaw),cos(yaw)]
    agent_fut = np.zeros((N, T, 4), dtype=np.float32)
    agent_fut[:, :, 0:2] = fut_pos
    agent_fut[:, :, 2] = np.sin(fut_yaw)
    agent_fut[:, :, 3] = np.cos(fut_yaw)

    # ---------- Map polylines ----------
    # Drivable area: simple rectangle polygon (counterclockwise)
    road_w = 12.0
    road_l = 80.0
    drivable_poly = np.array([
        [-10.0, -road_w],
        [road_l, -road_w],
        [road_l, road_w],
        [-10.0, road_w],
    ], dtype=np.float32)

    # Create map elements as lane centerline segments (polyline points)
    map_poly = np.zeros((M, P, 6), dtype=np.float32)  # [x,y,sin,cos,speed_limit,is_lane]
    map_center = np.zeros((M, 2), dtype=np.float32)
    map_on_route = np.zeros((M,), dtype=np.bool_)

    for m in range(M):
        is_route = (m < max(3, M // 6))  # first few are route lanes
        y = rng.normal(0.0, 0.8) if is_route else rng.normal(0.0, 5.0)
        x0 = rng.uniform(-5.0, road_l - 10.0)
        x1 = x0 + rng.uniform(8.0, 25.0)
        xs = np.linspace(x0, x1, P, dtype=np.float32)
        ys = np.full((P,), y, dtype=np.float32) + rng.normal(0, 0.05, size=(P,)).astype(np.float32)
        yaw = 0.0  # lane heading along x
        map_poly[m, :, 0] = xs
        map_poly[m, :, 1] = ys
        map_poly[m, :, 2] = np.sin(yaw)
        map_poly[m, :, 3] = np.cos(yaw)
        map_poly[m, :, 4] = rng.uniform(0.3, 1.0)  # normalized speed limit
        map_poly[m, :, 5] = 1.0  # lane indicator
        map_center[m] = np.array([xs.mean(), ys.mean()], dtype=np.float32)
        map_on_route[m] = bool(is_route)

    # ---------- Static obstacles ----------
    obstacles = np.zeros((O, 5), dtype=np.float32)
    for o in range(O):
        x = rng.uniform(5.0, road_l - 5.0)
        y = rng.uniform(-road_w + 1.0, road_w - 1.0)
        L = rng.uniform(1.0, 4.0)
        W = rng.uniform(0.8, 2.0)
        yaw = rng.uniform(-np.pi, np.pi)
        obstacles[o] = np.array([x, y, L, W, yaw], dtype=np.float32)

    sample = {
        "agent_hist": agent_hist,
        "agent_hist_valid": agent_hist_valid,
        "agent_fut": agent_fut,
        "agent_box": agent_box,
        "map_poly": map_poly,
        "map_poly_center": map_center,
        "map_on_route": map_on_route,
        "drivable_poly": drivable_poly,
        "obstacles": obstacles,
    }
    return sample

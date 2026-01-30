from __future__ import annotations

import argparse
import os

import torch

from .rap.data import build_dataloader
from .rap.model import RAPLITE
from .rap.utils import load_config, resolve_device, set_seed, to_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--ckpt", type=str, default="", help="Optional checkpoint path to load")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(cfg.get("device", "auto"))

    data_cfg = cfg.get("data", {})
    augment_cfg = cfg.get("augment", {})

    dl = build_dataloader({**data_cfg, "seed": int(cfg.get("seed", 42))}, augment=augment_cfg)

    model_cfg = cfg.get("model", {})
    model = RAPLITE(
        d_model=int(model_cfg.get("d_model", 128)),
        nhead=int(model_cfg.get("nhead", 8)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        num_modes=int(model_cfg.get("num_modes", 6)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        state_dropout_prob=float(augment_cfg.get("state_dropout_prob", 0.0)),
        fut_len=int(data_cfg.get("fut_len", 16)),
        use_refine=bool(model_cfg.get("use_refine", True)),
        hist_in_dim=7,
        map_in_dim=6,
    ).to(device)
    model.eval()

    if args.ckpt and os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
        print(f"Loaded checkpoint: {args.ckpt}")

    batch = to_device(next(iter(dl)), device)
    out = model(batch)

    # choose mode by highest logit (planning inference)
    k = out.mode_logits.argmax(dim=1)  # (B,)
    B, N, K, T, _ = out.traj_refined.shape
    idx = k.view(B, 1, 1, 1, 1).expand(B, N, 1, T, 4)
    plan = out.traj_refined.gather(dim=2, index=idx).squeeze(2)  # (B,N,T,4)

    print("Shapes:")
    print("  traj_refined:", tuple(out.traj_refined.shape))
    print("  mode_logits :", tuple(out.mode_logits.shape))
    print("  selected plan:", tuple(plan.shape))
    print("First sample ego planned trajectory (x,y) first 5 steps:")
    print(plan[0, 0, :5, 0:2].cpu().numpy())


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from .rap.data import build_dataloader
from .rap.losses import total_loss
from .rap.metrics import evaluate_batch
from .rap.model import RAPLITE
from .rap.utils import count_parameters, load_config, resolve_device, set_seed, to_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to a YAML config, e.g. configs/toy.yaml")
    p.add_argument("--workdir", type=str, default="runs/rap_lite", help="Output directory for checkpoints/logs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = resolve_device(cfg.get("device", "auto"))
    os.makedirs(args.workdir, exist_ok=True)

    # Data
    data_cfg = cfg.get("data", {})
    augment_cfg = cfg.get("augment", {})
    dl = build_dataloader({**data_cfg, "seed": seed}, augment=augment_cfg)

    # Model
    model_cfg = cfg.get("model", {})
    loss_cfg = cfg.get("loss", {})
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

    print(f"Device: {device}")
    print(f"Trainable params: {count_parameters(model):,}")

    # Optim
    optim_cfg = cfg.get("optim", {})
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(optim_cfg.get("lr", 8e-4)),
        weight_decay=float(optim_cfg.get("weight_decay", 0.01)),
    )

    grad_clip = float(optim_cfg.get("grad_clip", 1.0))

    epochs = int(data_cfg.get("epochs", 5))
    steps_per_epoch = int(data_cfg.get("steps_per_epoch", len(dl)))

    global_step = 0
    model.train()

    iterator = iter(dl)
    for epoch in range(epochs):
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs}")
        loss_acc = 0.0
        metric_acc: Dict[str, float] = {}

        for _ in pbar:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dl)
                batch = next(iterator)

            batch = to_device(batch, device)
            opt.zero_grad(set_to_none=True)

            out = model(batch)
            loss, comps = total_loss(
                out, batch,
                lambda_aux=float(loss_cfg.get("lambda_aux", 0.2)),
                eps_road=float(loss_cfg.get("eps_road", 0.2)),
                eps_obstacle=float(loss_cfg.get("eps_obstacle", 0.2)),
                eps_L=float(loss_cfg.get("eps_L", 0.3)),
                eps_W=float(loss_cfg.get("eps_W", 0.2)),
            )
            loss.backward()
            if grad_clip > 0:
                clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            loss_acc += float(loss.item())
            # show key losses
            pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "im": f"{comps['l_imitation'].item():.3f}",
                "aux": f"{comps['l_aux'].item():.3f}",
            })
            global_step += 1

        # quick eval on one batch
        model.eval()
        with torch.no_grad():
            batch = to_device(next(iter(dl)), device)
            out = model(batch)
            metrics = evaluate_batch(out, batch)
        model.train()

        ckpt_path = os.path.join(args.workdir, f"ckpt_epoch{epoch+1}.pt")
        torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_path)

        print(f"[Epoch {epoch+1}] avg_loss={loss_acc/steps_per_epoch:.4f}  "
              f"ego_ADE={metrics['ego_ADE']:.3f}  ego_FDE={metrics['ego_FDE']:.3f}  saved={ckpt_path}")


if __name__ == "__main__":
    main()

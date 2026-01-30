# RAP-Lite: Role-Aware Joint Prediction & Planning (Trainable Reference Implementation)

This repository provides a **trainable** PyTorch reference implementation inspired by the paper:

- *RAP: Role-Aware Joint Prediction and Planning in Autonomous Driving* (Tang et al., RA-L 2026)

The goal is not to exactly reproduce the nuPlan benchmark results, but to provide a clean, extensible codebase that captures
RAP's **three asymmetric designs**:

1. **Route–Identity Token Pairing** (route-aware yet agent-agnostic conditioning)
2. **Ego-only Vectorized Auxiliary Losses** (safety-constrained planning without harming agent prediction)
3. **Dropout + Perturbation** on ego history/kinematics (closed-loop robustness)

The repo includes a **Toy dataset** so you can run training end-to-end on CPU/GPU.

---

## Quick start

### 1) Install
```bash
pip install -r requirements.txt
```

### 2) Train on toy data
```bash
python -m src.train --config configs/toy.yaml
```

You should see the training loss decrease within a few epochs.

### 3) Run a quick inference demo
```bash
python -m src.demo_infer --config configs/toy.yaml
```

---

## Data format (for real datasets)

To train on a real dataset (e.g., nuPlan), prepare `.npz` samples with fields like:

- `agent_hist`: float32, shape (N+1, H, F_hist)  # index 0 is ego
- `agent_hist_valid`: bool, shape (N+1, H)
- `agent_fut`: float32, shape (N+1, T, 4)        # x, y, sin(yaw), cos(yaw)
- `agent_box`: float32, shape (N+1, 2)           # length, width
- `map_poly`: float32, shape (M, P, F_map)
- `map_poly_center`: float32, shape (M, 2)
- `map_on_route`: bool, shape (M,)
- `drivable_poly`: float32, shape (S, 2)         # polygon boundary (closed or open)
- `obstacles`: float32, shape (O, 5)             # x, y, length, width, yaw

See `src/rap/data.py` for details and the toy generator in `src/rap/toydata.py`.

---

## Repository layout

- `src/rap/model.py` : RAP-Lite model (role-aware encoder + multimodal interaction decoder)
- `src/rap/losses.py`: imitation loss + ego-only auxiliary losses (road/obstacle/agent collision)
- `src/rap/data.py`  : dataset + collate + augmentation (ego perturbation)
- `src/train.py`     : training loop
- `src/demo_infer.py`: inference demo

---

## Notes

- The decoder implements the paper's idea of **K scene modes** and **interaction layers**
  (spatio-temporal cross-attention → agent attention → mode attention).
- For simplicity, the "proposal refinement" step is optional (config flag).
- Vectorized safety losses require geometric inputs. The toy dataset provides simple polygons/rectangles.
  For nuPlan you should export lane boundaries / drivable polygons and obstacle boxes during preprocessing.

---

## License
MIT (see `LICENSE`).


---

## Push this project to your GitHub repository

If you downloaded this repo as a ZIP and want to push it to GitHub quickly:

### Option A (macOS/Linux): Bash script
```bash
chmod +x scripts/push_to_github.sh
./scripts/push_to_github.sh https://github.com/<user>/<repo>.git main
```

### Option B (Windows): PowerShell script
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\push_to_github.ps1 -RepoUrl https://github.com/<user>/<repo>.git -Branch main
```

> You may need to authenticate first (HTTPS token or SSH key). If your repo uses `master`, replace `main` accordingly.

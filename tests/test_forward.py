import torch

from src.rap.model import RAPLITE
from src.rap.toydata import ToySpec, generate_toy_sample
from src.rap.data import collate_batch


def test_forward():
    spec = ToySpec(num_agents=4, hist_len=6, fut_len=10, map_elems=8, map_points=6, obstacles=3, seed=0)
    rng = torch.Generator().manual_seed(0)
    # generate 2 samples
    import numpy as np
    np_rng = np.random.default_rng(0)
    samples = [generate_toy_sample(spec, np_rng) for _ in range(2)]
    batch = collate_batch(samples)
    model = RAPLITE(d_model=64, nhead=4, num_layers=1, num_modes=3, fut_len=10, use_refine=True)
    out = model(batch)
    assert out.traj_refined.shape[:4] == (2, 1+spec.num_agents, 3, 10)

"""Benchmark: per-step latency with 256 envs + GPU inference."""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from cpp_vecenv import make_cpp_vec_env

n_envs = 256
n_threads = 8

v = make_cpp_vec_env(n_envs=n_envs, map_size=280, seed=42, n_threads=n_threads)
obs_np = v.reset()

model = torch.nn.Sequential(
    torch.nn.Linear(203, 256), torch.nn.ReLU(),
    torch.nn.Linear(256, 256), torch.nn.ReLU(),
    torch.nn.Linear(256, 45),
).cuda()

obs_t = torch.from_numpy(np.asarray(obs_np, dtype=np.float32)).cuda()

torch.cuda.synchronize()
N = 200
t0 = time.perf_counter()
for _i in range(N):
    logits = model(obs_t)
    actions = torch.argmax(logits, dim=-1).cpu().numpy().astype(np.int32)
    v.step_async(actions)
    obs_np, _, _, _ = v.step_wait()
    obs_t = torch.from_numpy(np.asarray(obs_np, dtype=np.float32)).cuda()
torch.cuda.synchronize()
t1 = time.perf_counter()

total_steps = N * n_envs
fps = total_steps / (t1 - t0)
ms_per_step = (t1 - t0) / N * 1000
print(f"FPS: {fps:,.0f} ({ms_per_step:.1f} ms/step) with {n_envs} envs, {n_threads} threads")

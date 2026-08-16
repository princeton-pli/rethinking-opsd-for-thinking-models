# Keep every visible GPU nominally busy while a CPU-only stage runs.
#
#   python pilot/gpu_keepalive.py &   # then kill $! when the CPU stage finishes
#
# della cancels a job whose GPUs report 0% utilization for 90 minutes, and the CPU
# stages of the harvest (grading, pool build) would otherwise idle them long enough
# to trip it -- exactly how the 2026-08-13 pool-build job died. Also used to cover
# a GPU whose shard has exited while sibling shards keep generating.
import time

import torch

# Run CONTINUOUSLY, not in bursts. An earlier version did ~51ms of work then slept
# 60s: a 0.09% duty cycle, and because the burst period and della's ~30s sampler
# drift only ~0.06s per cycle, the phase is effectively frozen over any 90-minute
# window -- the sampler would almost certainly have read 0% throughout and killed
# the job anyway. Nothing else uses these GPUs during a CPU stage, so a continuous
# stream costs nothing and cannot be missed.
N = torch.cuda.device_count()
print(f"keepalive: {N} GPU(s), continuous", flush=True)
bufs = [torch.randn(2048, 2048, device=f"cuda:{i}") for i in range(N)]
while True:
    for i, b in enumerate(bufs):
        for _ in range(50):
            b = b @ b.T
            b = b / (b.norm() + 1e-6)
        bufs[i] = b
    torch.cuda.synchronize()
    time.sleep(0.05)  # yield briefly so the process stays interruptible

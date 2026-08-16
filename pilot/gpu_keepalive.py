# Keep every visible GPU nominally busy while a CPU-only stage runs.
#
#   python pilot/gpu_keepalive.py &   # then kill $! when the CPU stage finishes
#
# della cancels a job whose GPUs report 0% utilization for 90 minutes. The CPU
# stages of the harvest (grading ~40 min, pool build ~1h) would otherwise idle the
# GPUs long enough to trip it -- that is exactly how the 2026-08-13 pool-build job
# died. A tiny periodic matmul keeps utilization off the floor at negligible cost
# (a few ms of compute per GPU per minute).
import os
import time

import torch

INTERVAL = int(os.environ.get("KEEPALIVE_INTERVAL", "60"))
N = torch.cuda.device_count()
print(f"keepalive: {N} GPU(s), pinging every {INTERVAL}s", flush=True)
bufs = [torch.randn(2048, 2048, device=f"cuda:{i}") for i in range(N)]
while True:
    for i, b in enumerate(bufs):
        # A few hundred matmuls registers as real utilization to the sampler
        # without meaningfully competing with anything else that might run.
        for _ in range(200):
            b = b @ b.T
            b = b / (b.norm() + 1e-6)
        bufs[i] = b
    torch.cuda.synchronize()
    time.sleep(INTERVAL)

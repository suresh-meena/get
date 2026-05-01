from __future__ import annotations

import argparse
import time

import torch

from get.energy.ops import fused_motif_dot, fused_motif_dot_baseline


def _time_fn(fn, q, ku, kv, t, iters: int, warmup: int):
    for _ in range(warmup):
        _ = fn(q, ku, kv, t)
    if torch.cuda.is_available() and q.is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn(q, ku, kv, t)
    if torch.cuda.is_available() and q.is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    p = argparse.ArgumentParser(description="Micro-benchmark motif dot kernels.")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    p.add_argument("--L", type=int, default=4096)
    p.add_argument("--H", type=int, default=8)
    p.add_argument("--R", type=int, default=2)
    p.add_argument("--Dh", type=int, default=16)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=50)
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    device = torch.device(args.device)

    q = torch.randn(args.L, args.H, args.R, args.Dh, device=device, dtype=dtype)
    ku = torch.randn_like(q)
    kv = torch.randn_like(q)
    t = torch.randn_like(q)

    t_base = _time_fn(fused_motif_dot_baseline, q, ku, kv, t, args.iters, args.warmup)
    t_fast = _time_fn(fused_motif_dot, q, ku, kv, t, args.iters, args.warmup)

    speedup = (t_base / t_fast) if t_fast > 0 else float("inf")
    print(
        {
            "device": args.device,
            "dtype": args.dtype,
            "shape": [args.L, args.H, args.R, args.Dh],
            "baseline_s": round(t_base, 8),
            "fused_s": round(t_fast, 8),
            "speedup_x": round(speedup, 4),
        }
    )


if __name__ == "__main__":
    main()

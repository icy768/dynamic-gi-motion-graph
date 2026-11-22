"""End-to-end check of the PyTorch physical layer against the verified NumPy feasibility code.

Same 8 sequences (128 buckets, per-frame patterns, motion_pooling data generator), same TV weight and
starting motion. NumPy reconstructs one shared canvas; PyTorch reconstructs every frame on its own
64x64 grid from all 8 frames' buckets (the E1 formulation), so results should be close, not identical.

Writes e1_outputs/numpy_comparison.json.
"""
from __future__ import annotations

import json
import sys
import time
import warnings

import numpy as np
import torch

from project_paths import PROJECT_ROOT
from e1_physics import LayerConfig, MotionConfig, PoolConfig, pool_reconstruct

sys.path.insert(0, str(PROJECT_ROOT / "motion_pooling"))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import mcgi
    from run_feasibility import make_sequence, measure, motion_error, per_frame_tv

SEEDS = (0, 2, 3, 100, 102, 103, 200, 202)   # all three speed bins
M, ITERS = 128, 300
LAMS = json.loads((PROJECT_ROOT / "motion_pooling" / "lam_128_diff.json").read_text())["best"]


def numpy_run(seed):
    bin_ = seed // 100
    base, frames, pos, speed, ch = make_sequence(seed, bin_)
    rng = np.random.default_rng(seed + 99991)
    As, ys = measure(frames, pos, base, M, "diff", 0.0, rng)
    t0 = time.time()
    single = per_frame_tv(As, ys, LAMS["single"], ITERS)
    oracle, *_ = mcgi.pooled_recon(As, ys, pos, LAMS["all"], ITERS)
    rel = [np.zeros(2)]
    for t in range(mcgi.T - 1):
        rel.append(rel[-1] + mcgi.mdr_pair(As[t + 1], ys[t + 1], single[t]))
    p0 = np.array(rel)
    p0 = p0 - p0.mean(0)
    p = p0.copy()
    est, canvas, _ = mcgi.pooled_recon(As, ys, p, LAMS["all"], ITERS)
    for _ in range(8):
        shape, origin = mcgi.canvas_for(p)
        p = mcgi.refine_against_canvas(As, ys, canvas, shape, origin, p)
        p = p - p.mean(0)
        est, canvas, _ = mcgi.pooled_recon(As, ys, p, LAMS["all"], ITERS)
    seconds = time.time() - t0

    def mean_psnr(stack):
        return float(np.mean([mcgi.psnr(stack[t], frames[t]) for t in range(mcgi.T)]))

    return dict(seed=seed, speed=float(speed), frames=frames, pos=pos, As=As, ys=ys, p0=p0,
                numpy=dict(single=mean_psnr(single), oracle=mean_psnr(oracle), estimated=mean_psnr(est),
                           epe_start=motion_error(p0, pos), epe_final=motion_error(p, pos), seconds=seconds))


def torch_run(case, device):
    a = torch.from_numpy(np.stack(case["As"])).to(device)
    patterns = (a * 64.0).round().reshape(mcgi.T, M, 64, 64)
    bucket = torch.from_numpy(np.stack(case["ys"]))[None].to(device)
    truth = torch.from_numpy(case["pos"]).float()[None].to(device)
    zero = torch.zeros(1, mcgi.T, 64, 64, device=device)
    layer = LayerConfig(iters=ITERS, mu=0.0, tv=LAMS["all"], start="zero")
    target = torch.from_numpy(case["frames"]).float().to(device)

    def mean_psnr(x):
        return float((10 * torch.log10(1 / (x[0] - target).square().mean((-2, -1)))).mean())

    t0 = time.time()
    def pairs(p):
        return p[:, 1:] - p[:, :-1]

    oracle = pool_reconstruct(zero, patterns, bucket, pairs(truth), PoolConfig(layer=layer, refine=False))
    # Same start as NumPy (its mdr_pair chain), so no extra pairwise registration here.
    est = pool_reconstruct(zero, patterns, bucket, pairs(torch.from_numpy(case["p0"]).float()[None].to(device)),
                           PoolConfig(layer=layer, motion=MotionConfig(rounds=8), pair_stages=()), p_true=truth)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return dict(oracle=mean_psnr(oracle["x"]), estimated=mean_psnr(est["x"]),
                epe_start=float(est["epe_rounds"][0]), epe_final=float(est["epe_rounds"][-1]),
                epe_rounds=[float(e) for e in est["epe_rounds"]], seconds=time.time() - t0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for seed in SEEDS:
        case = numpy_run(seed)
        tr = torch_run(case, device)
        row = dict(seed=seed, speed=case["speed"], numpy=case["numpy"], torch=tr)
        rows.append(row)
        n = case["numpy"]
        print(f"seed {seed:3d} speed {case['speed']:.2f} | oracle np {n['oracle']:.2f} pt {tr['oracle']:.2f} | "
              f"estimated np {n['estimated']:.2f} pt {tr['estimated']:.2f} | "
              f"epe np {n['epe_final']:.3f} pt {tr['epe_final']:.3f} | s np {n['seconds']:.0f} pt {tr['seconds']:.1f}",
              flush=True)
    summary = {key: {side: float(np.mean([r[side][key] for r in rows])) for side in ("numpy", "torch")}
               for key in ("oracle", "estimated", "epe_start", "epe_final", "seconds")}
    out = PROJECT_ROOT / "e1_outputs" / "numpy_comparison.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(dict(device=str(device), measurements=M, iterations=ITERS, tv=LAMS["all"],
                                   seeds=list(SEEDS), summary=summary, rows=rows), indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

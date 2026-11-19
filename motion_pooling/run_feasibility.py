"""Run the motion-compensated pooling feasibility study. Writes JSON rows per sequence/frame."""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mcgi import *  # noqa

DIGITS = "0123456789"


def make_sequence(seed, speed_bin, intra=1):
    """Returns base image, frame stack (T,H,W), positions (T,2), and for intra>1 the
    sub-positions (T, intra, 2) at which each measurement chunk is taken."""
    rng = np.random.default_rng(seed)
    ch = DIGITS[rng.integers(10)]
    font = FONTS[rng.integers(len(FONTS))]
    base = render_digit(ch, font, rng)
    lo, hi = SPEED_BINS[speed_bin]
    speed = rng.uniform(lo, hi)
    pos, _ = trajectory(rng, speed)
    frames = np.stack([shift_image(base, p[0], p[1]) for p in pos])
    return base, frames, pos, speed, ch


def measure(frames, pos, base, M, mode, noise, rng, intra_motion=False):
    """Bucket measurements per frame. mode: 'same' patterns every frame or 'diff' per frame.
    intra_motion: object keeps moving during the M exposures of a frame (linear between frames)."""
    As, ys = [], []
    P_same = make_patterns(M, rng)
    for t in range(T):
        P = P_same if mode == "same" else make_patterns(M, rng)
        A = P / 64.0
        if not intra_motion:
            y = A @ frames[t].ravel()
        else:
            nxt = pos[t + 1] if t + 1 < T else pos[t] + (pos[t] - pos[t - 1])
            y = np.empty(M, np.float32)
            chunks = 16
            for c in range(chunks):
                a = (c + 0.5) / chunks
                p = (1 - a) * pos[t] + a * nxt
                img = shift_image(base, p[0], p[1])
                sl = slice(c * M // chunks, (c + 1) * M // chunks)
                y[sl] = A[sl] @ img.ravel()
        if noise > 0:
            y = y + rng.normal(0, noise * y.std(), size=y.shape).astype(np.float32)
        As.append(A.astype(np.float32)); ys.append(y.astype(np.float32))
    return As, ys


def per_frame_tv(As, ys, lam, iters):
    return np.stack([tv_blocks([(As[t], ys[t])], lam, iters=iters) for t in range(T)])


def window_pooled(As, ys, pos, lam, iters, half=1):
    out = []
    for t in range(T):
        ks = [k for k in range(t - half, t + half + 1) if 0 <= k < T]
        blocks = [(warped_operator(As[k], *(pos[k] - pos[t])), ys[k]) for k in ks]
        out.append(tv_blocks(blocks, lam, iters=iters))
    return np.stack(out)


def estimate_positions(recons):
    rel = [np.zeros(2)]
    for t in range(1, T):
        d = np.array(phase_shift(recons[t - 1], recons[t]))
        rel.append(rel[-1] + d)
    rel = np.array(rel)
    return rel - rel.mean(0)


def frame_metrics(est, frames, As, ys):
    return dict(
        psnr=[psnr(est[t], frames[t]) for t in range(T)],
        resid=[meas_residual(As[t], est[t], ys[t]) for t in range(T)],
    )


def run_one(seed, speed_bin, M, mode, noise, lams, iters, variants, intra=False):
    base, frames, pos, speed, ch = make_sequence(seed, speed_bin)
    rng = np.random.default_rng(seed + 99991)
    As, ys = measure(frames, pos, base, M, mode, noise, rng, intra_motion=intra)
    res = dict(seed=seed, speed_bin=speed_bin, speed=speed, digit=ch, M=M, mode=mode,
               noise=noise, intra=intra)
    t0 = time.time()
    if "dgi" in variants:
        est = np.stack([dgi(As[t] * 64, ys[t]) for t in range(T)])
        res["dgi"] = frame_metrics(est, frames, As, ys)
    single = per_frame_tv(As, ys, lams["single"], iters)
    res["single"] = frame_metrics(single, frames, As, ys)
    if "win3" in variants:
        est = window_pooled(As, ys, pos, lams["win3"], iters)
        res["win3_oracle"] = frame_metrics(est, frames, As, ys)
    if "all" in variants:
        est, *_ = pooled_recon(As, ys, pos, lams["all"], iters)
        res["all_oracle"] = frame_metrics(est, frames, As, ys)
    for sig in variants.get("noisy_motion", []):
        r2 = np.random.default_rng(seed + int(sig * 1000))
        noisy = pos + r2.normal(0, sig, pos.shape)
        est, *_ = pooled_recon(As, ys, noisy, lams["all"], iters)
        res[f"all_motionerr{sig}"] = frame_metrics(est, frames, As, ys)
    if "estimated" in variants:
        p = estimate_positions(single)
        err0 = motion_error(p, pos)
        est, canvas, _ = pooled_recon(As, ys, p, lams["all"], iters)
        res["all_est_r0"] = frame_metrics(est, frames, As, ys)
        errs = [err0]
        for r in range(variants["estimated"]):
            shape, origin = canvas_for(p)
            p = refine_shifts_by_measurement(As, ys, canvas, shape, origin, p)
            p = p - p.mean(0)
            errs.append(motion_error(p, pos))
            est, canvas, _ = pooled_recon(As, ys, p, lams["all"], iters)
            res[f"all_est_r{r+1}"] = frame_metrics(est, frames, As, ys)
        res["motion_epe_rounds"] = errs
    if "mdr" in variants:
        from mcgi import mdr_pair, refine_against_canvas
        rel = [np.zeros(2)]
        for t in range(1, T):
            rel.append(rel[-1] + mdr_pair(As[t], ys[t], single[t - 1]))
        p = np.array(rel); p = p - p.mean(0)
        errs = [motion_error(p, pos)]
        est, canvas, _ = pooled_recon(As, ys, p, lams["all"], iters)
        res["mdr_r0"] = frame_metrics(est, frames, As, ys)
        for r in range(variants["mdr"]):
            shape, origin = canvas_for(p)
            p = refine_against_canvas(As, ys, canvas, shape, origin, p)
            p = p - p.mean(0)
            errs.append(motion_error(p, pos))
            est, canvas, _ = pooled_recon(As, ys, p, lams["all"], iters)
            res[f"mdr_r{r+1}"] = frame_metrics(est, frames, As, ys)
        res["mdr_epe_rounds"] = errs
        res["mdr_final"] = res[f"mdr_r{variants['mdr']}"]
    res["seconds"] = time.time() - t0
    return res


def motion_error(p_est, p_true):
    """Error of relative motion (global offset removed)."""
    a = p_est - p_est.mean(0); b = p_true - p_true.mean(0)
    return float(np.linalg.norm(a - b, axis=1).mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--mode", default="diff")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--n-per-bin", type=int, default=6)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--lams", default="")
    ap.add_argument("--intra", action="store_true")
    ap.add_argument("--variants", default="dgi,win3,all,estimated")
    ap.add_argument("--noisy-motion", default="")
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--mdr-rounds", type=int, default=6)
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    if args.stage == "tune":
        # validation sequences: seeds 1000+; choose lambda per method by mean PSNR
        grid = [1e-4, 3e-4, 1e-3, 3e-3]
        seeds = [(1000 + 10 * b + i, b) for b in range(3) for i in range(2)]
        scores = {"single": {}, "win3": {}, "all": {}}
        for lam in grid:
            L = {"single": lam, "win3": lam, "all": lam}
            for s, b in seeds:
                r = run_one(s, b, args.M, args.mode, args.noise, L, args.iters,
                            {"win3": 1, "all": 1}, intra=args.intra)
                for k, key in (("single", "single"), ("win3", "win3_oracle"), ("all", "all_oracle")):
                    scores[k].setdefault(lam, []).append(np.mean(r[key]["psnr"]))
            print(lam, {k: round(float(np.mean(v[lam])), 2) for k, v in scores.items()}, flush=True)
        best = {k: max(v, key=lambda l: np.mean(v[l])) for k, v in scores.items()}
        out.write_text(json.dumps({"best": best, "scores": {k: {str(l): float(np.mean(x)) for l, x in v.items()} for k, v in scores.items()}}, indent=1))
        print("best", best)
    else:
        lams = json.loads(Path(args.lams).read_text())["best"]
        variants = {v: 1 for v in args.variants.split(",") if v}
        if "estimated" in variants:
            variants["estimated"] = 2
        if "mdr" in variants:
            variants["mdr"] = args.mdr_rounds
        if args.noisy_motion:
            variants["noisy_motion"] = [float(s) for s in args.noisy_motion.split(",")]
        rows = []
        for b in range(3):
            for i in range(args.n_per_bin):
                seed = args.seed_base + 100 * b + i
                r = run_one(seed, b, args.M, args.mode, args.noise, lams, args.iters, variants,
                            intra=args.intra)
                rows.append(r)
                summary = {k: round(float(np.mean(v["psnr"])), 2) for k, v in r.items()
                           if isinstance(v, dict)}
                print(b, i, f"{r['seconds']:.0f}s", summary, r.get("mdr_epe_rounds"), flush=True)
                out.write_text(json.dumps(rows))

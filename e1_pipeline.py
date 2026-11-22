"""Experiment E1: how much does motion-compensated measurement pooling add on top of the Transformer?

Stages (each resumable; completed stages are skipped):
  check      assets, configuration, unit checks
  data       generate train/validation target sequences (never the final test)
  coarse     retrain the coarse Transformer under per-frame speckles
  motion     frozen MotionNet vs. a retrained one; keep the lower validation pair EPE
  features   coarse frames, buckets, residuals and motion for train/validation
  cnn        group 2: Transformer + CNN refiner, one training per seed offset
  tune       physical-layer settings, chosen per measurement count on a validation subset
  physcnn    group 5: CNN refiner trained on physical-layer outputs, one training per seed offset
  evaluate   the final test, once: groups 1-5 plus diagnostics, paired statistics, preregistered criteria
  all        everything above in order

  smoke      the whole chain on a few development sequences in runs/e1_smoke (no validation/test data)

Groups (same observations, same test identities):
  1 coarse      Transformer
  2 cnn         Transformer + CNN refiner
  3 phys_est    Transformer + physical layer, motion estimated (MotionNet start + measurement-domain refinement)
  4 phys_true   Transformer + physical layer, true motion (upper bound)
  5 phys_cnn    Transformer + physical layer (estimated motion) + CNN refiner
  diagnostic: phys_own  physical layer with each frame's own buckets only (consistency without pooling)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from e1_physics import LayerConfig, MotionConfig, PoolConfig, pool_reconstruct, positions_from_pairs
from gi_experiment import atomic_torch_save, choose_device, make_bank, runtime_metadata, verify_assets, write_json
from project_paths import CHECKPOINT_DIR, PROJECT_ROOT, project_path, sha256_file
from train_graph_protocol_v2 import IMAGE_SIZE, RATES, SEED, BouncingMovingMNIST, seed_all, speed_bin_name
from train_madygraph_gi_plugin import (
    ResidualRefiner, dgi_from_bucket, gi_forward, metric_rows, physical_residual, relative_measurement_residual,
    train_one,
)

DEFAULT_CONFIG = "configs/e1_motion_pooling.json"
STAGES = ("check", "data", "coarse", "motion", "features", "cnn", "tune", "physcnn", "evaluate")
METRICS = ("mse", "psnr_db", "ssim", "foreground_psnr_db", "temporal_difference_error", "relative_measurement_residual")


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


# ----------------------------------------------------------------------------- configuration and data
def load_config(path):
    cfg = json.loads(project_path(path).read_text(encoding="utf-8"))
    manifest_path = project_path(cfg["split_manifest"])
    if sha256_file(manifest_path) != cfg["split_manifest_sha256"]:
        raise ValueError("E1 split manifest changed; make a new experiment configuration")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    excluded = set(manifest["historically_exposed_test_indices"])
    for role, item in cfg["roles"].items():
        split = manifest["splits"][item["split"]]
        ids = split["mnist_indices"][item["start"]:item["start"] + item["count"]]
        if len(ids) != item["count"]:
            raise ValueError(f"role {role} asks for more identities than {item['split']} has")
        if split["source"] == "official_test" and set(ids) & excluded:
            raise ValueError(f"role {role} contains excluded identities")
    if cfg["pattern_mode"] != "per_frame" or list(cfg["pattern_counts"]) != list(RATES):
        raise ValueError("E1 uses per-frame speckles at 128 and 256 buckets")
    return cfg, manifest


def smoke_config(cfg):
    """Tiny end-to-end run on development identities only."""
    s = copy.deepcopy(cfg)
    s["study_id"] += "_smoke"
    s["run_directory"] = "runs/e1_smoke"
    s["roles"] = {"train": {"split": "development", "start": 0, "count": 24},
                  "validation": {"split": "development", "start": 24, "count": 12},
                  "test": {"split": "development", "start": 36, "count": 12}}
    s["coarse"]["epochs"] = 1
    s["motion_net"]["epochs"] = 1
    s["refiner"]["epochs"] = 1
    s["refiner"]["seed_offsets"] = [0, 1]
    pl = s["physical_layer"]
    pl["tuning_per_speed_bin"] = 2
    pl["batch"] = 4
    pl["fixed"]["iters"] = 10
    pl["fixed"]["rounds"] = 1
    pl["grid"] = {"mu": pl["grid"]["mu"][:2], "tv": pl["grid"]["tv"][:1]}
    s["bootstrap_repeats"] = 200
    return s


def dev_config(cfg):
    """Local rehearsal: real training identities, shorter training, development identities for
    validation and evaluation (the reserved validation/test blocks are not touched)."""
    s = copy.deepcopy(cfg)
    s["study_id"] += "_dev"
    s["run_directory"] = "runs/e1_dev"
    s["roles"] = {"train": dict(cfg["roles"]["train"]),
                  "validation": {"split": "development", "start": 0, "count": 300},
                  "test": {"split": "development", "start": 300, "count": 300}}
    s["coarse"]["epochs"] = 6
    s["motion_net"]["epochs"] = 6
    s["refiner"]["epochs"] = 6
    s["refiner"]["seed_offsets"] = [0]
    s["physical_layer"]["batch"] = 16
    s["bootstrap_repeats"] = 1000
    return s


def role_dataset(cfg, manifest, role):
    item = cfg["roles"][role]
    split = manifest["splits"][item["split"]]
    ids = np.asarray(split["mnist_indices"][item["start"]:item["start"] + item["count"]])
    return BouncingMovingMNIST(PROJECT_ROOT / "data", split["source"] == "official_train", ids,
                               split["trajectory_seed"], interpolation=manifest["data_interpolation"])


def role_data(cfg, manifest, run, role):
    """Target sequences for a role, generated once and cached."""
    path = run / "data" / f"{role}.pt"
    if path.is_file():
        return torch.load(path, map_location="cpu", weights_only=True)
    if role == "test" and not (run / "evaluate" / "STARTED").is_file():
        raise RuntimeError("final test sequences are generated only by the evaluate stage")
    ds = role_dataset(cfg, manifest, role)
    parts = [ds[i] for i in range(len(ds))]
    data = {"target": torch.stack([p[0] for p in parts]), "labels": torch.tensor([p[1] for p in parts]),
            "truth_motion": torch.stack([p[2] for p in parts]),
            "speed": torch.tensor([p[3] for p in parts], dtype=torch.float32),
            "indices": torch.tensor([p[4] for p in parts])}
    atomic_torch_save(data, path)
    log(f"data {role}: {len(ds)} sequences")
    return data


# ----------------------------------------------------------------------------- models
def coarse_model(device):
    from train_spatiotemporal_transformer import MiniSpatioTemporalTransformer
    return MiniSpatioTemporalTransformer().to(device)


def motion_model(device):
    from train_spatial_motion_diagnostic import SpatialMotionNet
    return SpatialMotionNet().to(device)


def dgi_frames(target, bank, rate):
    bucket = gi_forward(target, bank[rate])
    return bucket, dgi_from_bucket(bucket, bank[rate])


def pair_estimate(net, dgi):
    b, t = dgi.shape[:2]
    pairs = torch.cat((dgi[:, :-1], dgi[:, 1:]), dim=2).reshape(b * (t - 1), 2, IMAGE_SIZE, IMAGE_SIZE)
    return net(pairs).reshape(b, t - 1, 2)


def _resume(path, key):
    if not path.is_file():
        return None
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["fingerprint"] != key:
        raise RuntimeError(f"{path} belongs to another configuration")
    return state


def _rng_state(device):
    return {"torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}


def _set_rng(state, device):
    torch.set_rng_state(state["torch_rng"])
    if device.type == "cuda" and state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])


# ----------------------------------------------------------------------------- stages
def stage_check(cfg, manifest, run, device):
    verify_assets()
    for script in ("verify_per_frame_patterns.py", "verify_e1_physics.py"):
        subprocess.run([sys.executable, str(PROJECT_ROOT / script)], cwd=PROJECT_ROOT, check=True)
    write_json(run / "check.json", {"config": cfg, "runtime": runtime_metadata(device)})


def stage_data(cfg, manifest, run, device):
    for role in ("train", "validation"):
        role_data(cfg, manifest, run, role)


@torch.no_grad()
def _val_coarse_mse(model, val, bank, device, batch=16):
    model.eval()
    values = []
    for rate in RATES:
        for s in range(0, len(val["target"]), batch):
            target = val["target"][s:s + batch].to(device)
            _, dgi = dgi_frames(target, bank, rate)
            values.extend((model(dgi) - target).square().mean((1, 2, 3, 4)).cpu().tolist())
    return float(np.mean(values))


def stage_coarse(cfg, manifest, run, device):
    from train_frame_cnn import reconstruction_loss
    from train_mixed_sampling import temporal_error

    out = run / "coarse"
    if (out / "done.json").is_file():
        return
    c = cfg["coarse"]
    key = fingerprint({"coarse": c, "roles": cfg["roles"], "manifest": cfg["split_manifest_sha256"]})
    train, val = role_data(cfg, manifest, run, "train"), role_data(cfg, manifest, run, "validation")
    bank = make_bank(device, per_frame=True)
    seed_all(SEED + 41000)
    model = coarse_model(device)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    history, best, first = [], float("inf"), 1
    state = _resume(out / "last.pt", key)
    if state:
        model.load_state_dict(state["model"]); opt.load_state_dict(state["optimizer"]); _set_rng(state, device)
        history, best, first = state["history"], state["best_val"], state["epoch"] + 1
        log(f"coarse resumed after epoch {state['epoch']}")
    n = len(train["target"])
    for epoch in range(first, c["epochs"] + 1):
        started = time.perf_counter()
        model.train()
        order = torch.randperm(n, generator=torch.Generator().manual_seed(SEED + 41000 + epoch))
        total = 0.0
        for bi, s in enumerate(range(0, n, c["batch"])):
            ids = order[s:s + c["batch"]]
            target = train["target"][ids].to(device)
            rate = RATES[(bi + epoch - 1) % len(RATES)]
            with torch.no_grad():
                _, dgi = dgi_frames(target, bank, rate)
            pred = model(dgi)
            flat = target.reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
            loss = reconstruction_loss(pred.reshape_as(flat), flat) + c["temporal_weight"] * temporal_error(pred, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite coarse loss at epoch {epoch}")
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += float(loss.detach()) * len(ids)
        val_mse = _val_coarse_mse(model, val, bank, device)
        history.append({"epoch": epoch, "train_loss": total / n, "val_mse": val_mse,
                        "seconds": time.perf_counter() - started})
        log(f"coarse epoch {epoch:02d}/{c['epochs']} loss={total / n:.5f} val_mse={val_mse:.6f}")
        if val_mse < best:
            best = val_mse
            atomic_torch_save(model.state_dict(), out / "best.pt")
        atomic_torch_save({"fingerprint": key, "epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(),
                           "history": history, "best_val": best, **_rng_state(device)}, out / "last.pt")
    write_json(out / "done.json", {"fingerprint": key, "history": history, "best_val_mse": best,
                                   "selection": "lowest validation MSE averaged over both bucket counts"})


@torch.no_grad()
def _val_pair_epe(net, val, bank, device, batch=32):
    net.eval()
    out = {}
    for rate in RATES:
        errs = []
        for s in range(0, len(val["target"]), batch):
            target = val["target"][s:s + batch].to(device)
            _, dgi = dgi_frames(target, bank, rate)
            est = pair_estimate(net, dgi)
            errs.extend(torch.linalg.vector_norm(est - val["truth_motion"][s:s + batch].to(device), dim=-1).flatten().cpu().tolist())
        out[rate] = float(np.mean(errs))
    return out


def stage_motion(cfg, manifest, run, device):
    out = run / "motion"
    if (out / "choice.json").is_file():
        return
    c = cfg["motion_net"]
    key = fingerprint({"motion": c, "roles": cfg["roles"], "manifest": cfg["split_manifest_sha256"]})
    train, val = role_data(cfg, manifest, run, "train"), role_data(cfg, manifest, run, "validation")
    shared, per_frame = make_bank(device), make_bank(device, per_frame=True)
    frozen = motion_model(device)
    frozen.load_state_dict(torch.load(CHECKPOINT_DIR / "best_spatial_motion_net.pt", map_location=device, weights_only=True))
    frozen_shared = _val_pair_epe(frozen, val, shared, device)
    frozen_per_frame = _val_pair_epe(frozen, val, per_frame, device)
    log(f"frozen MotionNet validation pair EPE: shared {frozen_shared}, per-frame {frozen_per_frame}")

    seed_all(SEED + 42000)
    net = motion_model(device)
    opt = torch.optim.AdamW(net.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    history, best, first = [], float("inf"), 1
    state = _resume(out / "last.pt", key)
    if state:
        net.load_state_dict(state["model"]); opt.load_state_dict(state["optimizer"]); _set_rng(state, device)
        history, best, first = state["history"], state["best_val"], state["epoch"] + 1
    n = len(train["target"])
    for epoch in range(first, c["epochs"] + 1):
        net.train()
        order = torch.randperm(n, generator=torch.Generator().manual_seed(SEED + 42000 + epoch))
        total = 0.0
        for bi, s in enumerate(range(0, n, c["batch"])):
            ids = order[s:s + c["batch"]]
            target, truth = train["target"][ids].to(device), train["truth_motion"][ids].to(device)
            with torch.no_grad():
                _, dgi = dgi_frames(target, per_frame, RATES[(bi + epoch - 1) % len(RATES)])
            loss = F.smooth_l1_loss(pair_estimate(net, dgi), truth)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            total += float(loss.detach()) * len(ids)
        epe = _val_pair_epe(net, val, per_frame, device)
        score = float(np.mean(list(epe.values())))
        history.append({"epoch": epoch, "train_loss": total / n, "val_pair_epe": epe})
        log(f"motion epoch {epoch:02d}/{c['epochs']} loss={total / n:.5f} val_epe={epe}")
        if score < best:
            best = score
            atomic_torch_save(net.state_dict(), out / "retrained_best.pt")
        atomic_torch_save({"fingerprint": key, "epoch": epoch, "model": net.state_dict(), "optimizer": opt.state_dict(),
                           "history": history, "best_val": best, **_rng_state(device)}, out / "last.pt")
    use_retrained = best < float(np.mean(list(frozen_per_frame.values())))
    write_json(out / "choice.json", {
        "fingerprint": key, "use": "retrained" if use_retrained else "frozen",
        "rule": "keep the network with the lower validation pair EPE averaged over both bucket counts (per-frame speckles)",
        "frozen_val_pair_epe_shared_patterns": frozen_shared, "frozen_val_pair_epe_per_frame": frozen_per_frame,
        "retrained_best_val_pair_epe_mean": best, "history": history,
    })


def load_motion(run, device):
    choice = json.loads((run / "motion" / "choice.json").read_text(encoding="utf-8"))
    net = motion_model(device)
    path = run / "motion" / "retrained_best.pt" if choice["use"] == "retrained" else CHECKPOINT_DIR / "best_spatial_motion_net.pt"
    net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return net.eval()


def load_coarse(run, device):
    model = coarse_model(device)
    model.load_state_dict(torch.load(run / "coarse" / "best.pt", map_location=device, weights_only=True))
    return model.eval()


@torch.no_grad()
def compute_features(data, coarse, motion, bank, device, batch=32):
    out = {}
    for rate in RATES:
        parts = {k: [] for k in ("bucket", "coarse", "residual", "motion")}
        for s in range(0, len(data["target"]), batch):
            target = data["target"][s:s + batch].to(device)
            bucket, dgi = dgi_frames(target, bank, rate)
            x_co = coarse(dgi)
            parts["bucket"].append(bucket.cpu()); parts["coarse"].append(x_co.cpu())
            parts["residual"].append(physical_residual(bucket, x_co, bank[rate]).cpu())
            parts["motion"].append(pair_estimate(motion, dgi).cpu())
        out[rate] = {k: torch.cat(v) for k, v in parts.items()}
        out[rate].update({k: data[k] for k in ("target", "labels", "truth_motion", "speed", "indices")})
    return out


def stage_features(cfg, manifest, run, device):
    bank = make_bank(device, per_frame=True)
    coarse, motion = load_coarse(run, device), load_motion(run, device)
    for role in ("train", "validation"):
        path = run / "features" / f"{role}.pt"
        if not path.is_file():
            atomic_torch_save(compute_features(role_data(cfg, manifest, run, role), coarse, motion, bank, device), path)
            log(f"features {role} saved")


def _refiner_splits(train_feat, val_feat):
    """train_one expects {train,val}[rate] with a normalised residual; scale from the training split."""
    splits, scales = {"train": {}, "val": {}}, {}
    for rate in RATES:
        scale = max(float(train_feat[rate]["residual"].square().mean().sqrt()), 1e-8)
        scales[rate] = scale
        for name, feat in (("train", train_feat), ("val", val_feat)):
            d = dict(feat[rate])
            d["residual"] = (d["residual"] / scale).clamp(-8, 8)
            splits[name][rate] = d
    return splits, scales


def _train_refiners(cfg, run, device, folder, train_feat, val_feat):
    splits, scales = _refiner_splits(train_feat, val_feat)
    write_json(run / folder / "residual_scales.json", {str(k): v for k, v in scales.items()})
    for offset in cfg["refiner"]["seed_offsets"]:
        out = run / folder / f"seed{offset}"
        if (out / "done.json").is_file():
            continue
        out.mkdir(parents=True, exist_ok=True)
        key = fingerprint({"folder": folder, "refiner": cfg["refiner"], "roles": cfg["roles"], "offset": offset})
        _, history, _ = train_one("cnn", splits, device, cfg["refiner"]["epochs"], out,
                                  run_seed=SEED + 910 + offset, training_fingerprint=key)
        write_json(out / "done.json", {"fingerprint": key, "seed_offset": offset, "history": history})
        log(f"{folder} seed {offset} trained")


def stage_cnn(cfg, manifest, run, device):
    train_feat = torch.load(run / "features" / "train.pt", map_location="cpu", weights_only=True)
    val_feat = torch.load(run / "features" / "validation.pt", map_location="cpu", weights_only=True)
    _train_refiners(cfg, run, device, "cnn", train_feat, val_feat)


def pool_config(cfg, params, window=None, refine=True):
    f = cfg["physical_layer"]["fixed"]
    layer = LayerConfig(iters=f["iters"], mu=params["mu"], tv=params["tv"], tv_eps=f["tv_eps"], window=window)
    motion = MotionConfig(rounds=f["rounds"], stages=tuple(tuple(s) for s in f["stages"]), pad=f["pad"])
    return PoolConfig(layer=layer, motion=motion, refine=refine,
                      pair_stages=tuple(tuple(s) for s in f["pair_stages"]))


def run_physics(feat, bank, rate, pcfg, device, batch, use_truth=False, ids=None):
    """Physical layer over a feature split. Returns frames (N,T,1,H,W), positions, per-round EPE (N,R)."""
    d = feat[rate]
    ids = torch.arange(len(d["target"])) if ids is None else ids
    xs, ps, epes = [], [], []
    for s in range(0, len(ids), batch):
        sl = ids[s:s + batch]
        x_co = d["coarse"][sl, :, 0].to(device)
        truth_pairs = d["truth_motion"][sl].to(device)
        start = truth_pairs if use_truth else d["motion"][sl].to(device)
        r = pool_reconstruct(x_co, bank[rate], d["bucket"][sl].to(device), start, pcfg,
                             p_true=positions_from_pairs(truth_pairs))
        xs.append(r["x"][:, :, None].cpu()); ps.append(r["p"].cpu())
        epes.append(torch.stack(r["epe_rounds"], 1).cpu())
    return torch.cat(xs), torch.cat(ps), torch.cat(epes)


def _psnr_per_sequence(pred, target):
    return 10 * torch.log10(1 / (pred - target).square().mean((1, 2, 3, 4)).clamp_min(1e-12))


def tuning_ids(speeds, per_bin):
    bins = [speed_bin_name(float(s)) for s in speeds]
    chosen = []
    for name in ("1-2", "2-4", "4-6"):
        chosen += [i for i, b in enumerate(bins) if b == name][:per_bin]
    return torch.tensor(sorted(chosen))


def stage_tune(cfg, manifest, run, device):
    out = run / "tune"
    if (out / "choice.json").is_file():
        return
    pl = cfg["physical_layer"]
    val_feat = torch.load(run / "features" / "validation.pt", map_location="cpu", weights_only=True)
    bank = make_bank(device, per_frame=True)
    ids = tuning_ids(val_feat[RATES[0]]["speed"], pl["tuning_per_speed_bin"])
    grid = [{"mu": mu, "tv": tv} for mu in pl["grid"]["mu"] for tv in pl["grid"]["tv"]]
    scores = json.loads((out / "scores.json").read_text(encoding="utf-8")) if (out / "scores.json").is_file() else []
    done = {(r["pattern_count"], r["mu"], r["tv"]) for r in scores}
    for rate in RATES:
        target = val_feat[rate]["target"][ids]
        coarse_psnr = float(_psnr_per_sequence(val_feat[rate]["coarse"][ids], target).mean())
        for params in grid:
            if (rate, params["mu"], params["tv"]) in done:
                continue
            started = time.perf_counter()
            x, _, epe = run_physics(val_feat, bank, rate, pool_config(cfg, params), device, pl["batch"], ids=ids)
            row = {"pattern_count": rate, **params, "coarse_psnr_db": coarse_psnr,
                   "psnr_db": float(_psnr_per_sequence(x, target).mean()),
                   "final_epe_px": float(epe[:, -1].mean()), "seconds": time.perf_counter() - started}
            scores.append(row)
            write_json(out / "scores.json", scores)
            log(f"tune M={rate} mu={params['mu']} tv={params['tv']}: {row['psnr_db']:.2f} dB "
                f"(coarse {coarse_psnr:.2f}), epe {row['final_epe_px']:.3f}")
    choice = {}
    for rate in RATES:
        best = max((r for r in scores if r["pattern_count"] == rate), key=lambda r: r["psnr_db"])
        choice[str(rate)] = {"mu": best["mu"], "tv": best["tv"], "validation_psnr_db": best["psnr_db"]}
    write_json(out / "choice.json", {"rule": "highest mean validation PSNR of group 3 (estimated motion) per bucket count",
                                     "tuning_sequences": len(ids), "choice": choice})


def chosen_params(run, rate):
    return json.loads((run / "tune" / "choice.json").read_text(encoding="utf-8"))["choice"][str(rate)]


def phys_features(cfg, run, feat, bank, device):
    """Group-3 outputs reshaped as refiner inputs (coarse := physical-layer frames)."""
    out = {}
    for rate in RATES:
        x, p, _ = run_physics(feat, bank, rate, pool_config(cfg, chosen_params(run, rate)), device, cfg["physical_layer"]["batch"])
        d = dict(feat[rate])
        d["coarse"] = x
        d["residual"] = torch.cat([physical_residual(d["bucket"][s:s + 64].to(device), x[s:s + 64].to(device), bank[rate]).cpu()
                                   for s in range(0, len(x), 64)])
        d["motion"] = p[:, 1:] - p[:, :-1]
        out[rate] = d
    return out


def stage_physcnn(cfg, manifest, run, device):
    if not cfg["refiner"].get("train_group5", True):
        return
    bank = make_bank(device, per_frame=True)
    feats = {}
    for role in ("train", "validation"):
        path = run / "features" / f"{role}_phys.pt"
        if not path.is_file():
            base = torch.load(run / "features" / f"{role}.pt", map_location="cpu", weights_only=True)
            started = time.perf_counter()
            atomic_torch_save(phys_features(cfg, run, base, bank, device), path)
            log(f"physical-layer features {role}: {time.perf_counter() - started:.0f} s")
        feats[role] = torch.load(path, map_location="cpu", weights_only=True)
    _train_refiners(cfg, run, device, "physcnn", feats["train"], feats["validation"])


# ----------------------------------------------------------------------------- final evaluation
def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def _refine(model, feat_rate, scale, device, batch=16):
    preds = []
    for s in range(0, len(feat_rate["target"]), batch):
        c = feat_rate["coarse"][s:s + batch].to(device)
        r = (feat_rate["residual"][s:s + batch].to(device) / scale).clamp(-8, 8)
        preds.append(model(c, r, feat_rate["motion"][s:s + batch].to(device)).clamp(0, 1).cpu())
    return torch.cat(preds)


def _load_refiner(path, device):
    model = ResidualRefiner("cnn").to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.eval()


def _rows(method, seed, pred, d, bank, rate, device, extra=None):
    rows = []
    for s in range(0, len(pred), 64):
        p = pred[s:s + 64].to(device)
        resid = relative_measurement_residual(p, d["bucket"][s:s + 64].to(device), bank[rate]).cpu()
        for i, row in enumerate(metric_rows(p, d["target"][s:s + 64].to(device), d["labels"][s:s + 64], d["indices"][s:s + 64])):
            j = s + i
            rows.append({"pattern_count": rate, "method": method, "seed_offset": seed,
                         "speed_pixels_per_frame": float(d["speed"][j]), "speed_bin": speed_bin_name(float(d["speed"][j])),
                         **row, "relative_measurement_residual": float(resid[i]),
                         **({k: float(v[j]) for k, v in extra.items()} if extra else {})})
    return rows


def _seed_mean(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["pattern_count"], r["method"], r["mnist_index"]), []).append(r)
    out = {}
    for key, g in groups.items():
        out[key] = {**{k: g[0][k] for k in ("label", "speed_bin", "speed_pixels_per_frame")},
                    **{m: float(np.mean([r[m] for r in g])) for m in METRICS}, "seeds": len(g)}
    return out


def paired(rows, means, treatment, control, rate, speed_bin, repeats):
    """Paired bootstrap over identities (seed-averaged); per-seed gains reported separately."""
    ids = sorted(k[2] for k in means if k[0] == rate and k[1] == control
                 and (speed_bin == "all" or means[k]["speed_bin"] == speed_bin))
    a = [means[(rate, treatment, i)] for i in ids]
    b = [means[(rate, control, i)] for i in ids]
    delta = np.array([x["psnr_db"] - y["psnr_db"] for x, y in zip(a, b)])
    rng = np.random.default_rng(SEED + rate + 501)
    dist = delta[rng.integers(0, len(ids), size=(repeats, len(ids)))].mean(1)
    lo, hi = np.percentile(dist, [2.5, 97.5])
    by_seed = {(r["pattern_count"], r["method"], r["seed_offset"], r["mnist_index"]): r["psnr_db"]
               for r in rows if r["pattern_count"] == rate and r["method"] in (treatment, control)}
    seeds = sorted({k[2] for k in by_seed if k[2] >= 0})

    def val(method, o, i):
        return by_seed.get((rate, method, o, i), by_seed.get((rate, method, -1, i)))

    per_seed = [float(np.mean([val(treatment, o, i) - val(control, o, i) for i in ids])) for o in seeds]
    return {"pattern_count": rate, "speed_bin": speed_bin, "comparison": f"{treatment}_vs_{control}", "sequences": len(ids),
            "paired_psnr_gain_mean_db": float(delta.mean()), "psnr_gain_ci95_lower_db": float(lo),
            "psnr_gain_ci95_upper_db": float(hi), "paired_win_rate": float(np.mean(delta > 0)),
            "ssim_gain_mean": float(np.mean([x["ssim"] - y["ssim"] for x, y in zip(a, b)])),
            "measurement_residual_change": float(np.mean([x["relative_measurement_residual"] - y["relative_measurement_residual"]
                                                          for x, y in zip(a, b)])),
            "per_seed_gain_db": per_seed}


def stage_evaluate(cfg, manifest, run, device):
    out = run / "evaluate"
    if (out / "results.json").is_file():
        log("final evaluation already exists; not repeating it")
        return
    for need in ("coarse/done.json", "motion/choice.json", "tune/choice.json"):
        if not (run / need).is_file():
            raise RuntimeError(f"cannot evaluate before {need}")
    seeds = cfg["refiner"]["seed_offsets"]
    groups5 = cfg["refiner"].get("train_group5", True)
    for folder in ("cnn",) + (("physcnn",) if groups5 else ()):
        for o in seeds:
            if not (run / folder / f"seed{o}" / "done.json").is_file():
                raise RuntimeError(f"cannot evaluate before {folder}/seed{o}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "STARTED").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    bank = make_bank(device, per_frame=True)
    test = role_data(cfg, manifest, run, "test")
    coarse, motion = load_coarse(run, device), load_motion(run, device)
    _sync(device); t0 = time.perf_counter()
    feat = compute_features(test, coarse, motion, bank, device)
    _sync(device); timing = {"coarse_and_motion_seconds": time.perf_counter() - t0}
    rows, epe_rounds = [], {}
    batch = cfg["physical_layer"]["batch"]
    for rate in RATES:
        d = feat[rate]
        scale_cnn = json.loads((run / "cnn" / "residual_scales.json").read_text())[str(rate)]
        rows += _rows("coarse", -1, d["coarse"].clamp(0, 1), d, bank, rate, device)
        for o in seeds:
            _sync(device); t0 = time.perf_counter()
            pred = _refine(_load_refiner(run / "cnn" / f"seed{o}" / "best_cnn_refiner.pt", device), d, scale_cnn, device)
            _sync(device); timing[f"cnn_M{rate}_seed{o}_seconds"] = time.perf_counter() - t0
            rows += _rows("cnn", o, pred, d, bank, rate, device)
        params = chosen_params(run, rate)
        _sync(device); t0 = time.perf_counter()
        x_est, p_est, epe = run_physics(feat, bank, rate, pool_config(cfg, params), device, batch)
        _sync(device); timing[f"phys_est_M{rate}_seconds"] = time.perf_counter() - t0
        epe_rounds[rate] = epe.mean(0).tolist()
        rows += _rows("phys_est", -1, x_est, d, bank, rate, device,
                      extra={"motion_epe_start_px": epe[:, 0], "motion_epe_final_px": epe[:, -1]})
        x_true, _, _ = run_physics(feat, bank, rate, pool_config(cfg, params, refine=False), device, batch, use_truth=True)
        rows += _rows("phys_true", -1, x_true, d, bank, rate, device)
        x_own, _, _ = run_physics(feat, bank, rate, pool_config(cfg, params, window=0, refine=False), device, batch)
        rows += _rows("phys_own", -1, x_own, d, bank, rate, device)
        if groups5:
            scale_p = json.loads((run / "physcnn" / "residual_scales.json").read_text())[str(rate)]
            pd = dict(d)
            pd["coarse"] = x_est
            pd["residual"] = torch.cat([physical_residual(d["bucket"][s:s + 64].to(device), x_est[s:s + 64].to(device),
                                                          bank[rate]).cpu() for s in range(0, len(x_est), 64)])
            pd["motion"] = p_est[:, 1:] - p_est[:, :-1]
            for o in seeds:
                pred = _refine(_load_refiner(run / "physcnn" / f"seed{o}" / "best_cnn_refiner.pt", device), pd, scale_p, device)
                rows += _rows("phys_cnn", o, pred, d, bank, rate, device)
        log(f"evaluated M={rate}")

    reps = cfg["bootstrap_repeats"]
    contrasts = [("phys_est", "cnn"), ("phys_est", "coarse"), ("phys_true", "phys_est"), ("cnn", "coarse"),
                 ("phys_own", "coarse"), ("phys_est", "phys_own")]
    if groups5:
        contrasts += [("phys_cnn", "cnn"), ("phys_cnn", "phys_est")]
    means = _seed_mean(rows)
    stats = [paired(rows, means, a, b, rate, sb, reps)
             for rate in RATES for sb in ("1-2", "2-4", "4-6", "all") for a, b in contrasts]

    crit = cfg["criteria"]
    def pick(rate, comp):
        return next(s for s in stats if s["pattern_count"] == rate and s["speed_bin"] == "all" and s["comparison"] == comp)
    checks = {}
    for rate in RATES:
        main, base = pick(rate, "phys_est_vs_cnn"), pick(rate, "phys_est_vs_coarse")
        final_epe = float(np.mean([r["motion_epe_final_px"] for r in rows if r["pattern_count"] == rate and r["method"] == "phys_est"]))
        checks[str(rate)] = {
            "gain_vs_cnn_db": main["paired_psnr_gain_mean_db"], "ci95_lower_db": main["psnr_gain_ci95_lower_db"],
            "ssim_change_vs_cnn": main["ssim_gain_mean"], "measurement_residual_change_vs_cnn": main["measurement_residual_change"],
            "gain_vs_coarse_db": base["paired_psnr_gain_mean_db"], "final_motion_epe_px": final_epe,
            "pass": bool(main["paired_psnr_gain_mean_db"] >= crit["min_gain_vs_cnn_db"] and main["psnr_gain_ci95_lower_db"] > 0
                         and main["ssim_gain_mean"] >= 0 and main["measurement_residual_change"] < 0),
            "stop_condition": bool(base["paired_psnr_gain_mean_db"] < crit["stop_gain_vs_coarse_db"]
                                   and final_epe < crit["stop_motion_epe_px"]),
        }
    verdict = {"primary_pass": all(c["pass"] for c in checks.values()),
               "stop_route_c": all(c["stop_condition"] for c in checks.values()), "by_count": checks}
    summary = []
    for rate in RATES:
        for method in sorted({k[1] for k in means}):
            vals = [v for k, v in means.items() if k[0] == rate and k[1] == method]
            summary.append({"pattern_count": rate, "method": method, "sequences": len(vals),
                            **{f"{m}_mean": float(np.mean([v[m] for v in vals])) for m in METRICS}})
    write_json(out / "rows.json", rows)
    write_json(out / "results.json", {
        "study_id": cfg["study_id"], "verdict": verdict, "criteria": crit, "summary": summary, "paired": stats,
        "motion_epe_by_round_px": {str(k): v for k, v in epe_rounds.items()}, "timing_seconds": timing,
        "test_sequences": len(test["target"]), "chosen_physical_layer": {str(r): chosen_params(run, r) for r in RATES},
        "runtime": runtime_metadata(device),
        "confidence_interval_scope": "paired bootstrap over test identities after averaging refiner seeds; "
                                     "conditional on one coarse Transformer and one motion network",
    })
    log("verdict:", json.dumps(verdict, indent=1))


def stage_package(cfg, manifest, run, device):
    """Zip the run (without regenerable sequence/feature tensors) plus a checksum archive."""
    import zipfile
    if not (run / "evaluate" / "results.json").is_file():
        raise RuntimeError("package after the final evaluation")
    output = PROJECT_ROOT / "dist" / f"gi_results_{run.name}.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(run.rglob("*")):
            rel = path.relative_to(run)
            if path.is_file() and path.suffix != ".tmp" and rel.parts[0] not in ("data", "features"):
                archive.write(path, "gi_results/" + rel.as_posix())
        archive.writestr("gi_results/requested_config.json", json.dumps(cfg, ensure_ascii=False, indent=2))
        archive.write(project_path(cfg["split_manifest"]), "gi_results/prepared_split_manifest.json")
        for name in ("CODE_VERSION.json", "PACKAGE_MANIFEST.json"):
            if (PROJECT_ROOT / name).is_file():
                archive.write(PROJECT_ROOT / name, "gi_results/input_" + name)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip():
            raise RuntimeError("result archive failed its integrity check")
    checksum = output.with_suffix(".sha256.json")
    write_json(checksum, {"archive": output.name, "bytes": output.stat().st_size, "sha256": sha256_file(output)})
    with zipfile.ZipFile(output.with_name(output.stem + "_checksums.zip"), "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(checksum, checksum.name)
    log(f"results packaged: {output} ({output.stat().st_size / 2**20:.1f} MiB)")


# ----------------------------------------------------------------------------- entry point
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=STAGES + ("all", "smoke", "package"))
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--dev", action="store_true", help="local rehearsal in runs/e1_dev (development identities only)")
    args = ap.parse_args()
    cfg, manifest = load_config(args.config)
    if args.stage == "smoke":
        cfg = smoke_config(cfg)
    elif args.dev:
        cfg = dev_config(cfg)
    run = project_path(cfg["run_directory"])
    run.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    log(f"E1 {cfg['study_id']} stage={args.stage} device={device}")
    stages = STAGES if args.stage in ("all", "smoke") else (args.stage,)
    for name in stages:
        started = time.perf_counter()
        globals()[f"stage_{name}"](cfg, manifest, run, device)
        log(f"stage {name} finished in {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()

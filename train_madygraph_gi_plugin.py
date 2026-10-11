"""Pilot a MadyGraph-style residual plugin for per-frame bucket GI.

The current GI simulator measures each frame independently.  Therefore the
SCI re-masking formula is replaced by the per-frame adjoint measurement
residual A^T(y - A x_coarse).  Three refiners receive identical frozen
Transformer coarse reconstructions and residual features:
  * 3D CNN residual head
  * fixed-coordinate spatiotemporal graph
  * estimated-motion-guided spatiotemporal graph
"""

from __future__ import annotations

from project_paths import PROJECT_ROOT, DATA_DIR as PROJECT_DATA_DIR

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from project_paths import (
    CHECKPOINT_DIR, DEFAULT_RUN, MOTION_PATH as FROZEN_MOTION_PATH,
    asset_hashes, code_version, project_path,
)

from train_graph_protocol_v2 import (
    BouncingMovingMNIST,
    IMAGE_SIZE,
    RATES,
    SEED,
    TRAIN_N,
    TEST_N,
    VAL_N,
    make_splits,
    make_train_validation,
    make_test_block,
    seed_all,
    speed_bin_name,
)
from train_spatial_motion_diagnostic import SpatialMotionNet


HERE = Path(__file__).resolve().parent
PROJECT = PROJECT_ROOT
DATA_DIR = PROJECT / "data"
BASE_DIR = CHECKPOINT_DIR
MOTION_PATH = FROZEN_MOTION_PATH
DEFAULT_OUT = DEFAULT_RUN / "seed0"
EPOCHS = 5
BATCH = 8


def csv_write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def metric_rows(pred, target, labels, indices):
    batch, frames, _, height, width = target.shape
    mse = (pred - target).square().mean((1, 2, 3, 4))
    psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))
    from pytorch_msssim import ssim
    score = ssim(
        pred.reshape(batch * frames, 1, height, width),
        target.reshape(batch * frames, 1, height, width),
        data_range=1.0,
        size_average=False,
    ).reshape(batch, frames).mean(1)
    temporal = (
        (pred[:, 1:] - pred[:, :-1]) - (target[:, 1:] - target[:, :-1])
    ).abs().mean((1, 2, 3, 4))
    foreground = target > 0.1
    fg_mse = ((pred - target).square() * foreground).sum((1, 2, 3, 4))
    fg_count = foreground.sum((1, 2, 3, 4)).clamp_min(1)
    fg_psnr = 10 * torch.log10(fg_count / fg_mse.clamp_min(1e-12))
    return [
        {
            "label": int(labels[i]),
            "mnist_index": int(indices[i]),
            "mse": float(mse[i]),
            "psnr_db": float(psnr[i]),
            "ssim": float(score[i]),
            "foreground_psnr_db": float(fg_psnr[i]),
            "temporal_difference_error": float(temporal[i]),
        }
        for i in range(batch)
    ]


def gi_forward(image: torch.Tensor, patterns: torch.Tensor) -> torch.Tensor:
    """Apply the per-frame GI sensing matrix to image sequences."""
    batch, frames, _, height, width = image.shape
    matrix = patterns.reshape(patterns.shape[0], -1) / math.sqrt(height * width)
    return image.reshape(batch, frames, -1) @ matrix.T


def physical_residual(bucket: torch.Tensor, coarse: torch.Tensor, patterns: torch.Tensor):
    """Backproject measured bucket residuals; does not require target images."""
    batch, frames, _, height, width = coarse.shape
    matrix = patterns.reshape(patterns.shape[0], -1) / math.sqrt(height * width)
    coarse_bucket = gi_forward(coarse, patterns)
    residual = ((bucket - coarse_bucket) @ matrix) / patterns.shape[0]
    return residual.reshape(batch, frames, 1, height, width)


def dgi_from_bucket(bucket: torch.Tensor, patterns: torch.Tensor) -> torch.Tensor:
    """DGI from measured buckets using the same P/sqrt(H*W) operator."""
    batch, frames, count = bucket.shape
    height, width = patterns.shape[-2:]
    if count != len(patterns):
        raise ValueError("bucket count does not match patterns")
    raw_bucket = bucket.reshape(batch * frames, count) * math.sqrt(height * width)
    illumination = patterns.sum((1, 2)).clamp_min(1e-8)
    normalized = raw_bucket / illumination[None]
    centered = normalized - normalized.mean(1, keepdim=True)
    centered_patterns = patterns - patterns.mean(0, keepdim=True)
    reconstruction = torch.einsum("nm,mhw->nhw", centered, centered_patterns) / count
    low = reconstruction.amin((-2, -1), keepdim=True)
    high = reconstruction.amax((-2, -1), keepdim=True)
    reconstruction = (reconstruction - low) / (high - low + 1e-8)
    return reconstruction.reshape(batch, frames, 1, height, width)


def relative_measurement_residual(pred, bucket, patterns):
    error = (gi_forward(pred, patterns) - bucket).flatten(1)
    observed = bucket.flatten(1)
    return torch.linalg.vector_norm(error, dim=1) / torch.linalg.vector_norm(
        observed, dim=1
    ).clamp_min(1e-12)


class ResidualRefiner(nn.Module):
    def __init__(self, kind: str, dim: int = 32, blocks: int = 2):
        super().__init__()
        self.kind = kind
        self.enc1 = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=2, padding=1), nn.GELU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, dim, 3, stride=2, padding=1), nn.GELU()
        )
        if kind == "cnn":
            self.temporal = nn.Sequential(
                nn.Conv3d(dim, dim, 3, padding=1), nn.GELU(),
                nn.Conv3d(dim, dim, 3, padding=1), nn.GELU(),
            )
        else:
            self.self_proj = nn.ModuleList([nn.Conv2d(dim, dim, 1) for _ in range(blocks)])
            self.spatial_proj = nn.ModuleList([nn.Conv2d(dim, dim, 1) for _ in range(blocks)])
            self.prev_proj = nn.ModuleList([nn.Conv2d(dim, dim, 1) for _ in range(blocks)])
            self.next_proj = nn.ModuleList([nn.Conv2d(dim, dim, 1) for _ in range(blocks)])
            self.norm = nn.ModuleList([nn.GroupNorm(4, dim) for _ in range(blocks)])
        self.dec1 = nn.Sequential(nn.Conv2d(dim + 18, 24, 3, padding=1), nn.GELU())
        self.dec2 = nn.Sequential(nn.Conv2d(24, 16, 3, padding=1), nn.GELU())
        self.out = nn.Conv2d(16, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, coarse, measurement_residual, pair_motion):
        from train_oracle_motion_tgnn import spatial_neighbor_mean, warp_feature

        batch, frames, _, height, width = coarse.shape
        raw = torch.cat((coarse, measurement_residual), dim=2)
        flat = raw.reshape(batch * frames, 2, height, width)
        shallow = self.enc1(flat)
        feature = self.enc2(shallow).reshape(batch, frames, -1, height // 4, width // 4)

        if self.kind == "cnn":
            feature = feature.permute(0, 2, 1, 3, 4)
            feature = feature + self.temporal(feature)
            feature = feature.permute(0, 2, 1, 3, 4)
        else:
            if self.kind == "fixed":
                pair_motion = torch.zeros_like(pair_motion)
            for block in range(len(self.norm)):
                # Vectorize over batch and frame; mathematically this is the
                # same per-frame neighbor aggregation, without Python frame loops.
                current = feature.reshape(batch * frames, -1, height // 4, width // 4)
                spatial = spatial_neighbor_mean(current)
                previous = feature[:, :1]
                following = feature[:, -1:]
                if frames > 1:
                    prev_warped = warp_feature(
                        feature[:, :-1].reshape(batch * (frames - 1), -1, height // 4, width // 4),
                        pair_motion[:, :, 0].reshape(-1),
                        pair_motion[:, :, 1].reshape(-1),
                    ).reshape(batch, frames - 1, -1, height // 4, width // 4)
                    next_warped = warp_feature(
                        feature[:, 1:].reshape(batch * (frames - 1), -1, height // 4, width // 4),
                        -pair_motion[:, :, 0].reshape(-1),
                        -pair_motion[:, :, 1].reshape(-1),
                    ).reshape(batch, frames - 1, -1, height // 4, width // 4)
                    previous = torch.cat((previous, prev_warped), dim=1)
                    following = torch.cat((next_warped, following), dim=1)
                previous = previous.reshape_as(current)
                following = following.reshape_as(current)
                update = self.self_proj[block](current)
                update = update + self.spatial_proj[block](spatial)
                update = update + self.prev_proj[block](previous)
                update = update + self.next_proj[block](following)
                feature = (F.gelu(self.norm[block](update)) + current).reshape(
                    batch, frames, -1, height // 4, width // 4
                )

        decoded = F.interpolate(
            feature.reshape(batch * frames, -1, height // 4, width // 4),
            size=(height, width), mode="bilinear", align_corners=False,
        )
        shallow = F.interpolate(shallow, size=(height, width), mode="bilinear", align_corners=False)
        decoded = self.dec1(torch.cat((decoded, flat, shallow), dim=1))
        delta = self.out(self.dec2(decoded)).reshape_as(coarse)
        return coarse + delta


def precompute(loader, transformer, motion_net, bank, device):
    out = {rate: {k: [] for k in ("target", "bucket", "coarse", "residual", "motion", "truth_motion", "labels", "indices", "speed")} for rate in RATES}
    transformer.eval()
    motion_net.eval()
    with torch.inference_mode():
        for target, labels, truth_motion, speeds, indices in loader:
            target_gpu = target.to(device)
            for rate in RATES:
                bucket = gi_forward(target_gpu, bank[rate])  # simulator only; inference supplies measured bucket
                coarse_dgi = dgi_from_bucket(bucket, bank[rate])
                coarse = transformer(coarse_dgi)
                residual = physical_residual(bucket, coarse, bank[rate])
                b, t = coarse_dgi.shape[:2]
                pairs = torch.cat((coarse_dgi[:, :-1], coarse_dgi[:, 1:]), dim=2)
                estimated = motion_net(pairs.reshape(b * (t - 1), 2, IMAGE_SIZE, IMAGE_SIZE))
                estimated = estimated.reshape(b, t - 1, 2)
                dest = out[rate]
                dest["target"].append(target.cpu())
                dest["bucket"].append(bucket.cpu())
                dest["coarse"].append(coarse.cpu())
                dest["residual"].append(residual.cpu())
                dest["motion"].append(estimated.cpu())
                dest["truth_motion"].append(truth_motion.cpu())
                dest["labels"].append(labels.cpu())
                dest["indices"].append(indices.cpu())
                dest["speed"].append(speeds.cpu())
    return {
        rate: {key: torch.cat(parts, dim=0) for key, parts in values.items()}
        for rate, values in out.items()
    }


def batches(data, order, start, batch_size, device):
    ids = order[start:start + batch_size]
    return {k: v[ids].to(device) for k, v in data.items()}


def evaluate_model(model, splits, device, motion_key="motion", bank=None):
    rows = []
    model.eval()
    with torch.inference_mode():
        for rate in RATES:
            data = splits["test"][rate]
            for start in range(0, len(data["target"]), 4):
                batch = {k: v[start:start + 4].to(device) for k, v in data.items()}
                pred = model(batch["coarse"], batch["residual"], batch[motion_key])
                pred = pred.clamp(0, 1)
                residuals = relative_measurement_residual(pred, batch["bucket"], bank[rate]) if bank else None
                for i, row in enumerate(metric_rows(
                    pred, batch["target"], batch["labels"], batch["indices"]
                )):
                    rows.append({
                        "pattern_count": rate,
                        "speed_pixels_per_frame": float(batch["speed"][i]),
                        "speed_bin": speed_bin_name(float(batch["speed"][i])),
                        **row,
                        **({"relative_measurement_residual": float(residuals[i])} if residuals is not None else {}),
                    })
    return rows


def train_one(kind, splits, device, epochs, output, run_seed=None):
    from train_frame_cnn import reconstruction_loss
    from train_mixed_sampling import temporal_error
    from pytorch_msssim import ssim

    if run_seed is not None:
        seed_all(run_seed)
    order_seed = (run_seed if run_seed is not None else torch.initial_seed()) + 100000
    order_generator = torch.Generator().manual_seed(order_seed)
    model = ResidualRefiner(kind).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    best_val = float("inf")
    best_path = output / f"best_{kind}_refiner.pt"
    history = []
    train_n = len(splits["train"][RATES[0]]["target"])
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(train_n, generator=order_generator)
        total = count = 0
        rate_offset = epoch % len(RATES)
        for batch_index, start in enumerate(range(0, train_n, BATCH)):
            rate = RATES[(batch_index + rate_offset) % len(RATES)]
            batch = batches(splits["train"][rate], order, start, BATCH, device)
            model_motion = batch["truth_motion"] if kind == "oracle" else batch["motion"]
            pred = model(batch["coarse"], batch["residual"], model_motion)
            flat_target = batch["target"].reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
            loss = reconstruction_loss(
                pred.reshape_as(flat_target), flat_target
            ) + 0.15 * temporal_error(pred, batch["target"])
            loss = loss + 0.10 * (
                1.0 - ssim(
                    pred.reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE),
                    flat_target,
                    data_range=1.0,
                    size_average=True,
                )
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n = len(batch["target"])
            total += float(loss.detach()) * n
            count += n

        model.eval()
        vals = []
        val_ssims = []
        with torch.inference_mode():
            for rate in RATES:
                data = splits["val"][rate]
                for start in range(0, len(data["target"]), BATCH):
                    batch = {k: v[start:start + BATCH].to(device) for k, v in data.items()}
                    model_motion = batch["truth_motion"] if kind == "oracle" else batch["motion"]
                    pred = model(batch["coarse"], batch["residual"], model_motion)
                    vals.extend((pred - batch["target"]).square().mean((1, 2, 3, 4)).cpu().tolist())
                    val_ssims.extend(ssim(
                        pred.reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE),
                        batch["target"].reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE),
                        data_range=1.0,
                        size_average=False,
                    ).reshape(len(pred), -1).mean(1).cpu().tolist())
        val_mse = float(np.mean(vals))
        val_ssim = float(np.mean(val_ssims))
        val_score = val_mse + 0.02 * (1.0 - val_ssim)
        history.append({"epoch": epoch, "train_loss": total / count, "val_mse": val_mse, "val_ssim": val_ssim, "val_selection_score": val_score,
                        "minibatch_order_sha256": hashlib.sha256(order.numpy().tobytes()).hexdigest()})
        print(f"{kind} epoch={epoch:02d}/{epochs} loss={total/count:.5f} val_mse={val_mse:.6f} val_ssim={val_ssim:.4f}", flush=True)
        if val_score < best_val:
            best_val = val_score
            torch.save(model.state_dict(), best_path)
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    return model, history, best_path


def summarize(rows):
    results = []
    methods = sorted({row["method"] for row in rows})
    for rate in RATES:
        for speed_bin in ("1-2", "2-4", "4-6", "all"):
            for method in methods:
                group = [r for r in rows if r["pattern_count"] == rate and r["method"] == method
                         and (speed_bin == "all" or r["speed_bin"] == speed_bin)]
                if not group:
                    continue
                result = {"pattern_count": rate, "speed_bin": speed_bin, "method": method, "sequences": len(group)}
                for key in ("mse", "psnr_db", "ssim", "foreground_psnr_db", "temporal_difference_error", "relative_measurement_residual"):
                    if not all(key in row for row in group):
                        continue
                    values = np.asarray([r[key] for r in group], dtype=float)
                    result[f"{key}_mean"] = float(values.mean())
                    result[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                results.append(result)
    return results


def main():
    from gi_experiment import choose_device, load_frozen, make_bank, runtime_metadata, verify_assets, write_json

    parser = argparse.ArgumentParser(description="Train GI refiners; final-test evaluation is a separate command")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--variants", default="cnn,fixed,motion")
    parser.add_argument("--test-block", type=int, default=3)
    parser.add_argument("--train-limit", type=int, default=TRAIN_N)
    parser.add_argument("--val-limit", type=int, default=VAL_N)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--skip-test", dest="skip_test", action="store_true")
    mode.add_argument("--evaluate-test", dest="skip_test", action="store_false", help="Legacy explicit test evaluation; do not use for the reserved final block")
    parser.set_defaults(skip_test=True)
    args = parser.parse_args()
    kinds = [kind.strip() for kind in args.variants.split(",") if kind.strip()]
    if not kinds or len(kinds) != len(set(kinds)) or set(kinds) - {"cnn", "fixed", "motion", "oracle"}:
        parser.error("variants must be distinct values from cnn,fixed,motion,oracle")
    if args.epochs < 1 or not 1 <= args.train_limit <= TRAIN_N or not 1 <= args.val_limit <= VAL_N:
        parser.error("invalid epochs or train/validation limits")
    if args.seed_offset < 0:
        parser.error("seed offset must be nonnegative")
    if not args.skip_test and args.test_block >= 3:
        parser.error("reserved final blocks must be evaluated with evaluate_madygraph_gi_final.py")
    verify_assets()
    output = project_path(args.output_dir or (DEFAULT_RUN / f"seed{args.seed_offset}"))
    if (output / "results.json").exists():
        parser.error("completed training already exists; choose another output directory")
    output.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    run_seed = SEED + 910 + args.seed_offset
    seed_all(run_seed)
    print(f"device={device} seed_offset={args.seed_offset} test_evaluation={not args.skip_test}", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_set, val_set = make_train_validation(PROJECT)
    train_ids = train_set.indices[:args.train_limit].tolist()
    val_ids = val_set.indices[:args.val_limit].tolist()
    loaders = {
        "train": DataLoader(Subset(train_set, range(args.train_limit)), batch_size=BATCH, shuffle=False),
        "val": DataLoader(Subset(val_set, range(args.val_limit)), batch_size=BATCH, shuffle=False),
    }
    if not args.skip_test:
        loaders["test"] = DataLoader(make_test_block(PROJECT, args.test_block), batch_size=4, shuffle=False)
    transformer, motion_net = load_frozen(device)
    bank = make_bank(device)
    start_time = time.perf_counter()
    splits = {}
    for split_name, loader in loaders.items():
        print(f"precompute={split_name}", flush=True)
        splits[split_name] = precompute(loader, transformer, motion_net, bank, device)
    residual_scales = {}
    for rate in RATES:
        scale = max(float(splits["train"][rate]["residual"].square().mean().sqrt()), 1e-8)
        residual_scales[rate] = scale
        for split_name in splits:
            splits[split_name][rate]["residual"] = (splits[split_name][rate]["residual"] / scale).clamp(-8, 8)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    precompute_seconds = time.perf_counter() - start_time
    manifest = {
        "schema_version": 2, "phase": "training_only" if args.skip_test else "legacy_training_and_test",
        "task": "Motion-guided refinement for per-frame bucket GI",
        "data_seed": SEED, "refiner_run_seed": run_seed, "seed_offset": args.seed_offset,
        "rates": list(RATES), "resolution": [IMAGE_SIZE, IMAGE_SIZE], "frames": 8,
        "train_val_test_sequences": [args.train_limit, args.val_limit, 0 if args.skip_test else TEST_N],
        "training_mnist_indices": train_ids, "validation_mnist_indices": val_ids,
        "test_evaluated": not args.skip_test, "reserved_test_block": args.test_block,
        "coarse_backbone": "checkpoints/best_transformer.pt",
        "motion_estimator": "checkpoints/best_spatial_motion_net.pt",
        "checkpoint_sha256": asset_hashes(),
        "measurement_operator": "P/sqrt(H*W); per-frame bucket observations",
        "measurement_residual_train_rms_by_rate": residual_scales,
        "refiner_objective": "reconstruction + 0.15 temporal difference + 0.10*(1-SSIM)",
        "checkpoint_selection": "validation MSE + 0.02*(1-validation SSIM)",
        "graph_variants": kinds, "epochs": args.epochs, "batch_size": BATCH,
        "minibatch_order_seed": run_seed + 100000,
        "fairness": "each variant resets initialization seed; an independent CPU generator produces identical epoch orders",
        "runtime": runtime_metadata(device),
    }
    write_json(output / "data_manifest.json", manifest)
    all_rows = []
    if not args.skip_test:
        for rate in RATES:
            data = splits["test"][rate]
            prediction = data["coarse"].clamp(0, 1)
            residuals = relative_measurement_residual(prediction.to(device), data["bucket"].to(device), bank[rate])
            for i, row in enumerate(metric_rows(prediction, data["target"], data["labels"], data["indices"])):
                all_rows.append({"pattern_count": rate, "speed_pixels_per_frame": float(data["speed"][i]),
                                 "speed_bin": speed_bin_name(float(data["speed"][i])), "method": "TransformerCoarse",
                                 **row, "relative_measurement_residual": float(residuals[i])})
    histories, params = {}, {}
    for kind in kinds:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_start = time.perf_counter()
        model, histories[kind], checkpoint = train_one(kind, splits, device, args.epochs, output, run_seed=run_seed)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        params[kind] = {"parameters": sum(p.numel() for p in model.parameters()),
                        "checkpoint": checkpoint.relative_to(PROJECT_ROOT).as_posix(),
                        "training_seconds": time.perf_counter() - train_start}
        csv_write(output / f"history_{kind}.csv", histories[kind])
        if not args.skip_test:
            rows = evaluate_model(model, splits, device, motion_key="truth_motion" if kind == "oracle" else "motion", bank=bank)
            all_rows.extend({**row, "method": {"cnn":"CNNResidual", "fixed":"FixedGraphResidual", "motion":"MotionGraphResidual", "oracle":"OracleMotionGraphResidual"}[kind]} for row in rows)
        del model
    orders = [[epoch["minibatch_order_sha256"] for epoch in histories[kind]] for kind in kinds]
    if any(order != orders[0] for order in orders[1:]):
        raise RuntimeError("Variant minibatch orders differ")
    summary = summarize(all_rows) if all_rows else []
    if all_rows:
        csv_write(output / "benchmark_detail.csv", all_rows)
        csv_write(output / "benchmark_summary.csv", summary)
    result = {"manifest": manifest, "histories": histories, "parameters": params, "summary": summary,
              "precompute_seconds": precompute_seconds, "elapsed_seconds": time.perf_counter() - start_time,
              "peak_cuda_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None}
    write_json(output / "results.json", result)
    print(f"saved={output} elapsed_seconds={result['elapsed_seconds']:.2f}", flush=True)


if __name__ == "__main__":
    main()

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

import argparse
import csv
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
from torch.utils.data import DataLoader

from train_graph_protocol_v2 import (
    BouncingMovingMNIST,
    IMAGE_SIZE,
    RATES,
    SEED,
    TEST_N,
    VAL_N,
    make_splits,
    seed_all,
    speed_bin_name,
)
from train_spatial_motion_diagnostic import SpatialMotionNet


HERE = Path(__file__).resolve().parent
PROJECT = Path(r"D:\download\DynGhost-mini-repro")
DATA_DIR = PROJECT / "data"
BASE_DIR = HERE / "graph_protocol_v2_trained"
MOTION_PATH = HERE / "spatial_motion_outputs" / "best_spatial_motion_net.pt"
DEFAULT_OUT = HERE / "madygraph_gi_plugin_outputs"
EPOCHS = 10
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
    from train_mixed_sampling import coarse_sequence

    out = {rate: {k: [] for k in ("target", "coarse", "residual", "motion", "truth_motion", "labels", "indices", "speed")} for rate in RATES}
    transformer.eval()
    motion_net.eval()
    with torch.inference_mode():
        for target, labels, truth_motion, speeds, indices in loader:
            target_gpu = target.to(device)
            for rate in RATES:
                coarse_dgi = coarse_sequence(target_gpu, bank[rate])
                coarse = transformer(coarse_dgi)
                bucket = gi_forward(target_gpu, bank[rate])  # simulator only; inference supplies measured bucket
                residual = physical_residual(bucket, coarse, bank[rate])
                b, t = coarse_dgi.shape[:2]
                pairs = torch.cat((coarse_dgi[:, :-1], coarse_dgi[:, 1:]), dim=2)
                estimated = motion_net(pairs.reshape(b * (t - 1), 2, IMAGE_SIZE, IMAGE_SIZE))
                estimated = estimated.reshape(b, t - 1, 2)
                dest = out[rate]
                dest["target"].append(target.cpu())
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


def evaluate_model(model, splits, device, motion_key="motion"):
    rows = []
    model.eval()
    with torch.inference_mode():
        for rate in RATES:
            data = splits["test"][rate]
            for start in range(0, len(data["target"]), 4):
                batch = {k: v[start:start + 4].to(device) for k, v in data.items()}
                pred = model(batch["coarse"], batch["residual"], batch[motion_key])
                pred = pred.clamp(0, 1)
                for i, row in enumerate(metric_rows(
                    pred, batch["target"], batch["labels"], batch["indices"]
                )):
                    rows.append({
                        "pattern_count": rate,
                        "speed_pixels_per_frame": float(batch["speed"][i]),
                        "speed_bin": speed_bin_name(float(batch["speed"][i])),
                        **row,
                    })
    return rows


def train_one(kind, splits, device, epochs, output):
    from train_frame_cnn import reconstruction_loss
    from train_mixed_sampling import temporal_error
    from pytorch_msssim import ssim

    model = ResidualRefiner(kind).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    best_val = float("inf")
    best_path = output / f"best_{kind}_refiner.pt"
    history = []
    train_n = len(splits["train"][RATES[0]]["target"])
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(train_n)
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
        history.append({"epoch": epoch, "train_loss": total / count, "val_mse": val_mse, "val_ssim": val_ssim, "val_selection_score": val_score})
        print(f"{kind} epoch={epoch:02d}/{epochs} loss={total/count:.5f} val_mse={val_mse:.6f} val_ssim={val_ssim:.4f}", flush=True)
        if val_score < best_val:
            best_val = val_score
            torch.save(model.state_dict(), best_path)
    model.load_state_dict(torch.load(best_path, map_location=device))
    return model, history, best_path


def summarize(rows):
    results = []
    methods = sorted({row["method"] for row in rows})
    for rate in RATES:
        for speed_bin in ("1-2", "2-4", "4-6", "all"):
            for method in methods:
                group = [r for r in rows if r["pattern_count"] == rate and r["method"] == method
                         and (speed_bin == "all" or r["speed_bin"] == speed_bin)]
                result = {"pattern_count": rate, "speed_bin": speed_bin, "method": method, "sequences": len(group)}
                for key in ("psnr_db", "ssim", "foreground_psnr_db", "temporal_difference_error"):
                    values = np.asarray([r[key] for r in group], dtype=float)
                    result[f"{key}_mean"] = float(values.mean())
                    result[f"{key}_std"] = float(values.std(ddof=1))
                results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--seed-offset", type=int, default=0,
        help="vary refiner initialization and minibatch order while holding data/masks fixed",
    )
    parser.add_argument(
        "--variants", default="cnn,fixed,motion",
        help="comma-separated refiners: cnn, fixed, motion, oracle",
    )
    parser.add_argument(
        "--test-block", type=int, default=1,
        help="held-out 120-sequence block after validation in the deterministic MNIST test permutation",
    )
    args = parser.parse_args()
    sys.path[:0] = [str(PROJECT), str(HERE)]
    from mini_repro_dynghost import make_patterns
    from train_spatiotemporal_transformer import MiniSpatioTemporalTransformer

    run_seed = SEED + 910 + args.seed_offset
    seed_all(run_seed)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    train_set, val_set, _ = make_splits(PROJECT)
    split_rng = np.random.default_rng(SEED + 700)
    split_rng.permutation(60000)
    heldout_indices = split_rng.permutation(10000)
    test_start = VAL_N + TEST_N * args.test_block
    test_set = BouncingMovingMNIST(
        PROJECT / "data", False,
        heldout_indices[test_start:test_start + TEST_N],
        SEED + 703 + args.test_block,
    )
    loaders = {
        "train": DataLoader(train_set, batch_size=8, shuffle=False),
        "val": DataLoader(val_set, batch_size=8, shuffle=False),
        "test": DataLoader(test_set, batch_size=4, shuffle=False),
    }
    bank = {rate: make_patterns(rate, IMAGE_SIZE, IMAGE_SIZE, seed=SEED).to(device) for rate in RATES}
    transformer = MiniSpatioTemporalTransformer().to(device)
    transformer.load_state_dict(torch.load(BASE_DIR / "best_transformer.pt", map_location=device))
    motion_net = SpatialMotionNet().to(device)
    motion_net.load_state_dict(torch.load(MOTION_PATH, map_location=device))
    for parameter in transformer.parameters():
        parameter.requires_grad_(False)
    for parameter in motion_net.parameters():
        parameter.requires_grad_(False)

    start_time = time.time()
    splits = {}
    for split_name, loader in loaders.items():
        print(f"precompute={split_name}", flush=True)
        splits[split_name] = precompute(loader, transformer, motion_net, bank, device)
    residual_scales = {}
    for rate in RATES:
        scale = float(splits["train"][rate]["residual"].square().mean().sqrt())
        residual_scales[rate] = scale
        for split_name in splits:
            splits[split_name][rate]["residual"] = (
                splits[split_name][rate]["residual"] / max(scale, 1e-8)
            ).clamp(-8, 8)
    manifest = {
        "paper_reference": "Lu et al., Motion-aware Dynamic Graph Neural Network for Video Compressive Sensing, TPAMI 2024",
        "task": "MadyGraph-inspired residual plugin for per-frame bucket GI; not a video-SCI reproduction",
        "data_seed": SEED,
        "refiner_run_seed": run_seed,
        "seed_offset": args.seed_offset,
        "rates": list(RATES),
        "train_val_test_sequences": [len(train_set), len(val_set), len(test_set)],
        "test_block": args.test_block,
        "data_split": f"same training/validation identities; deterministic held-out block indices {test_start}:{test_start + TEST_N} in the MNIST test permutation",
        "coarse_backbone": str(BASE_DIR / "best_transformer.pt"),
        "motion_estimator": str(MOTION_PATH),
        "gi_measurement_consistency": "A^T(y_t-Ax_coarse_t), scaled by one per-rate RMS from training to preserve per-frame confidence",
        "measurement_residual_train_rms_by_rate": residual_scales,
        "refiner_objective": "reconstruction + 0.15 temporal difference + 0.10*(1-SSIM); checkpoint selected on validation MSE + 0.02*(1-validation SSIM)",
        "graph_variants": args.variants.split(","),
        "motion_source_at_test": "simulator ground-truth displacement for oracle; estimated from adjacent DGI frames otherwise",
        "epochs": args.epochs,
    }
    (output / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    all_rows = []
    histories = {}
    params = {}
    baseline_rows = []
    for rate in RATES:
        data = splits["test"][rate]
        rows = metric_rows(data["coarse"], data["target"], data["labels"], data["indices"])
        for i, row in enumerate(rows):
            baseline_rows.append({
                "pattern_count": rate,
                "speed_pixels_per_frame": float(data["speed"][i]),
                "speed_bin": speed_bin_name(float(data["speed"][i])),
                "method": "TransformerCoarse",
                **row,
            })
    all_rows.extend(baseline_rows)

    kinds = [kind.strip() for kind in args.variants.split(",") if kind.strip()]
    if not kinds or set(kinds) - {"cnn", "fixed", "motion", "oracle"}:
        raise ValueError(f"unsupported variants: {kinds}")
    for kind in kinds:
        model, histories[kind], checkpoint = train_one(kind, splits, device, args.epochs, output)
        params[kind] = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "checkpoint": str(checkpoint),
        }
        rows = evaluate_model(
            model, splits, device,
            motion_key="truth_motion" if kind == "oracle" else "motion",
        )
        for row in rows:
            all_rows.append({**row, "method": {"cnn": "CNNResidual", "fixed": "FixedGraphResidual", "motion": "MotionGraphResidual", "oracle": "OracleMotionGraphResidual"}[kind]})

    summary = summarize(all_rows)
    index = {(r["pattern_count"], r["method"], r["mnist_index"]): r for r in all_rows}
    gains = []
    for rate in RATES:
        for method in ({"cnn": "CNNResidual", "fixed": "FixedGraphResidual", "motion": "MotionGraphResidual", "oracle": "OracleMotionGraphResidual"}[kind] for kind in kinds):
            for speed_bin in ("1-2", "2-4", "4-6", "all"):
                ids = [r["mnist_index"] for r in all_rows if r["pattern_count"] == rate and r["method"] == "TransformerCoarse"
                       and (speed_bin == "all" or r["speed_bin"] == speed_bin)]
                delta = np.asarray([index[(rate, method, i)]["psnr_db"] - index[(rate, "TransformerCoarse", i)]["psnr_db"] for i in ids])
                gains.append({
                    "pattern_count": rate,
                    "speed_bin": speed_bin,
                    "method": method,
                    "paired_psnr_gain_mean_db": float(delta.mean()),
                    "paired_psnr_gain_std_db": float(delta.std(ddof=1)),
                    "paired_win_rate": float(np.mean(delta > 0)),
                })
    csv_write(output / "benchmark_detail.csv", all_rows)
    csv_write(output / "benchmark_summary.csv", summary)
    csv_write(output / "paired_gains.csv", gains)
    result = {
        "manifest": manifest,
        "histories": histories,
        "parameters": params,
        "summary": summary,
        "paired_gains": gains,
        "elapsed_seconds": time.time() - start_time,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for row in summary:
        if row["speed_bin"] == "all":
            print(
                f"rate={row['pattern_count']} method={row['method']} "
                f"PSNR={row['psnr_db_mean']:.2f} SSIM={row['ssim_mean']:.3f} "
                f"temporal={row['temporal_difference_error_mean']:.4f}", flush=True
            )
    print(f"saved={output}", flush=True)


if __name__ == "__main__":
    main()

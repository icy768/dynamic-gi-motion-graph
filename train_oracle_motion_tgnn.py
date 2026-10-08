import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import DataLoader

from mini_repro_dynghost import IMAGE_SIZE, SEED, make_patterns
from train_frame_cnn import reconstruction_loss
from train_mixed_sampling import RATES, coarse_sequence, stats, temporal_error
from train_patch_tgnn import measure_efficiency
from train_spatiotemporal_transformer import MiniSpatioTemporalTransformer
from train_temporal_convgru import MovingMNISTSequences


TRAIN_N, VAL_N, TEST_N = 800, 120, 100
BATCH, EPOCHS, LR = 8, 10, 5e-4
DIM = 32
OUT = "oracle_motion_tgnn_outputs"
TRANSFORMER_DIR = "spatiotemporal_transformer_outputs"


class MovingMNISTWithMotion(MovingMNISTSequences):
    """Return the known frame-to-frame translation used by the simulator."""

    def __getitem__(self, index):
        frames, label = super().__getitem__(index)
        dx = self.speed[index] * math.cos(self.angle[index])
        dy = self.speed[index] * math.sin(self.angle[index])
        return frames, label, torch.tensor([dx, dy], dtype=torch.float32)


def warp_feature(feature, dx, dy):
    """Forward-warp a feature map using bilinear edge weights."""
    batch = feature.shape[0]
    theta = torch.zeros(batch, 2, 3, device=feature.device, dtype=feature.dtype)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = -2.0 * dx.to(feature.dtype) / IMAGE_SIZE
    theta[:, 1, 2] = -2.0 * dy.to(feature.dtype) / IMAGE_SIZE
    grid = F.affine_grid(theta, feature.shape, align_corners=False)
    return F.grid_sample(feature, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def spatial_neighbor_mean(feature):
    """Four-neighbor graph aggregation on the regular feature grid."""
    channels = feature.shape[1]
    kernel = torch.zeros(channels, 1, 3, 3, device=feature.device, dtype=feature.dtype)
    kernel[:, 0, 0, 1] = 1.0
    kernel[:, 0, 1, 0] = 1.0
    kernel[:, 0, 1, 2] = 1.0
    kernel[:, 0, 2, 1] = 1.0
    summed = F.conv2d(feature, kernel, padding=1, groups=channels)
    degree_kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        device=feature.device, dtype=feature.dtype,
    )[None, None]
    degree = F.conv2d(torch.ones_like(feature[:, :1]), degree_kernel, padding=1)
    return summed / degree.clamp_min(1.0)


class OracleMotionGraphBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.self_proj = nn.Conv2d(dim, dim, 1)
        self.spatial_proj = nn.Conv2d(dim, dim, 1)
        self.prev_proj = nn.Conv2d(dim, dim, 1)
        self.next_proj = nn.Conv2d(dim, dim, 1)
        self.norm = nn.GroupNorm(4, dim)

    def forward(self, sequence, motion):
        batch, frames, channels, height, width = sequence.shape
        dx, dy = motion[:, 0], motion[:, 1]
        updates = []
        for frame in range(frames):
            current = sequence[:, frame]
            spatial = spatial_neighbor_mean(current)
            if frame == 0:
                previous = current
            else:
                previous = warp_feature(sequence[:, frame - 1], dx, dy)
            if frame + 1 == frames:
                following = current
            else:
                following = warp_feature(sequence[:, frame + 1], -dx, -dy)
            update = self.self_proj(current)
            update = update + self.spatial_proj(spatial)
            update = update + self.prev_proj(previous) + self.next_proj(following)
            updates.append(F.gelu(self.norm(update)) + current)
        return torch.stack(updates, dim=1)


class OracleMotionTGNN(nn.Module):
    """Diagnostic graph model using simulator-provided motion, not a deployable model."""

    def __init__(self, dim=DIM, blocks=2):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, dim, 3, stride=2, padding=1), nn.ReLU(inplace=True)
        )
        self.blocks = nn.ModuleList([OracleMotionGraphBlock(dim) for _ in range(blocks)])
        self.up1 = nn.ConvTranspose2d(dim, 16, 4, stride=2, padding=1)
        self.fuse1 = nn.Sequential(
            nn.Conv2d(32, 24, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(24, 16, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1)
        self.out = nn.Sequential(
            nn.Conv2d(9, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def forward(self, x, motion):
        batch, frames, channels, height, width = x.shape
        flat = x.reshape(batch * frames, channels, height, width)
        enc1 = self.enc1(flat)
        enc2 = self.enc2(enc1)
        feature_height, feature_width = enc2.shape[-2:]
        graph = enc2.reshape(batch, frames, DIM, feature_height, feature_width)
        for block in self.blocks:
            graph = block(graph, motion)
        graph = graph.reshape(batch * frames, DIM, feature_height, feature_width)
        graph = F.relu(self.up1(graph))
        graph = self.fuse1(torch.cat([graph, enc1], dim=1))
        graph = F.relu(self.up2(graph))
        output = self.out(torch.cat([graph, flat], dim=1))
        return torch.sigmoid(output).reshape(batch, frames, 1, height, width)


@torch.no_grad()
def evaluate_oracle(model, loader, patterns, device):
    model.eval()
    rows = []
    for target, labels, motion in loader:
        target = target.to(device)
        motion = motion.to(device)
        coarse = coarse_sequence(target, patterns)
        prediction = model(coarse, motion)
        batch, frames, channels, height, width = target.shape
        mse = (prediction - target).square().mean((1, 2, 3, 4))
        psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))
        frame_ssim = ssim(
            prediction.reshape(batch * frames, channels, height, width),
            target.reshape(batch * frames, channels, height, width),
            data_range=1.0, size_average=False,
        ).reshape(batch, frames).mean(1)
        difference_error = (
            (prediction[:, 1:] - prediction[:, :-1])
            - (target[:, 1:] - target[:, :-1])
        ).abs().mean((1, 2, 3, 4))
        for index in range(batch):
            rows.append({
                "label": int(labels[index]),
                "mse": float(mse[index]),
                "psnr_db": float(psnr[index]),
                "ssim": float(frame_ssim[index]),
                "temporal_difference_error": float(difference_error[index]),
            })
    return rows


def measure_oracle_efficiency(model, sample, motion, device, repeats=30):
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            model(sample, motion)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(repeats):
            model(sample, motion)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency = (time.perf_counter() - start) / repeats * 1000.0
        memory = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    return {"latency_ms": latency, "peak_memory_mb": memory}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed(SEED)
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    bank = {count: make_patterns(count, IMAGE_SIZE, IMAGE_SIZE, seed=SEED).to(device)
            for count in RATES}
    train_loader = DataLoader(
        MovingMNISTWithMotion("./data", True, TRAIN_N, SEED + 200),
        batch_size=BATCH, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        MovingMNISTWithMotion("./data", False, VAL_N, SEED + 201),
        batch_size=BATCH, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        MovingMNISTWithMotion("./data", False, TEST_N, SEED + 300),
        batch_size=4, shuffle=False, num_workers=0,
    )
    model = OracleMotionTGNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    history = []
    best = float("inf")
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        sample_count = 0
        for batch_index, (target, _, motion) in enumerate(train_loader):
            target = target.to(device)
            motion = motion.to(device)
            pattern_count = RATES[(batch_index + epoch - 1) % len(RATES)]
            with torch.no_grad():
                coarse = coarse_sequence(target, bank[pattern_count])
            prediction = model(coarse, motion)
            flat_target = target.reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
            loss = reconstruction_loss(prediction.reshape_as(flat_target), flat_target)
            loss = loss + 0.15 * temporal_error(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * target.shape[0]
            sample_count += target.shape[0]
        val_values = []
        for pattern_count in RATES:
            val_values.extend(
                row["mse"] for row in evaluate_oracle(
                    model, val_loader, bank[pattern_count], device
                )
            )
        val_mse = float(np.mean(val_values))
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / sample_count,
            "val_mse": val_mse,
        })
        print(
            f"epoch={epoch:02d}/{EPOCHS} train_loss={total_loss / sample_count:.6f} "
            f"val_mse={val_mse:.6f}"
        )
        if val_mse < best:
            best = val_mse
            torch.save(model.state_dict(), os.path.join(OUT, "best_oracle_motion_tgnn.pt"))

    model.load_state_dict(
        torch.load(os.path.join(OUT, "best_oracle_motion_tgnn.pt"), map_location=device)
    )
    detail = []
    summary = []
    for pattern_count in RATES:
        rows = evaluate_oracle(model, test_loader, bank[pattern_count], device)
        for sequence_id, row in enumerate(rows):
            detail.append({
                "pattern_count": pattern_count,
                "sampling_ratio": pattern_count / IMAGE_SIZE**2,
                "method": "OracleMotionTGNN",
                "sequence_id": sequence_id,
                **row,
            })
        mse_mean, mse_std = stats(rows, "mse")
        psnr_mean, psnr_std = stats(rows, "psnr_db")
        ssim_mean, ssim_std = stats(rows, "ssim")
        temporal_mean, temporal_std = stats(rows, "temporal_difference_error")
        result = {
            "pattern_count": pattern_count,
            "sampling_ratio": pattern_count / IMAGE_SIZE**2,
            "method": "OracleMotionTGNN",
            "mse_mean": mse_mean,
            "mse_std": mse_std,
            "psnr_mean_db": psnr_mean,
            "psnr_std_db": psnr_std,
            "ssim_mean": ssim_mean,
            "ssim_std": ssim_std,
            "temporal_difference_error_mean": temporal_mean,
            "temporal_difference_error_std": temporal_std,
        }
        summary.append(result)
        print(
            f"patterns={pattern_count:4d} OracleMotionTGNN "
            f"PSNR={psnr_mean:.2f}+/-{psnr_std:.2f} "
            f"SSIM={ssim_mean:.3f} temp_err={temporal_mean:.4f}"
        )

    first_target, _, first_motion = next(iter(test_loader))
    first_target = first_target[:1].to(device)
    first_motion = first_motion[:1].to(device)
    with torch.no_grad():
        first_coarse = coarse_sequence(first_target, bank[256])
    transformer = MiniSpatioTemporalTransformer().to(device)
    transformer.load_state_dict(
        torch.load(
            os.path.join(TRANSFORMER_DIR, "best_spatiotemporal_transformer.pt"),
            map_location=device,
        )
    )
    efficiency = {
        "OracleMotionTGNN": measure_oracle_efficiency(
            model, first_coarse, first_motion, device
        ),
        "SpatioTemporalTransformer": measure_efficiency(
            transformer, first_coarse, device
        ),
    }
    parameters = {
        "OracleMotionTGNN": sum(parameter.numel() for parameter in model.parameters()),
        "SpatioTemporalTransformer": sum(
            parameter.numel() for parameter in transformer.parameters()
        ),
    }
    elapsed = time.time() - start

    with open(os.path.join(OUT, "benchmark_detail.csv"), "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail[0].keys())
        writer.writeheader()
        writer.writerows(detail)
    with open(os.path.join(OUT, "benchmark_summary.csv"), "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": {
                    "seed": SEED,
                    "rates": list(RATES),
                    "train_sequences": TRAIN_N,
                    "val_sequences": VAL_N,
                    "test_sequences": TEST_N,
                    "epochs": EPOCHS,
                    "feature_grid": "16x16",
                    "graph": "four-neighbor spatial edges plus oracle bilinear motion edges",
                    "diagnostic_only": True,
                },
                "history": history,
                "summary": summary,
                "parameters": parameters,
                "efficiency": efficiency,
                "elapsed_seconds": elapsed,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    for name, values in efficiency.items():
        print(
            f"efficiency={name} params={parameters[name]} "
            f"latency_ms={values['latency_ms']:.3f} "
            f"peak_memory_mb={values['peak_memory_mb']:.1f}"
        )
    print(f"elapsed_seconds={elapsed:.1f}")
    print(f"saved={OUT}")


if __name__ == "__main__":
    main()

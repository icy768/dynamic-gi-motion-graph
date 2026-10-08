import csv
import json
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mini_repro_dynghost import IMAGE_SIZE, SEED, make_patterns
from train_frame_cnn import reconstruction_loss
from train_mixed_sampling import RATES, coarse_sequence, evaluate, stats, temporal_error
from train_spatiotemporal_transformer import MiniSpatioTemporalTransformer
from train_temporal_convgru import MovingMNISTSequences


TRAIN_N, VAL_N, TEST_N = 800, 120, 100
BATCH, EPOCHS, LR = 8, 10, 5e-4
PATCH_GRID = 8
HIDDEN = 32
OUT = "tgnn_outputs"
TRANSFORMER_DIR = "spatiotemporal_transformer_outputs"


def build_normalized_adjacency(seq_len=8, grid=8):
    n_per_frame = grid * grid
    n = seq_len * n_per_frame
    a = torch.zeros(n, n, dtype=torch.float32)
    for t in range(seq_len):
        for r in range(grid):
            for c in range(grid):
                i = t * n_per_frame + r * grid + c
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < grid and 0 <= cc < grid:
                        j = t * n_per_frame + rr * grid + cc
                        a[i, j] = 1.0
                if t > 0:
                    a[i, (t - 1) * n_per_frame + r * grid + c] = 1.0
                if t + 1 < seq_len:
                    a[i, (t + 1) * n_per_frame + r * grid + c] = 1.0
    a = a + torch.eye(n)
    deg = a.sum(dim=1).clamp_min(1.0)
    d_inv_sqrt = deg.rsqrt()
    return d_inv_sqrt[:, None] * a * d_inv_sqrt[None, :]


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.self_proj = nn.Linear(in_channels, out_channels)
        self.neighbor_proj = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, adjacency):
        neighbor = torch.matmul(adjacency, x)
        return F.gelu(self.norm(self.self_proj(x) + self.neighbor_proj(neighbor)))


class PatchTGNN(nn.Module):
    """A small fixed patch graph baseline for dynamic GI reconstruction."""

    def __init__(self, dim=HIDDEN, seq_len=8, grid=PATCH_GRID):
        super().__init__()
        self.seq_len = seq_len
        self.grid = grid
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, dim, 3, stride=2, padding=1), nn.ReLU(inplace=True)
        )
        self.graph1 = GraphConv(dim, dim)
        self.graph2 = GraphConv(dim, dim)
        self.temporal_gru = nn.GRU(dim, dim, batch_first=True)
        self.fuse16 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.up1 = nn.ConvTranspose2d(dim, 16, 4, stride=2, padding=1)
        self.fuse32 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.up2 = nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1)
        self.out = nn.Sequential(
            nn.Conv2d(9, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1)
        )
        self.register_buffer("adjacency", build_normalized_adjacency(seq_len, grid))

    def forward(self, x):
        b, t, c, h, w = x.shape
        flat = x.reshape(b * t, c, h, w)
        e1 = self.enc1(flat)
        e2 = self.enc2(e1)
        _, dim, hh, ww = e2.shape
        pooled = F.adaptive_avg_pool2d(e2, (self.grid, self.grid))
        nodes = pooled.reshape(b, t, dim, self.grid * self.grid).permute(0, 1, 3, 2)
        nodes = nodes.reshape(b, t * self.grid * self.grid, dim)
        nodes = self.graph2(self.graph1(nodes, self.adjacency), self.adjacency)
        nodes = nodes.reshape(b, t, self.grid * self.grid, dim)
        nodes = nodes.permute(0, 2, 1, 3).reshape(b * self.grid * self.grid, t, dim)
        nodes, _ = self.temporal_gru(nodes)
        nodes = nodes.reshape(b, self.grid * self.grid, t, dim).permute(0, 2, 3, 1)
        nodes = nodes.reshape(b * t, dim, self.grid, self.grid)
        graph_map = F.interpolate(nodes, size=(hh, ww), mode="bilinear", align_corners=False)
        z = self.fuse16(torch.cat([e2, graph_map], dim=1))
        z = F.relu(self.up1(z))
        z = self.fuse32(torch.cat([z, e1], dim=1))
        z = F.relu(self.up2(z))
        logits = self.out(torch.cat([z, flat], dim=1))
        return torch.sigmoid(logits).reshape(b, t, 1, h, w)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def measure_efficiency(model, sample, device, repeats=30):
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            _ = model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(repeats):
            _ = model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / repeats * 1000.0
        peak = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    return {"latency_ms": elapsed, "peak_memory_mb": peak}


def main():
    set_seed(SEED)
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    bank = {m: make_patterns(m, IMAGE_SIZE, IMAGE_SIZE, seed=SEED).to(device) for m in RATES}
    train = DataLoader(MovingMNISTSequences("./data", True, TRAIN_N, SEED + 200),
                       batch_size=BATCH, shuffle=True, num_workers=0)
    val = DataLoader(MovingMNISTSequences("./data", False, VAL_N, SEED + 201),
                     batch_size=BATCH, shuffle=False, num_workers=0)
    test = DataLoader(MovingMNISTSequences("./data", False, TEST_N, SEED + 300),
                      batch_size=4, shuffle=False, num_workers=0)
    model = PatchTGNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    history = []
    best = float("inf")
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = n = 0
        for batch_index, (target, _) in enumerate(train):
            target = target.to(device)
            pattern_count = RATES[(batch_index + epoch - 1) % len(RATES)]
            with torch.no_grad():
                coarse = coarse_sequence(target, bank[pattern_count])
            pred = model(coarse)
            flat_target = target.reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
            loss = reconstruction_loss(pred.reshape_as(flat_target), flat_target)
            loss = loss + 0.15 * temporal_error(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * target.shape[0]
            n += target.shape[0]
        vals = []
        for pattern_count in RATES:
            vals.extend(r["mse"] for r in evaluate(model, "PatchTGNN", val, bank[pattern_count], device))
        val_mse = float(np.mean(vals))
        history.append({"epoch": epoch, "train_loss": total / n, "val_mse": val_mse})
        print(f"epoch={epoch:02d}/{EPOCHS} train_loss={total / n:.6f} val_mse={val_mse:.6f}")
        if val_mse < best:
            best = val_mse
            torch.save(model.state_dict(), os.path.join(OUT, "best_patch_tgnn.pt"))

    model.load_state_dict(torch.load(os.path.join(OUT, "best_patch_tgnn.pt"), map_location=device))
    detail = []
    summary = []
    for pattern_count in RATES:
        rows = evaluate(model, "PatchTGNN", test, bank[pattern_count], device)
        for index, row in enumerate(rows):
            detail.append(dict(pattern_count=pattern_count,
                               sampling_ratio=pattern_count / IMAGE_SIZE**2,
                               method="PatchTGNN", sequence_id=index, **row))
        pm, ps = stats(rows, "psnr_db")
        sm, ss = stats(rows, "ssim")
        em, es = stats(rows, "temporal_difference_error")
        mm, ms = stats(rows, "mse")
        summary.append(dict(pattern_count=pattern_count,
                            sampling_ratio=pattern_count / IMAGE_SIZE**2,
                            method="PatchTGNN", mse_mean=mm, mse_std=ms,
                            psnr_mean_db=pm, psnr_std_db=ps, ssim_mean=sm, ssim_std=ss,
                            temporal_difference_error_mean=em,
                            temporal_difference_error_std=es))
        print(f"patterns={pattern_count:4d} PatchTGNN PSNR={pm:.2f}+/-{ps:.2f} "
              f"SSIM={sm:.3f} temp_err={em:.4f}")

    transformer = MiniSpatioTemporalTransformer().to(device)
    transformer.load_state_dict(torch.load(
        os.path.join(TRANSFORMER_DIR, "best_spatiotemporal_transformer.pt"), map_location=device))
    sample_target, _ = next(iter(test))
    sample_target = sample_target[:1].to(device)
    with torch.no_grad():
        sample = coarse_sequence(sample_target, bank[256])
    efficiency = {
        "PatchTGNN": measure_efficiency(model, sample, device),
        "SpatioTemporalTransformer": measure_efficiency(transformer, sample, device),
    }
    params = {
        "PatchTGNN": sum(p.numel() for p in model.parameters()),
        "SpatioTemporalTransformer": sum(p.numel() for p in transformer.parameters()),
    }
    for name in efficiency:
        efficiency[name]["parameters"] = params[name]
        print(f"efficiency={name} params={params[name]} "
              f"latency_ms={efficiency[name]['latency_ms']:.3f} "
              f"peak_memory_mb={efficiency[name]['peak_memory_mb']:.1f}")

    def save_csv(name, rows):
        with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    save_csv("benchmark_summary.csv", summary)
    save_csv("benchmark_detail.csv", detail)
    elapsed = time.time() - start
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as handle:
        json.dump({"config": {"seed": SEED, "rates": list(RATES), "train_sequences": TRAIN_N,
                               "val_sequences": VAL_N, "test_sequences": TEST_N, "epochs": EPOCHS,
                               "patch_grid": PATCH_GRID, "hidden_channels": HIDDEN,
                               "graph": "4-neighbor spatial plus same-position temporal edges"},
                   "history": history, "summary": summary, "parameters": params,
                   "efficiency": efficiency, "elapsed_seconds": elapsed},
                  handle, ensure_ascii=False, indent=2)

    x = np.array(RATES) / IMAGE_SIZE**2 * 100
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for method, marker, values in (("PatchTGNN", "D", summary),):
        axes[0].errorbar(x, [r["psnr_mean_db"] for r in values],
                         yerr=[r["psnr_std_db"] for r in values], marker=marker, capsize=3, label=method)
        axes[1].errorbar(x, [r["ssim_mean"] for r in values],
                         yerr=[r["ssim_std"] for r in values], marker=marker, capsize=3, label=method)
        axes[2].errorbar(x, [r["temporal_difference_error_mean"] for r in values],
                         yerr=[r["temporal_difference_error_std"] for r in values], marker=marker, capsize=3, label=method)
    for ax, title in zip(axes, ("PSNR", "SSIM", "Temporal difference error")):
        ax.set_xlabel("Sampling ratio (%)")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Fixed patch TGNN on 100 unseen Moving-MNIST sequences")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "patch_tgnn_metrics.png"), dpi=180)
    plt.close(fig)
    print(f"elapsed_seconds={elapsed:.1f}")
    print(f"saved={OUT}")


if __name__ == "__main__":
    main()

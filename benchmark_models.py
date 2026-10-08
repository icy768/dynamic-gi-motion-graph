import csv
import json
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mini_repro_dynghost import IMAGE_SIZE, SEED, make_patterns
from train_frame_cnn import FrameCNN
from train_temporal_convgru import BidirectionalConvGRU, MovingMNISTSequences


PATTERN_COUNTS = (128, 256, 512, 1024)
TEST_SEQUENCES = 100
BATCH_SIZE = 4
OUT_DIR = "benchmark_outputs"
FRAME_MODEL_PATH = os.path.join("frame_cnn_outputs", "best_frame_cnn.pt")
TEMPORAL_MODEL_PATH = os.path.join("temporal_convgru_outputs", "best_temporal_convgru.pt")


def normalize_per_image(x, eps=1e-8):
    lo = x.amin(dim=(-2, -1), keepdim=True)
    hi = x.amax(dim=(-2, -1), keepdim=True)
    return (x - lo) / (hi - lo + eps)


def dgi_dynamic(frames, patterns):
    # frames [N,1,H,W], patterns [M,H,W]
    n = frames.shape[0]
    m = patterns.shape[0]
    a = patterns.reshape(m, -1)
    buckets = frames.reshape(n, -1) @ a.T
    illumination = patterns.sum(dim=(1, 2)).clamp_min(1e-8)
    normalized_bucket = buckets / illumination[None]
    centered_bucket = normalized_bucket - normalized_bucket.mean(dim=1, keepdim=True)
    centered_patterns = patterns - patterns.mean(dim=0, keepdim=True)
    recon = torch.einsum("nm,mhw->nhw", centered_bucket, centered_patterns) / m
    return normalize_per_image(recon[:, None])


def per_sequence_metrics(pred, target):
    # Average spatial/frame error within each sequence, then compute PSNR.
    mse = (pred - target).square().mean(dim=(1, 2, 3, 4))
    psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))
    return mse.cpu().numpy(), psnr.cpu().numpy()


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    dataset = MovingMNISTSequences("./data", False, TEST_SEQUENCES, SEED + 100)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    frame_model = FrameCNN().to(device)
    frame_model.load_state_dict(torch.load(FRAME_MODEL_PATH, map_location=device))
    frame_model.eval()
    temporal_model = BidirectionalConvGRU().to(device)
    temporal_model.load_state_dict(torch.load(TEMPORAL_MODEL_PATH, map_location=device))
    temporal_model.eval()

    rows = []
    summary_rows = []
    saved_example = None
    with torch.no_grad():
        for pattern_count in PATTERN_COUNTS:
            patterns = make_patterns(pattern_count, IMAGE_SIZE, IMAGE_SIZE, seed=SEED).to(device)
            collected = {name: {"mse": [], "psnr": []} for name in ("DGI", "FrameCNN", "BiConvGRU")}
            offset = 0
            for target, labels in loader:
                target = target.to(device)
                b, t, c, h, w = target.shape
                coarse = dgi_dynamic(target.reshape(b * t, c, h, w), patterns).reshape_as(target)
                frame_pred = frame_model(coarse.reshape(b * t, c, h, w)).reshape_as(target)
                temporal_pred = temporal_model(coarse)
                outputs = {"DGI": coarse, "FrameCNN": frame_pred, "BiConvGRU": temporal_pred}
                for method, pred in outputs.items():
                    mse_values, psnr_values = per_sequence_metrics(pred, target)
                    collected[method]["mse"].extend(mse_values.tolist())
                    collected[method]["psnr"].extend(psnr_values.tolist())
                    for j in range(b):
                        rows.append({
                            "pattern_count": pattern_count,
                            "sampling_ratio": pattern_count / (IMAGE_SIZE * IMAGE_SIZE),
                            "trained_at_this_ratio": pattern_count == 256,
                            "sequence_id": offset + j,
                            "label": int(labels[j]),
                            "method": method,
                            "mse": float(mse_values[j]),
                            "psnr_db": float(psnr_values[j]),
                        })
                if pattern_count == 256 and offset == 0:
                    saved_example = {
                        "target": target[0].cpu(), "dgi": coarse[0].cpu(),
                        "frame": frame_pred[0].cpu(), "temporal": temporal_pred[0].cpu(),
                        "label": int(labels[0]),
                    }
                offset += b

            frame_psnr = np.asarray(collected["FrameCNN"]["psnr"])
            temporal_psnr = np.asarray(collected["BiConvGRU"]["psnr"])
            paired_gain = temporal_psnr - frame_psnr
            win_rate = float((paired_gain > 0).mean())
            for method in ("DGI", "FrameCNN", "BiConvGRU"):
                mse_stats = summarize(collected[method]["mse"])
                psnr_stats = summarize(collected[method]["psnr"])
                summary_rows.append({
                    "pattern_count": pattern_count,
                    "sampling_ratio": pattern_count / (IMAGE_SIZE * IMAGE_SIZE),
                    "trained_at_this_ratio": pattern_count == 256,
                    "method": method,
                    "mse_mean": mse_stats["mean"], "mse_std": mse_stats["std"],
                    "psnr_mean_db": psnr_stats["mean"], "psnr_std_db": psnr_stats["std"],
                    "psnr_median_db": psnr_stats["median"],
                    "temporal_minus_frame_mean_db": float(paired_gain.mean()) if method == "BiConvGRU" else "",
                    "temporal_win_rate": win_rate if method == "BiConvGRU" else "",
                })
                print(
                    f"patterns={pattern_count:4d} method={method:10s} "
                    f"PSNR={psnr_stats['mean']:.2f}+/-{psnr_stats['std']:.2f} dB"
                )
            print(
                f"patterns={pattern_count:4d} temporal-frame gain={paired_gain.mean():.2f} dB "
                f"win_rate={win_rate:.1%}"
            )

    detail_path = os.path.join(OUT_DIR, "per_sequence_results.csv")
    with open(detail_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary_path = os.path.join(OUT_DIR, "benchmark_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    with open(os.path.join(OUT_DIR, "benchmark_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "seed": SEED, "test_sequences": TEST_SEQUENCES,
                "pattern_counts": list(PATTERN_COUNTS),
                "training_pattern_count": 256,
                "note": "Only 256 patterns is in-distribution; other rates test zero-shot sampling-rate generalization.",
            },
            "summary": summary_rows,
        }, f, ensure_ascii=False, indent=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.asarray(PATTERN_COUNTS) / (IMAGE_SIZE * IMAGE_SIZE) * 100
    for method, marker in (("DGI", "o"), ("FrameCNN", "s"), ("BiConvGRU", "^")):
        selected = [r for r in summary_rows if r["method"] == method]
        means = [r["psnr_mean_db"] for r in selected]
        stds = [r["psnr_std_db"] for r in selected]
        ax.errorbar(x, means, yerr=stds, marker=marker, capsize=3, label=method)
    ax.axvline(6.25, color="gray", linestyle="--", linewidth=1, label="training ratio")
    ax.set_xlabel("Sampling ratio (%)")
    ax.set_ylabel("Sequence PSNR (dB), mean +/- std")
    ax.set_title("100 unseen Moving-MNIST sequences")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "sampling_generalization.png"), dpi=180)
    plt.close(fig)

    if saved_example is not None:
        sequences = [saved_example[k][:, 0] for k in ("target", "dgi", "frame", "temporal")]
        names = ("GT", "DGI", "Frame CNN", "Bi-ConvGRU")
        fig, axes = plt.subplots(4, 8, figsize=(16, 8))
        for row, (sequence, name) in enumerate(zip(sequences, names)):
            for t in range(8):
                axes[row, t].imshow(sequence[t], cmap="gray", vmin=0, vmax=1)
                axes[row, t].set_title(f"{name} t={t}", fontsize=8)
                axes[row, t].axis("off")
        fig.suptitle(f"Unseen benchmark example, label={saved_example['label']}, sampling=6.25%")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "unseen_example.png"), dpi=180)
        plt.close(fig)
    print(f"saved={OUT_DIR}")


if __name__ == "__main__":
    main()

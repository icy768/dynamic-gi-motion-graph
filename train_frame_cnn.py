
from project_paths import PROJECT_ROOT, DATA_DIR as PROJECT_DATA_DIR
import csv
import json
import math
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from mini_repro_dynghost import IMAGE_SIZE, SEED, make_patterns, make_sequence


NUM_PATTERNS = 256
TRAIN_SAMPLES = 2000
VAL_SAMPLES = 300
BATCH_SIZE = 32
EPOCHS = 12
LEARNING_RATE = 1e-3
OUT_DIR = str(PROJECT_ROOT / 'frame_cnn_outputs')


def normalize_per_image(x, eps=1e-8):
    lo = x.amin(dim=(-2, -1), keepdim=True)
    hi = x.amax(dim=(-2, -1), keepdim=True)
    return (x - lo) / (hi - lo + eps)


def dgi_batch(frames, patterns):
    # frames: [B,1,H,W], patterns: [M,H,W]
    b = frames.shape[0]
    a = patterns.reshape(NUM_PATTERNS, -1)
    y = frames.reshape(b, -1) @ a.T
    s = patterns.sum(dim=(1, 2)).clamp_min(1e-8)
    b_norm = y / s[None]
    centered_b = b_norm - b_norm.mean(dim=1, keepdim=True)
    centered_patterns = patterns - patterns.mean(dim=0, keepdim=True)
    recon = torch.einsum("bm,mhw->bhw", centered_b, centered_patterns) / NUM_PATTERNS
    return normalize_per_image(recon[:, None])


class MovingMNISTFrames(Dataset):
    def __init__(self, root, train, length, seed):
        self.mnist = datasets.MNIST(
            root=root, train=train, download=True, transform=transforms.ToTensor()
        )
        self.length = length
        rng = np.random.default_rng(seed)
        self.indices = rng.integers(0, len(self.mnist), size=length)
        self.dx = rng.uniform(-12.0, 12.0, size=length)
        self.dy = rng.uniform(-8.0, 8.0, size=length)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        image, label = self.mnist[int(self.indices[i])]
        image = F.interpolate(
            image[None], size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
        )[0]
        theta = torch.tensor(
            [[[1.0, 0.0, -2.0 * self.dx[i] / IMAGE_SIZE],
              [0.0, 1.0, -2.0 * self.dy[i] / IMAGE_SIZE]]],
            dtype=torch.float32,
        )
        grid = F.affine_grid(theta, (1, 1, IMAGE_SIZE, IMAGE_SIZE), align_corners=False)
        frame = F.grid_sample(
            image[None], grid, align_corners=False, padding_mode="zeros"
        )[0]
        return frame, label


class FrameCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))


def reconstruction_loss(pred, target):
    # MNIST is mostly black background. Foreground weighting prevents the
    # deceptively good but useless all-black solution.
    weight = 1.0 + 6.0 * target
    weighted_mse = (weight * (pred - target).square()).mean()
    edge_target_x = target[:, :, :, 1:] - target[:, :, :, :-1]
    edge_pred_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    edge_target_y = target[:, :, 1:, :] - target[:, :, :-1, :]
    edge_pred_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    edge_loss = F.l1_loss(edge_pred_x, edge_target_x) + F.l1_loss(edge_pred_y, edge_target_y)
    return weighted_mse + 0.2 * edge_loss


def metrics(pred, target):
    mse_value = F.mse_loss(pred, target).item()
    return mse_value, 10.0 * math.log10(1.0 / max(mse_value, 1e-12))


@torch.no_grad()
def evaluate_loader(model, loader, patterns, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    for target, _ in loader:
        target = target.to(device, non_blocking=True)
        coarse = dgi_batch(target, patterns)
        pred = model(coarse)
        total_loss += F.mse_loss(pred, target, reduction="sum").item()
        total_count += target.numel()
    mse_value = total_loss / total_count
    return mse_value, 10.0 * math.log10(1.0 / max(mse_value, 1e-12))


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    patterns = make_patterns(NUM_PATTERNS, IMAGE_SIZE, IMAGE_SIZE, seed=SEED).to(device)
    train_set = MovingMNISTFrames(str(PROJECT_DATA_DIR), True, TRAIN_SAMPLES, SEED + 1)
    val_set = MovingMNISTFrames(str(PROJECT_DATA_DIR), False, VAL_SAMPLES, SEED + 2)
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
    )

    model = FrameCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = []
    best_val = float("inf")
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_sum = 0.0
        train_count = 0
        for target, _ in train_loader:
            target = target.to(device, non_blocking=True)
            with torch.no_grad():
                coarse = dgi_batch(target, patterns)
            pred = model(coarse)
            loss = reconstruction_loss(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_sum += loss.item() * target.shape[0]
            train_count += target.shape[0]
        train_mse = train_sum / train_count
        val_mse, val_psnr = evaluate_loader(model, val_loader, patterns, device)
        row = {
            "epoch": epoch,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "val_psnr_db": val_psnr,
        }
        history.append(row)
        print(
            f"epoch={epoch:02d}/{EPOCHS} train_mse={train_mse:.6f} "
            f"val_mse={val_mse:.6f} val_psnr={val_psnr:.2f} dB"
        )
        if val_mse < best_val:
            best_val = val_mse
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_frame_cnn.pt"))

    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_frame_cnn.pt"), map_location=device))
    model.eval()
    frames, label, trajectory = make_sequence()
    target = frames[:, None].to(device)
    with torch.no_grad():
        coarse = dgi_batch(target, patterns)
        refined = model(coarse)
    coarse_mse, coarse_psnr = metrics(coarse, target)
    cnn_mse, cnn_psnr = metrics(refined, target)
    print(f"fixed_test label={label} DGI_MSE={coarse_mse:.6f} DGI_PSNR={coarse_psnr:.2f} dB")
    print(f"fixed_test label={label} CNN_MSE={cnn_mse:.6f} CNN_PSNR={cnn_psnr:.2f} dB")

    summary = {
        "config": {
            "seed": SEED,
            "image_size": IMAGE_SIZE,
            "num_patterns": NUM_PATTERNS,
            "sampling_ratio": NUM_PATTERNS / (IMAGE_SIZE * IMAGE_SIZE),
            "train_samples": TRAIN_SAMPLES,
            "val_samples": VAL_SAMPLES,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
        },
        "fixed_test": {
            "mnist_split": "test",
            "mnist_index": 7,
            "label": int(label),
            "dgi_mse": coarse_mse,
            "dgi_psnr_db": coarse_psnr,
            "cnn_mse": cnn_mse,
            "cnn_psnr_db": cnn_psnr,
        },
        "history": history,
        "elapsed_seconds": time.time() - start,
    }
    with open(os.path.join(OUT_DIR, "frame_cnn_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT_DIR, "training_history.csv"), "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    coarse = coarse.cpu()[:, 0]
    refined = refined.cpu()[:, 0]
    fig, axes = plt.subplots(3, len(frames), figsize=(16, 6))
    for t in range(len(frames)):
        axes[0, t].imshow(frames[t], cmap="gray", vmin=0, vmax=1)
        axes[0, t].set_title(f"GT t={t}", fontsize=8)
        axes[1, t].imshow(coarse[t], cmap="gray", vmin=0, vmax=1)
        axes[1, t].set_title("DGI", fontsize=8)
        axes[2, t].imshow(refined[t], cmap="gray", vmin=0, vmax=1)
        axes[2, t].set_title("Frame CNN", fontsize=8)
        for row in range(3):
            axes[row, t].axis("off")
    fig.suptitle(
        f"Fixed MNIST test[7], label={label}: DGI {coarse_psnr:.2f} dB -> CNN {cnn_psnr:.2f} dB"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "frame_cnn_comparison.png"), dpi=180)
    plt.close(fig)

    torch.save(
        {
            "frames": frames,
            "dgi_recon": coarse,
            "cnn_recon": refined,
            "trajectory": trajectory,
            "summary": summary,
        },
        os.path.join(OUT_DIR, "frame_cnn_experiment.pt"),
    )
    print(f"elapsed_seconds={summary['elapsed_seconds']:.1f}")
    print(f"saved={OUT_DIR}")


if __name__ == "__main__":
    main()

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
from train_frame_cnn import FrameCNN, dgi_batch, reconstruction_loss


NUM_PATTERNS = 256
SEQ_LEN = 8
TRAIN_SEQUENCES = 800
VAL_SEQUENCES = 120
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-3
HIDDEN_CHANNELS = 16
OUT_DIR = "temporal_convgru_outputs"
FRAME_CNN_PATH = os.path.join("frame_cnn_outputs", "best_frame_cnn.pt")


class MovingMNISTSequences(Dataset):
    def __init__(self, root, train, length, seed):
        self.mnist = datasets.MNIST(
            root=root, train=train, download=True, transform=transforms.ToTensor()
        )
        self.length = length
        rng = np.random.default_rng(seed)
        self.indices = rng.integers(0, len(self.mnist), size=length)
        self.start_x = rng.uniform(-7.0, 7.0, size=length)
        self.start_y = rng.uniform(-5.0, 5.0, size=length)
        self.speed = rng.uniform(1.0, 3.8, size=length)
        self.angle = rng.uniform(0.0, 2.0 * math.pi, size=length)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        image, label = self.mnist[int(self.indices[i])]
        image = F.interpolate(
            image[None], size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
        )[0]
        frames = []
        for t in range(SEQ_LEN):
            dx = self.start_x[i] + self.speed[i] * math.cos(self.angle[i]) * t
            dy = self.start_y[i] + self.speed[i] * math.sin(self.angle[i]) * t
            theta = torch.tensor(
                [[[1.0, 0.0, -2.0 * dx / IMAGE_SIZE],
                  [0.0, 1.0, -2.0 * dy / IMAGE_SIZE]]],
                dtype=torch.float32,
            )
            grid = F.affine_grid(theta, (1, 1, IMAGE_SIZE, IMAGE_SIZE), align_corners=False)
            frame = F.grid_sample(
                image[None], grid, align_corners=False, padding_mode="zeros"
            )[0]
            frames.append(frame)
        return torch.stack(frames), label


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super().__init__()
        both = input_channels + hidden_channels
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(both, 2 * hidden_channels, 3, padding=1)
        self.candidate = nn.Conv2d(both, hidden_channels, 3, padding=1)

    def forward(self, x, h):
        if h is None:
            h = torch.zeros(
                x.shape[0], self.hidden_channels, x.shape[-2], x.shape[-1],
                device=x.device, dtype=x.dtype
            )
        reset, update = torch.sigmoid(self.gates(torch.cat([x, h], dim=1))).chunk(2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat([x, reset * h], dim=1)))
        return (1.0 - update) * h + update * candidate


class BidirectionalConvGRU(nn.Module):
    def __init__(self, hidden_channels=HIDDEN_CHANNELS):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.forward_cell = ConvGRUCell(16, hidden_channels)
        self.backward_cell = ConvGRUCell(16, hidden_channels)
        self.decoder = nn.Sequential(
            nn.Conv2d(2 * hidden_channels, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, sequence):
        # sequence: [B,T,1,H,W]
        b, t, _, h, w = sequence.shape
        features = self.encoder(sequence.reshape(b * t, 1, h, w)).reshape(b, t, 16, h, w)
        forward_states = []
        state = None
        for i in range(t):
            state = self.forward_cell(features[:, i], state)
            forward_states.append(state)
        backward_states = [None] * t
        state = None
        for i in range(t - 1, -1, -1):
            state = self.backward_cell(features[:, i], state)
            backward_states[i] = state
        fused = torch.cat(
            [torch.stack(forward_states, dim=1), torch.stack(backward_states, dim=1)], dim=2
        )
        output = self.decoder(fused.reshape(b * t, -1, h, w))
        return torch.sigmoid(output).reshape(b, t, 1, h, w)


def sequence_dgi(target, patterns):
    b, t, c, h, w = target.shape
    return dgi_batch(target.reshape(b * t, c, h, w), patterns).reshape(b, t, c, h, w)


def plain_metrics(pred, target):
    value = F.mse_loss(pred, target).item()
    return value, 10.0 * math.log10(1.0 / max(value, 1e-12))


@torch.no_grad()
def validate(model, loader, patterns, device):
    model.eval()
    squared_error = 0.0
    count = 0
    for target, _ in loader:
        target = target.to(device, non_blocking=True)
        coarse = sequence_dgi(target, patterns)
        pred = model(coarse)
        squared_error += (pred - target).square().sum().item()
        count += target.numel()
    value = squared_error / count
    return value, 10.0 * math.log10(1.0 / max(value, 1e-12))


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
    train_set = MovingMNISTSequences("./data", True, TRAIN_SEQUENCES, SEED + 10)
    val_set = MovingMNISTSequences("./data", False, VAL_SEQUENCES, SEED + 11)
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
    )

    model = BidirectionalConvGRU().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = []
    best_val = float("inf")
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for target, _ in train_loader:
            target = target.to(device, non_blocking=True)
            with torch.no_grad():
                coarse = sequence_dgi(target, patterns)
            pred = model(coarse)
            flat_pred = pred.reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
            flat_target = target.reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
            spatial_loss = reconstruction_loss(flat_pred, flat_target)
            temporal_target = target[:, 1:] - target[:, :-1]
            temporal_pred = pred[:, 1:] - pred[:, :-1]
            temporal_loss = F.l1_loss(temporal_pred, temporal_target)
            loss = spatial_loss + 0.15 * temporal_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * target.shape[0]
            sample_count += target.shape[0]
        val_mse, val_psnr = validate(model, val_loader, patterns, device)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / sample_count,
            "val_mse": val_mse,
            "val_psnr_db": val_psnr,
        }
        history.append(row)
        print(
            f"epoch={epoch:02d}/{EPOCHS} train_loss={row['train_loss']:.6f} "
            f"val_mse={val_mse:.6f} val_psnr={val_psnr:.2f} dB"
        )
        if val_mse < best_val:
            best_val = val_mse
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_temporal_convgru.pt"))

    model.load_state_dict(
        torch.load(os.path.join(OUT_DIR, "best_temporal_convgru.pt"), map_location=device)
    )
    model.eval()
    frame_model = FrameCNN().to(device)
    frame_model.load_state_dict(torch.load(FRAME_CNN_PATH, map_location=device))
    frame_model.eval()

    frames, label, trajectory = make_sequence()
    target = frames[None, :, None].to(device)
    with torch.no_grad():
        coarse = sequence_dgi(target, patterns)
        frame_pred = frame_model(coarse.reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE)).reshape_as(target)
        temporal_pred = model(coarse)
    dgi_mse, dgi_psnr = plain_metrics(coarse, target)
    frame_mse, frame_psnr = plain_metrics(frame_pred, target)
    temporal_mse, temporal_psnr = plain_metrics(temporal_pred, target)
    print(f"fixed_test DGI_MSE={dgi_mse:.6f} DGI_PSNR={dgi_psnr:.2f} dB")
    print(f"fixed_test FrameCNN_MSE={frame_mse:.6f} FrameCNN_PSNR={frame_psnr:.2f} dB")
    print(f"fixed_test Temporal_MSE={temporal_mse:.6f} Temporal_PSNR={temporal_psnr:.2f} dB")

    summary = {
        "config": {
            "seed": SEED,
            "image_size": IMAGE_SIZE,
            "seq_len": SEQ_LEN,
            "num_patterns": NUM_PATTERNS,
            "sampling_ratio": NUM_PATTERNS / (IMAGE_SIZE * IMAGE_SIZE),
            "train_sequences": TRAIN_SEQUENCES,
            "val_sequences": VAL_SEQUENCES,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "hidden_channels_per_direction": HIDDEN_CHANNELS,
        },
        "fixed_test": {
            "mnist_split": "test", "mnist_index": 7, "label": int(label),
            "dgi_mse": dgi_mse, "dgi_psnr_db": dgi_psnr,
            "frame_cnn_mse": frame_mse, "frame_cnn_psnr_db": frame_psnr,
            "temporal_mse": temporal_mse, "temporal_psnr_db": temporal_psnr,
        },
        "history": history,
        "elapsed_seconds": time.time() - start,
    }
    with open(os.path.join(OUT_DIR, "temporal_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT_DIR, "training_history.csv"), "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    images = [target.cpu()[0, :, 0], coarse.cpu()[0, :, 0],
              frame_pred.cpu()[0, :, 0], temporal_pred.cpu()[0, :, 0]]
    names = ["GT", "DGI", "Frame CNN", "Bi-ConvGRU"]
    fig, axes = plt.subplots(4, SEQ_LEN, figsize=(16, 8))
    for row, (sequence, name) in enumerate(zip(images, names)):
        for t in range(SEQ_LEN):
            axes[row, t].imshow(sequence[t], cmap="gray", vmin=0, vmax=1)
            axes[row, t].set_title(f"{name} t={t}", fontsize=8)
            axes[row, t].axis("off")
    fig.suptitle(
        f"MNIST test[7], 6.25%: DGI {dgi_psnr:.2f} | Frame CNN {frame_psnr:.2f} | "
        f"Bi-ConvGRU {temporal_psnr:.2f} dB"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "temporal_comparison.png"), dpi=180)
    plt.close(fig)
    torch.save(
        {"frames": frames, "dgi": coarse.cpu(), "frame_cnn": frame_pred.cpu(),
         "temporal": temporal_pred.cpu(), "trajectory": trajectory, "summary": summary},
        os.path.join(OUT_DIR, "temporal_experiment.pt"),
    )
    print(f"elapsed_seconds={summary['elapsed_seconds']:.1f}")
    print(f"saved={OUT_DIR}")


if __name__ == "__main__":
    main()

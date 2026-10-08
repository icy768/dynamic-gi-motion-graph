import os
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

SEED = 20260902
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

IMAGE_SIZE = 64
SEQ_LEN = 8
NUM_PATTERNS = 256
OUT_DIR = 'mini_outputs'
os.makedirs(OUT_DIR, exist_ok=True)

def normalize01(x, eps=1e-8):
    return (x - x.min()) / (x.max() - x.min() + eps)

def make_patterns(m, h, w, seed=SEED):
    gen = torch.Generator().manual_seed(seed)
    return (torch.rand((m, h, w), generator=gen) > 0.5).float()

def translate(img, dx, dy):
    theta = torch.tensor([[[1.0, 0.0, -2.0 * dx / IMAGE_SIZE],
                           [0.0, 1.0, -2.0 * dy / IMAGE_SIZE]]])
    grid = F.affine_grid(theta, (1, 1, IMAGE_SIZE, IMAGE_SIZE), align_corners=False)
    return F.grid_sample(img[None, None], grid, align_corners=False, padding_mode='zeros')[0, 0]

def make_sequence():
    mnist = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    img, label = mnist[7]
    img = F.interpolate(img[None], size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)[0, 0]
    angle = math.radians(25)
    speed = 3.2
    frames = []
    traj = []
    for t in range(SEQ_LEN):
        dx = speed * math.cos(angle) * t
        dy = speed * math.sin(angle) * t
        frames.append(translate(img, dx, dy))
        traj.append((dx, dy))
    return torch.stack(frames), label, traj

def buckets_from_patterns(frames, patterns):
    a = patterns.view(NUM_PATTERNS, -1)
    y = []
    for frame in frames:
        y.append(a @ frame.reshape(-1))
    return torch.stack(y)

def dgi_reconstruct(patterns, buckets):
    # Differential/correlation-style GI reconstruction per frame.
    s = patterns.sum(dim=(1, 2)).clamp_min(1e-8)
    i_mean = patterns.mean(dim=0)
    recons = []
    for y in buckets:
        b_norm = y / s
        recon = ((b_norm - b_norm.mean())[:, None, None] * (patterns - i_mean)).mean(dim=0)
        recons.append(normalize01(recon))
    return torch.stack(recons)

def mse(x, y):
    return torch.mean((x - y) ** 2).item()

def psnr(x, y):
    val = mse(x, y)
    return 10 * math.log10(1.0 / max(val, 1e-12))

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={device}')
    if device == 'cuda':
        print(f'gpu={torch.cuda.get_device_name(0)}')
        print(f'vram_mb={torch.cuda.get_device_properties(0).total_memory // 1024**2}')

    frames, label, traj = make_sequence()
    patterns = make_patterns(NUM_PATTERNS, IMAGE_SIZE, IMAGE_SIZE)
    buckets = buckets_from_patterns(frames, patterns)
    recon = dgi_reconstruct(patterns, buckets)

    print(f'label={label}')
    print(f'frames_shape={tuple(frames.shape)} patterns_shape={tuple(patterns.shape)} buckets_shape={tuple(buckets.shape)}')
    print(f'sampling_ratio={NUM_PATTERNS/(IMAGE_SIZE*IMAGE_SIZE):.4f}')
    print(f'DGI_MSE={mse(recon, frames):.6f}')
    print(f'DGI_PSNR={psnr(recon, frames):.2f} dB')

    fig, axes = plt.subplots(3, SEQ_LEN, figsize=(2 * SEQ_LEN, 6))
    for t in range(SEQ_LEN):
        axes[0, t].imshow(frames[t], cmap='gray', vmin=0, vmax=1)
        axes[0, t].set_title(f'GT t={t}', fontsize=8)
        axes[1, t].imshow(recon[t], cmap='gray', vmin=0, vmax=1)
        axes[1, t].set_title('DGI', fontsize=8)
        axes[2, t].imshow((recon[t] - frames[t]).abs(), cmap='magma')
        axes[2, t].set_title('|err|', fontsize=8)
        for r in range(3):
            axes[r, t].axis('off')
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'mini_dgi_moving_mnist.png')
    plt.savefig(fig_path, dpi=160)
    print(f'saved={fig_path}')

    torch.save({
        'frames': frames,
        'patterns': patterns,
        'buckets': buckets,
        'dgi_recon': recon,
        'trajectory': traj,
        'config': {
            'image_size': IMAGE_SIZE,
            'seq_len': SEQ_LEN,
            'num_patterns': NUM_PATTERNS,
            'seed': SEED,
        }
    }, os.path.join(OUT_DIR, 'mini_dgi_moving_mnist.pt'))

if __name__ == '__main__':
    main()

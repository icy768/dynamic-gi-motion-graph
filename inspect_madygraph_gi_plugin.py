"""Render held-out coarse/refined samples for the MadyGraph GI pilot."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from train_graph_protocol_v2 import IMAGE_SIZE, RATES, SEED, make_splits, seed_all
from train_madygraph_gi_plugin import (
    BASE_DIR,
    MOTION_PATH,
    PROJECT,
    ResidualRefiner,
    precompute,
)
from train_spatial_motion_diagnostic import SpatialMotionNet


def main():
    sys.path[:0] = [str(PROJECT), str(Path(__file__).resolve().parent)]
    from mini_repro_dynghost import make_patterns
    from train_spatiotemporal_transformer import MiniSpatioTemporalTransformer
    from pytorch_msssim import ssim

    seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_set = make_splits(PROJECT)
    loader = DataLoader(test_set, batch_size=4, shuffle=False)
    bank = {rate: make_patterns(rate, IMAGE_SIZE, IMAGE_SIZE, seed=SEED).to(device) for rate in RATES}
    transformer = MiniSpatioTemporalTransformer().to(device)
    transformer.load_state_dict(torch.load(BASE_DIR / "best_transformer.pt", map_location=device))
    motion_net = SpatialMotionNet().to(device)
    motion_net.load_state_dict(torch.load(MOTION_PATH, map_location=device))
    test = precompute(loader, transformer, motion_net, bank, device)
    rate = 128
    sample = {k: v[:1].to(device) for k, v in test[rate].items()}
    preds = {"Ground truth": sample["target"], "Transformer": sample["coarse"]}
    for kind, title in (("cnn", "CNN residual"), ("fixed", "Fixed graph"), ("motion", "Motion graph")):
        model = ResidualRefiner(kind).to(device)
        model.load_state_dict(torch.load(f"madygraph_gi_plugin_outputs/best_{kind}_refiner.pt", map_location=device))
        model.eval()
        with torch.inference_mode():
            pred = model(sample["coarse"], sample["residual"], sample["motion"]).clamp(0, 1)
        preds[title] = pred
        print(title, "pred_mean/std", float(pred.mean()), float(pred.std()), "target_mean/std", float(sample["target"].mean()), float(sample["target"].std()))
        print(title, "frame0_ssim", float(ssim(pred[:, 0], sample["target"][:, 0], data_range=1.0)))
    fig, axes = plt.subplots(5, 8, figsize=(16, 10))
    for row, (name, video) in enumerate(preds.items()):
        for frame in range(8):
            axes[row, frame].imshow(video[0, frame, 0].cpu(), cmap="gray", vmin=0, vmax=1)
            axes[row, frame].axis("off")
            if frame == 0:
                axes[row, frame].set_ylabel(name)
    fig.suptitle("Held-out dynamic GI, 128 patterns, sequence 0")
    fig.tight_layout()
    path = Path("madygraph_gi_plugin_outputs/test_visuals.png").resolve()
    fig.savefig(path, dpi=160)
    print(f"saved={path}")


if __name__ == "__main__":
    main()

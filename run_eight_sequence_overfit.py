"""Overfit one tiny training subset to validate the refiner learning path."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Subset

HERE = Path(__file__).resolve().parent
PROJECT = Path(r"D:\download\DynGhost-mini-repro")
sys.path[:0] = [str(PROJECT), str(HERE)]

from mini_repro_dynghost import make_patterns
from train_graph_protocol_v2 import IMAGE_SIZE, SEED, make_splits
from train_madygraph_gi_plugin import ResidualRefiner, gi_forward, physical_residual
from train_spatial_motion_diagnostic import SpatialMotionNet
from train_spatiotemporal_transformer import MiniSpatioTemporalTransformer
from train_mixed_sampling import coarse_sequence


def main() -> None:
    torch.manual_seed(20260930)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, _, _ = make_splits(PROJECT)
    rows = [Subset(train_set, range(8))[i] for i in range(8)]
    target = torch.stack([r[0] for r in rows]).to(device)
    transformer = MiniSpatioTemporalTransformer().to(device).eval()
    transformer.load_state_dict(torch.load(HERE / "graph_protocol_v2_trained" / "best_transformer.pt", map_location=device))
    motion_net = SpatialMotionNet().to(device).eval()
    motion_net.load_state_dict(torch.load(HERE / "spatial_motion_outputs" / "best_spatial_motion_net.pt", map_location=device))
    rate = 128
    patterns = make_patterns(rate, IMAGE_SIZE, IMAGE_SIZE, seed=SEED).to(device)
    with torch.inference_mode():
        dgi = coarse_sequence(target, patterns)
        coarse = transformer(dgi)
        bucket = gi_forward(target, patterns)
        residual = physical_residual(bucket, coarse, patterns)
        b, t = target.shape[:2]
        pairs = torch.cat((dgi[:, :-1], dgi[:, 1:]), dim=2)
        motion = motion_net(pairs.reshape(b * (t - 1), 2, IMAGE_SIZE, IMAGE_SIZE)).reshape(b, t - 1, 2)
    model = ResidualRefiner("motion").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    def score(pred):
        mse = (pred.clamp(0, 1) - target).square().mean()
        return float(10 * torch.log10(1 / mse.clamp_min(1e-12)))
    with torch.no_grad():
        initial = score(model(coarse, residual, motion))
    epochs = 30
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(8, device=device)
        total = 0.0
        for start in range(0, 8, 4):
            ids = permutation[start:start + 4]
            pred = model(coarse[ids], residual[ids], motion[ids])
            loss = F.mse_loss(pred, target[ids])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
        with torch.no_grad():
            final = score(model(coarse, residual, motion))
        history.append({"epoch": epoch, "train_mse_batch_mean": total / 2, "train_psnr_db": final})
    payload = {
        "purpose": "training-path overfit check only; same 8 sequences used for optimization and score",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "sequences": 8,
        "rate": rate,
        "epochs": epochs,
        "initial_psnr_db": initial,
        "final_train_psnr_db": final,
        "gain_db": final - initial,
        "history": history,
    }
    out = HERE / "notebook_verification_outputs"
    out.mkdir(exist_ok=True)
    (out / "eight_sequence_overfit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()


from project_paths import PROJECT_ROOT, DATA_DIR as PROJECT_DATA_DIR
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mini_repro_dynghost import IMAGE_SIZE, SEED, make_patterns
from train_frame_cnn import reconstruction_loss
from train_mixed_sampling import RATES, coarse_sequence, evaluate, stats, temporal_error
from train_patch_tgnn import PatchTGNN, TRAIN_N, VAL_N, TEST_N, BATCH, EPOCHS, LR, OUT, set_seed
from train_temporal_convgru import MovingMNISTSequences


def spatial_only_adjacency(seq_len=8, grid=8):
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
                        a[i, t * n_per_frame + rr * grid + cc] = 1.0
    return a


def candidate_indices(grid=8):
    result = []
    for r in range(grid):
        for c in range(grid):
            candidates = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < grid and 0 <= cc < grid:
                        candidates.append(rr * grid + cc)
            result.append(candidates)
    return result


class MotionAwarePatchTGNN(PatchTGNN):
    def __init__(self, dim=32, seq_len=8, grid=8):
        super().__init__(dim=dim, seq_len=seq_len, grid=grid)
        self.register_buffer("spatial_adjacency", spatial_only_adjacency(seq_len, grid))
        self.candidates = candidate_indices(grid)

    def make_adjacency(self, pooled, b, t, dim):
        # Use local feature matching to choose a temporal edge for each patch.
        n_per_frame = self.grid * self.grid
        adjacency = self.spatial_adjacency.unsqueeze(0).expand(b, -1, -1).clone()
        with torch.no_grad():
            features = pooled.reshape(b, t, dim, n_per_frame).permute(0, 1, 3, 2)
            for frame in range(t - 1):
                current = features[:, frame]
                following = features[:, frame + 1]
                for node, candidates in enumerate(self.candidates):
                    candidate = following[:, candidates]
                    distance = (candidate - current[:, node:node + 1]).square().mean(dim=-1)
                    best = distance.argmin(dim=-1)
                    target_indices = torch.as_tensor(candidates, device=pooled.device)[best]
                    src = frame * n_per_frame + node
                    dst = (frame + 1) * n_per_frame + target_indices
                    batch_indices = torch.arange(b, device=pooled.device)
                    adjacency[batch_indices, src, dst] = 1.0
                    adjacency[batch_indices, dst, src] = 1.0
        adjacency = adjacency + torch.eye(adjacency.shape[-1], device=pooled.device)[None]
        degree = adjacency.sum(dim=-1).clamp_min(1.0)
        inv_sqrt = degree.rsqrt()
        return inv_sqrt.unsqueeze(-1) * adjacency * inv_sqrt.unsqueeze(-2)

    def forward(self, x):
        b, t, c, h, w = x.shape
        flat = x.reshape(b * t, c, h, w)
        e1 = self.enc1(flat)
        e2 = self.enc2(e1)
        _, dim, hh, ww = e2.shape
        pooled = F.adaptive_avg_pool2d(e2, (self.grid, self.grid))
        adjacency = self.make_adjacency(pooled, b, t, dim)
        nodes = pooled.reshape(b, t, dim, self.grid * self.grid).permute(0, 1, 3, 2)
        nodes = nodes.reshape(b, t * self.grid * self.grid, dim)
        neighbor = torch.bmm(adjacency, nodes)
        nodes = self.graph1.norm(self.graph1.self_proj(nodes) + self.graph1.neighbor_proj(neighbor))
        nodes = F.gelu(nodes)
        neighbor = torch.bmm(adjacency, nodes)
        nodes = self.graph2.norm(self.graph2.self_proj(nodes) + self.graph2.neighbor_proj(neighbor))
        nodes = F.gelu(nodes)
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


def main():
    set_seed(SEED)
    motion_out = str(PROJECT_ROOT / 'motion_tgnn_outputs')
    os.makedirs(motion_out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    bank = {m: make_patterns(m, IMAGE_SIZE, IMAGE_SIZE, seed=SEED).to(device) for m in RATES}
    train = DataLoader(MovingMNISTSequences(str(PROJECT_DATA_DIR), True, TRAIN_N, SEED + 200),
                       batch_size=BATCH, shuffle=True, num_workers=0)
    val = DataLoader(MovingMNISTSequences(str(PROJECT_DATA_DIR), False, VAL_N, SEED + 201),
                     batch_size=BATCH, shuffle=False, num_workers=0)
    test = DataLoader(MovingMNISTSequences(str(PROJECT_DATA_DIR), False, TEST_N, SEED + 300),
                      batch_size=4, shuffle=False, num_workers=0)
    model = MotionAwarePatchTGNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    best = float("inf")
    history = []
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
            flat = target.reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
            loss = reconstruction_loss(pred.reshape_as(flat), flat) + 0.15 * temporal_error(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * target.shape[0]
            n += target.shape[0]
        values = []
        for pattern_count in RATES:
            values.extend(r["mse"] for r in evaluate(model, "MotionAwarePatchTGNN", val, bank[pattern_count], device))
        val_mse = float(np.mean(values))
        history.append({"epoch": epoch, "train_loss": total / n, "val_mse": val_mse})
        print(f"epoch={epoch:02d}/{EPOCHS} train_loss={total / n:.6f} val_mse={val_mse:.6f}")
        if val_mse < best:
            best = val_mse
            torch.save(model.state_dict(), os.path.join(motion_out, "best_motion_tgnn.pt"))
    model.load_state_dict(torch.load(os.path.join(motion_out, "best_motion_tgnn.pt"), map_location=device))
    summary = []
    for pattern_count in RATES:
        rows = evaluate(model, "MotionAwarePatchTGNN", test, bank[pattern_count], device)
        pm, ps = stats(rows, "psnr_db")
        sm, ss = stats(rows, "ssim")
        em, es = stats(rows, "temporal_difference_error")
        mm, ms = stats(rows, "mse")
        summary.append(dict(pattern_count=pattern_count, sampling_ratio=pattern_count / IMAGE_SIZE**2,
                            method="MotionAwarePatchTGNN", mse_mean=mm, mse_std=ms,
                            psnr_mean_db=pm, psnr_std_db=ps, ssim_mean=sm, ssim_std=ss,
                            temporal_difference_error_mean=em, temporal_difference_error_std=es))
        print(f"patterns={pattern_count:4d} MotionAwarePatchTGNN PSNR={pm:.2f}+/-{ps:.2f} "
              f"SSIM={sm:.3f} temp_err={em:.4f}")
    elapsed = time.time() - start
    with open(os.path.join(motion_out, "results.json"), "w", encoding="utf-8") as handle:
        json.dump({"config": {"seed": SEED, "rates": list(RATES), "train_sequences": TRAIN_N,
                               "val_sequences": VAL_N, "test_sequences": TEST_N, "epochs": EPOCHS,
                               "graph": "4-neighbor spatial edges plus locally matched temporal edges"},
                   "history": history, "summary": summary,
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "elapsed_seconds": elapsed}, handle, ensure_ascii=False, indent=2)
    print(f"elapsed_seconds={elapsed:.1f}")
    print(f"saved={motion_out}")


if __name__ == "__main__":
    main()

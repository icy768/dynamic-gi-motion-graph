"""Unit checks for the E1 physical layer (e1_physics.py).

Shift conventions are checked against motion_pooling/mcgi.shift_matrix and against the
BouncingMovingMNIST generator; the autograd gradient against an explicit dense formula built
from mcgi; the GPU grid search against mcgi.refine_against_canvas on identical inputs.

Run:  python verify_e1_physics.py      (or: python -m pytest verify_e1_physics.py)
"""
from __future__ import annotations

import math
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from project_paths import PROJECT_ROOT
from e1_physics import (
    _warp_all, LayerConfig, PooledObjective, System, frame_mask, motion_epe, physical_layer, positions_from_pairs,
    reference_canvas, refine_pairs, refine_positions, shift, shift_many,
)

sys.path.insert(0, str(PROJECT_ROOT / "motion_pooling"))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import mcgi  # noqa: E402

H = W = 64
D = torch.float64


def _delta(y=8, x=8):
    img = torch.zeros(1, 1, H, W, dtype=D)
    img[0, 0, y, x] = 1.0
    return img


def _digit(seed=3):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(mcgi.render_digit("5", mcgi.FONTS[0], rng)).to(D)


def test_shift_basic_conventions():
    img = _delta()
    assert float((shift(img, torch.zeros(1, 2, dtype=D)) - img).abs().max()) == 0.0
    moved = shift(img, torch.tensor([[3.0, -2.0]], dtype=D))[0, 0]     # content moves +3 in x, -2 in y
    assert torch.nonzero(moved == moved.max()).tolist() == [[6, 11]] and abs(float(moved.max()) - 1) < 1e-12
    half = shift(img, torch.tensor([[0.5, 0.0]], dtype=D))[0, 0]
    assert abs(float(half[8, 8]) - 0.5) < 1e-12 and abs(float(half[8, 9]) - 0.5) < 1e-12


def test_shift_matches_mcgi_shift_matrix():
    rng = np.random.default_rng(0)
    img = torch.rand(1, 1, H, W, dtype=D)
    for _ in range(6):
        dx, dy = rng.uniform(-7, 7, 2)
        ours = shift(img, torch.tensor([[dx, dy]], dtype=D))[0, 0].numpy().ravel()
        ref = mcgi.shift_matrix(dx, dy) @ img[0, 0].numpy().ravel()
        assert np.abs(ours - ref).max() < 1e-12, (dx, dy)


def test_shift_matches_dataset_generator():
    """BouncingMovingMNIST places the digit with affine_grid theta = -2 d / 64 (bilinear)."""
    img = _digit()[None, None]
    for dx, dy in ((3.3, -1.7), (-6.25, 4.5), (0.0, 0.0)):
        theta = torch.tensor([[[1.0, 0.0, -2.0 * dx / H], [0.0, 1.0, -2.0 * dy / H]]], dtype=D)
        grid = F.affine_grid(theta, (1, 1, H, W), align_corners=False)
        ref = F.grid_sample(img, grid, align_corners=False, padding_mode="zeros")
        ours = shift(img, torch.tensor([[dx, dy]], dtype=D))
        assert float((ours - ref).abs().max()) < 1e-12


def test_shift_many_and_canvas_crop():
    img = _digit()[None, None]
    d = torch.tensor([[[1.25, -0.5], [-3.0, 2.75], [0.0, 0.0]]], dtype=D)
    many = shift_many(img, d, (H, W))
    for k in range(3):
        assert float((many[0, k] - shift(img, d[:, k])[0, 0]).abs().max()) < 1e-12
    pad = 16
    canvas = shift(img, torch.zeros(1, 2, dtype=D), out_hw=(H + 2 * pad, W + 2 * pad), origin=(-pad, -pad))
    back = shift(canvas, torch.zeros(1, 2, dtype=D), out_hw=(H, W), origin=(pad, pad))
    assert float((back - img).abs().max()) < 1e-12
    moved = shift(canvas, torch.tensor([[2.5, -1.25]], dtype=D), out_hw=(H, W), origin=(pad, pad))
    assert float((moved - shift(img, torch.tensor([[2.5, -1.25]], dtype=D))).abs().max()) < 1e-12


def test_positions_and_epe():
    pairs = torch.tensor([[[1.0, 0.0], [1.0, 0.5], [-1.0, 0.5]]], dtype=D)
    p = positions_from_pairs(pairs)
    assert torch.allclose(p.mean(1), torch.zeros(1, 2, dtype=D))
    assert torch.allclose(p[:, 1:] - p[:, :-1], pairs)
    assert float(motion_epe(p + torch.tensor([5.0, -3.0], dtype=D), p)) < 1e-12


def _small_problem(t=3, m=24, seed=1):
    gen = torch.Generator().manual_seed(seed)
    patterns = (torch.rand((t, m, H, W), generator=gen) > 0.5).to(D)
    p = torch.tensor([[[-1.3, 0.4], [0.2, -0.7], [1.1, 0.3]]], dtype=D)[:, :t]
    x = torch.rand((1, t, H, W), generator=gen, dtype=D)
    bucket = torch.rand((1, t, m), generator=gen, dtype=D)
    return patterns, p, x, bucket


def test_gradient_matches_explicit_adjoint():
    patterns, p, x, bucket = _small_problem()
    t, m = patterns.shape[:2]
    system = System.build(patterns, bucket)
    dcw = torch.tensor([[0.7, 1.3, 2.1]], dtype=D)
    obj = PooledObjective(system, p, frame_mask(t, None, x.device), dc_weight=dcw)
    xg = x.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(obj.per_frame(xg).sum(), xg)
    a = (patterns.reshape(t, m, -1) / 64.0).numpy()
    y = bucket[0].numpy()
    for tt in range(t):
        g = np.zeros(H * W)
        for k in range(t):
            dx, dy = (p[0, k] - p[0, tt]).tolist()
            B = np.asarray((mcgi.shift_matrix(dx, dy).T @ a[k].T).T, dtype=np.float64)  # A_k S(d), float64
            Bc, yc = B - B.mean(0), y[k] - y[k].mean()
            xv = x[0, tt].numpy().ravel()
            g += Bc.T @ (Bc @ xv - yc)
            g += float(dcw[0, tt]) * m * B.mean(0) * (B.mean(0) @ xv - y[k].mean())
        err = np.abs(grad[0, tt].numpy().ravel() - g).max() / np.abs(g).max()
        assert err < 1e-10, (tt, err)


def test_margin_model_matches_frame_model_inside():
    """With content away from the border, the extended-grid model equals the frame-only model."""
    patterns, p, x, bucket = _small_problem()
    x = x.clone()
    x[..., :8, :] = 0; x[..., -8:, :] = 0; x[..., :, :8] = 0; x[..., :, -8:] = 0
    system = System.build(patterns, bucket)
    mask = frame_mask(3, None, x.device)
    plain = PooledObjective(system, p, mask).residuals(x)
    padded = PooledObjective(system, p, mask, pad=20).residuals(F.pad(x, (20, 20, 20, 20)))
    for a, b in zip(plain, padded):
        assert float((a - b).abs().max()) < 1e-12


def test_margin_sees_content_outside_frame():
    """A frame whose digit is cut by the border is predicted from a neighbour that sees it whole."""
    base = _digit()[None, None]
    p = torch.tensor([[[-18.0, 0.0], [18.0, 0.0]]], dtype=D)          # both frames cut the digit at the border
    frames = torch.stack([shift(base, p[:, k])[0, 0] for k in range(2)])[None]
    canvas0 = shift(base, p[:, 0], out_hw=(H + 40, W + 40), origin=(-20, -20))   # frame 0 plus margin
    pred = _warp_all(canvas0[:, 0][:, None].expand(1, 2, H + 40, W + 40).contiguous(), p, pad=20)[0, 0, 1]
    assert float((pred - frames[0, 1]).abs().max()) < 1e-9
    pred_plain = _warp_all(frames[:, :1].expand(1, 2, H, W).contiguous(), p)[0, 0, 1]
    assert float((pred_plain - frames[0, 1]).abs().max()) > 0.2      # without margin the content is lost


def test_layer_fits_measurements():
    """Noise-free, true motion: the layer must drive the pooled measurement residual down."""
    gen = torch.Generator().manual_seed(5)
    t, m = 8, 128
    patterns = (torch.rand((t, m, H, W), generator=gen) > 0.5).to(D)
    pairs = torch.tensor([[[1.5, -0.75]] * (t - 1)], dtype=D)
    p = positions_from_pairs(pairs)
    base = _digit()[None, None]
    frames = torch.stack([shift(base, p[:, k])[0, 0] for k in range(t)])[None]
    bucket = torch.einsum("btn,tmn->btm", frames.flatten(2), patterns.flatten(2) / 64.0)
    coarse = (frames + 0.15 * torch.randn(frames.shape, generator=gen, dtype=D)).clamp(0, 1)
    system = System.build(patterns, bucket)

    def resid(z):
        pred = torch.einsum("btn,tmn->btm", z.flatten(2), patterns.flatten(2) / 64.0)
        return float((torch.linalg.vector_norm(pred - bucket, dim=-1) / torch.linalg.vector_norm(bucket, dim=-1)).mean())

    def psnr(z):
        return float(10 * torch.log10(1 / (z - frames).square().mean()))

    out = physical_layer(coarse, system, p, LayerConfig(iters=200, mu=1e-3))
    assert resid(out) < 0.2 * resid(coarse), (resid(coarse), resid(out))
    assert psnr(out) > psnr(coarse) + 3, (psnr(coarse), psnr(out))
    own = physical_layer(coarse, system, p, LayerConfig(iters=200, mu=1e-3, window=0))
    assert psnr(out) > psnr(own), "pooling all frames should beat own-frame data only"


def test_grid_search_matches_mcgi():
    gen = torch.Generator().manual_seed(9)
    t, m = 4, 96
    rng = np.random.default_rng(2)
    canvas_np = np.clip(np.pad(mcgi.render_digit("3", mcgi.FONTS[0], rng), 8), 0, 1)
    shape = canvas_np.shape
    origin = (8, 8)
    p_true = np.array([[-2.2, 1.1], [-0.6, 0.4], [0.9, -0.3], [1.9, -1.2]])
    patterns = (torch.rand((t, m, H, W), generator=gen) > 0.5).to(D)
    a = (patterns.reshape(t, m, -1) / 64.0)
    a_np = [a[k].numpy() for k in range(t)]
    ys = [a_np[k] @ (mcgi.crop_shift_matrix(*p_true[k], shape, origin) @ canvas_np.ravel()) for k in range(t)]
    start = p_true + rng.uniform(-0.6, 0.6, p_true.shape)
    ref = mcgi.refine_against_canvas(a_np, ys, canvas_np, shape, origin, start.copy())
    ours = refine_positions(torch.from_numpy(canvas_np)[None], origin, a,
                            torch.from_numpy(np.stack(ys))[None], torch.from_numpy(start)[None],
                            ((0.75, 0.125), (0.15, 0.03)))[0].numpy()
    assert np.abs(ours - ref).max() < 1e-9, (ours, ref)
    assert np.abs(ours - p_true).max() < 0.05


def test_pairwise_registration_recovers_motion():
    """Start up to 3 px off; registration against the (clean) previous frame finds each step."""
    gen = torch.Generator().manual_seed(11)
    t, m = 8, 128
    patterns = (torch.rand((t, m, H, W), generator=gen) > 0.5).to(D)
    a = patterns.reshape(t, m, -1) / 64.0
    truth_pairs = torch.tensor([[[2.3, -1.1], [2.3, -1.1], [-2.3, -1.1], [-2.3, 1.1],
                                 [-2.3, 1.1], [2.3, 1.1], [2.3, 1.1]]], dtype=D)
    p = positions_from_pairs(truth_pairs)
    base = _digit()[None, None]
    frames = torch.stack([shift(base, p[:, k])[0, 0] for k in range(t)])[None]
    bucket = torch.einsum("btn,tmn->btm", frames.flatten(2), a)
    start = truth_pairs + (torch.rand(truth_pairs.shape, generator=gen, dtype=D) * 6 - 3)
    found = refine_pairs(frames, a, bucket, start, ((4.0, 0.5), (0.5, 0.125), (0.125, 0.03)))
    assert float((found - truth_pairs).abs().max()) < 0.05, (found - truth_pairs)


def test_reference_canvas_recovers_frames():
    base = _digit()[None, None]
    p = positions_from_pairs(torch.tensor([[[2.0, 1.0]] * 7], dtype=D))
    frames = torch.stack([shift(base, p[:, k])[0, 0] for k in range(8)])[None]
    canvas = reference_canvas(frames, p, pad=16)
    crop = shift(canvas[:, None], p[:, 3], out_hw=(H, W), origin=(16, 16))[0, 0]
    inner = (slice(12, 52), slice(12, 52))
    assert float((crop[inner] - frames[0, 3][inner]).abs().max()) < 0.05


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"all {len(tests)} checks passed")

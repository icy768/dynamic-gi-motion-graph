"""Checks for per-frame speckle patterns (route C / E1).

Patterns may be (M,H,W), one set shared by every frame (all experiments up to scaleup_v1),
or (T,M,H,W), an independent set per frame. These checks show that
  * the shared bank is bit-identical to the one used before this change;
  * every per-frame operator equals running the shared operator frame by frame;
  * a per-frame bank whose frames are all the same set reproduces the shared results;
  * the per-frame forward model and back-projection are exact adjoints.

Run:  python verify_per_frame_patterns.py     (or: python -m pytest verify_per_frame_patterns.py)
"""
from __future__ import annotations

import hashlib
import math

import torch

from mini_repro_dynghost import SEED, make_patterns
from train_madygraph_gi_plugin import dgi_from_bucket, gi_forward, physical_residual
from train_mixed_sampling import coarse_sequence

H = W = 64
T = 8
DTYPE = torch.float64
TOL = 1e-10

# sha256 of make_patterns(M, 64, 64, seed=SEED) as uint8, recorded before per-frame support was added.
SHARED_BANK_SHA256 = {
    128: "c77315b6ec2ad72b6a9d33465929012a7b19e763a1030a392cf6b9e6bd394d9d",
    256: "0b676a2eb098d83cbaf7ce330526e716e97a38b0dcb32afe7b1860e17c7109d3",
}


def _sha(p: torch.Tensor) -> str:
    return hashlib.sha256(p.to(torch.uint8).numpy().tobytes()).hexdigest()


def _data(m=128, batch=3, seed=20261119):
    gen = torch.Generator().manual_seed(seed)
    per_frame = (torch.rand((T, m, H, W), generator=gen) > 0.5).to(DTYPE)
    target = torch.rand((batch, T, 1, H, W), generator=gen, dtype=DTYPE)
    coarse = torch.rand((batch, T, 1, H, W), generator=gen, dtype=DTYPE)
    return per_frame, target, coarse


def _max_err(a, b) -> float:
    return float((a - b).abs().max())


def test_shared_bank_unchanged():
    for m, expected in SHARED_BANK_SHA256.items():
        p = make_patterns(m, H, W, seed=SEED)
        assert p.shape == (m, H, W)
        assert _sha(p) == expected, f"shared {m}-pattern bank changed"


def test_per_frame_bank_shape_and_independence():
    for m in SHARED_BANK_SHA256:
        bank = make_patterns(m, H, W, seed=SEED, frames=T)
        assert bank.shape == (T, m, H, W)
        assert set(bank.unique().tolist()) <= {0.0, 1.0}
        assert abs(float(bank.mean()) - 0.5) < 0.01
        # frame 0 is the old shared set; the other frames are new, independent draws
        assert torch.equal(bank[0], make_patterns(m, H, W, seed=SEED))
        for a in range(T):
            for b in range(a + 1, T):
                agree = float((bank[a] == bank[b]).float().mean())
                assert 0.48 < agree < 0.52, f"frames {a},{b} agree on {agree:.3f} of entries"


def test_make_bank_modes():
    from gi_experiment import make_bank
    shared = make_bank(torch.device("cpu"))
    per_frame = make_bank(torch.device("cpu"), per_frame=True)
    for m in (128, 256):
        assert shared[m].shape == (m, H, W) and _sha(shared[m]) == SHARED_BANK_SHA256[m]
        assert per_frame[m].shape == (T, m, H, W)


def test_per_frame_equals_frame_by_frame():
    patterns, target, coarse = _data()
    bucket = gi_forward(target, patterns)
    residual = physical_residual(bucket, coarse, patterns)
    dgi = dgi_from_bucket(bucket, patterns)
    seq = coarse_sequence(target, patterns)
    for t in range(T):
        one = slice(t, t + 1)
        b_t = gi_forward(target[:, one], patterns[t])
        assert _max_err(bucket[:, one], b_t) < TOL
        assert _max_err(residual[:, one], physical_residual(b_t, coarse[:, one], patterns[t])) < TOL
        assert _max_err(dgi[:, one], dgi_from_bucket(b_t, patterns[t])) < TOL
        assert _max_err(seq[:, one], coarse_sequence(target[:, one], patterns[t])) < TOL


def test_identical_frames_reproduce_shared():
    patterns, target, coarse = _data()
    shared = patterns[0]
    repeated = shared.expand(T, -1, -1, -1).contiguous()
    b_s, b_r = gi_forward(target, shared), gi_forward(target, repeated)
    assert _max_err(b_s, b_r) < TOL
    assert _max_err(physical_residual(b_s, coarse, shared), physical_residual(b_r, coarse, repeated)) < TOL
    assert _max_err(dgi_from_bucket(b_s, shared), dgi_from_bucket(b_r, repeated)) < TOL
    assert _max_err(coarse_sequence(target, shared), coarse_sequence(target, repeated)) < TOL


def test_dgi_paths_agree():
    """The bucket-based DGI used by the experiments equals the image-based coarse_sequence."""
    for m in (128, 256):
        patterns, target, _ = _data(m=m)
        assert _max_err(dgi_from_bucket(gi_forward(target, patterns), patterns),
                        coarse_sequence(target, patterns)) < 1e-6


def test_per_frame_adjoint_and_residual_formula():
    patterns, target, coarse = _data()
    m = patterns.shape[1]
    matrix = patterns.reshape(T, m, -1) / math.sqrt(H * W)
    x = torch.randn((2, T, 1, H, W), dtype=DTYPE)
    z = torch.randn((2, T, m), dtype=DTYPE)
    left = float((gi_forward(x, patterns) * z).sum())
    back = torch.einsum("btm,tmn->btn", z, matrix)
    right = float((x.reshape(2, T, -1) * back).sum())
    assert abs(left - right) / max(abs(left), abs(right), 1.0) < 1e-12
    bucket = gi_forward(target, patterns)
    explicit = torch.stack([(bucket[:, t] - coarse[:, t].reshape(3, -1) @ matrix[t].T) @ matrix[t] / m
                            for t in range(T)], dim=1)
    assert _max_err(physical_residual(bucket, coarse, patterns).flatten(2), explicit) < TOL


def test_frame_count_mismatch_raises():
    patterns, target, coarse = _data()
    wrong = patterns[:T - 1]
    for call in (lambda: gi_forward(target, wrong),
                 lambda: physical_residual(gi_forward(target, patterns), coarse, wrong),
                 lambda: dgi_from_bucket(gi_forward(target, patterns), wrong),
                 lambda: coarse_sequence(target, wrong)):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("mismatched frame count was not rejected")


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"all {len(tests)} checks passed")

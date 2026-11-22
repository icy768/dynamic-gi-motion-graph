"""E1 physical layer: motion-compensated pooling of per-frame bucket measurements (PyTorch).

For target frame t, every frame k's buckets are a measurement of x_t moved by p_k - p_t:

    y_k = A_k S(p_k - p_t) x_t                 S(d): content moves by +d pixels (bilinear, zeros outside)

The layer starts from the coarse Transformer image x_co and runs box-constrained FISTA on

    1/2 sum_k ||A_k S(p_k-p_t) x - y_k||^2  +  mu/2 ||x - x_co||^2  +  tv * TV_eps(x)

with the exact adjoint from autograd. The bright "DC" direction of 0/1 speckles is split off and
down-weighted exactly as motion_pooling/mcgi.centred_system does, otherwise the step size collapses.

Motion: positions p (B,T,2) start from MotionNet pair motion (cumulative sum) and are refined in the
measurement domain by grid search against a reference canvas built from the current reconstruction,
with a free gain/offset per frame -- the GPU version of mcgi.refine_against_canvas.

Conventions match motion_pooling/mcgi.py and the BouncingMovingMNIST generator: shifts are (dx, dy)
in pixels of the 64x64 frame, content moves by +d, pixel centres at integer indices.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------- sampling
def _base_grid(height, width, device, dtype):
    ii = torch.arange(height, device=device, dtype=dtype)
    jj = torch.arange(width, device=device, dtype=dtype)
    return torch.meshgrid(ii, jj, indexing="ij")


def sample(img: torch.Tensor, ys: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    """Bilinear sample img (N,1,Hi,Wi) at pixel coordinates ys, xs (N,Ho,Wo); zeros outside."""
    hi, wi = img.shape[-2:]
    grid = torch.stack(((xs + 0.5) / wi * 2 - 1, (ys + 0.5) / hi * 2 - 1), dim=-1)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def shift(img: torch.Tensor, d: torch.Tensor, out_hw=None, origin=(0.0, 0.0)) -> torch.Tensor:
    """out[i,j] = img[i + oy - dy, j + ox - dx].

    img (N,1,Hi,Wi); d (N,2) as (dx,dy). With origin (0,0) and out_hw=None this is S(d): content
    moves by +d. A canvas whose frame pixel (0,0) sits at canvas index (oy,ox) is cropped to a frame
    with out_hw=(H,W), origin=(oy,ox); a frame is placed on a canvas with origin=(-oy,-ox).
    """
    ho, wo = out_hw or img.shape[-2:]
    ii, jj = _base_grid(ho, wo, img.device, img.dtype)
    ys = ii[None] + origin[0] - d[:, 1, None, None]
    xs = jj[None] + origin[1] - d[:, 0, None, None]
    return sample(img, ys, xs)


def shift_many(img: torch.Tensor, d: torch.Tensor, out_hw, origin=(0.0, 0.0)) -> torch.Tensor:
    """Like shift, but K shifts per image without copying it: img (N,1,Hi,Wi), d (N,K,2) -> (N,K,Ho,Wo)."""
    n, k, _ = d.shape
    ho, wo = out_hw
    ii, jj = _base_grid(ho, wo, img.device, img.dtype)
    ys = ii[None, None] + origin[0] - d[..., 1, None, None]                 # (N,K,Ho,Wo)
    xs = jj[None, None] + origin[1] - d[..., 0, None, None]
    out = sample(img, ys.reshape(n, k * ho, wo), xs.reshape(n, k * ho, wo))
    return out.reshape(n, k, ho, wo)


def positions_from_pairs(pair_motion: torch.Tensor) -> torch.Tensor:
    """(B,T-1,2) per-step motion -> (B,T,2) positions with zero mean over frames."""
    p = torch.cat((torch.zeros_like(pair_motion[:, :1]), pair_motion.cumsum(1)), dim=1)
    return p - p.mean(1, keepdim=True)


def motion_epe(p_est: torch.Tensor, p_true: torch.Tensor) -> torch.Tensor:
    """Per-sequence mean end-point error of positions after removing the global offset."""
    a = p_est - p_est.mean(1, keepdim=True)
    b = p_true - p_true.mean(1, keepdim=True)
    return torch.linalg.vector_norm(a - b, dim=-1).mean(1)


# ----------------------------------------------------------------------------- measurement system
@dataclass
class System:
    """Per-frame operator A_k = P_k / sqrt(N), split into centred rows and one DC row per frame."""
    a_c: torch.Tensor      # (T,M,N) centred rows A_k - 1 abar_k^T
    abar: torch.Tensor     # (T,N) mean row of A_k
    y_c: torch.Tensor      # (B,T,M) y_k - mean(y_k)
    ybar: torch.Tensor     # (B,T)
    count: int             # M

    @staticmethod
    def build(patterns: torch.Tensor, bucket: torch.Tensor) -> "System":
        frames, m, h, w = patterns.shape
        a = patterns.reshape(frames, m, h * w).to(bucket.dtype) / math.sqrt(h * w)
        abar = a.mean(1)
        return System(a - abar[:, None], abar, bucket - bucket.mean(2, keepdim=True), bucket.mean(2), m)


def _warp_all(x: torch.Tensor, p: torch.Tensor, pad: int = 0) -> torch.Tensor:
    """x (B,T,H+2pad,W+2pad) -> W[b,t,k] = frame k as predicted from x_t, shape (B,T,T,H,W).

    With pad > 0, x_t is frame t plus a margin: frame k is the 64x64 crop of S(p_k - p_t) x_t, so
    content that other frames see outside frame t's view is still modelled.
    """
    b, t, hc, wc = x.shape
    h, w = hc - 2 * pad, wc - 2 * pad
    d = (p[:, None, :, :] - p[:, :, None, :]).reshape(b * t, t, 2)      # [b,t,k] = p_k - p_t
    out = shift_many(x.reshape(b * t, 1, hc, wc), d, (h, w), origin=(pad, pad))
    return out.reshape(b, t, t, h, w)


def _tv(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Smoothed isotropic TV with forward differences (zero at the last row/column), per image."""
    gx = F.pad(x[..., :, 1:] - x[..., :, :-1], (0, 1))
    gy = F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(gx * gx + gy * gy + eps * eps).sum((-2, -1))


class PooledObjective:
    """Data term of the physical layer for all target frames of a batch at once."""

    def __init__(self, system: System, p: torch.Tensor, frame_mask: torch.Tensor, dc_weight=None, pad: int = 0):
        self.s, self.p, self.mask = system, p, frame_mask        # mask (T,T): use frame k for target t
        self.dc_weight = dc_weight
        self.pad = pad

    def residuals(self, x):
        warped = _warp_all(x, self.p, self.pad).flatten(-2)                      # (B,T,K,N)
        r_c = torch.einsum("btkn,kmn->btkm", warped, self.s.a_c) - self.s.y_c[:, None]
        r_dc = torch.einsum("btkn,kn->btk", warped, self.s.abar) - self.s.ybar[:, None]
        return r_c, r_dc

    def per_frame(self, x, centred=1.0, dc=None):
        r_c, r_dc = self.residuals(x)
        mask = self.mask.to(x.dtype)
        f = 0.5 * centred * (r_c.square().sum(-1) * mask).sum(-1)
        w = self.dc_weight if dc is None else dc
        return f + 0.5 * w * self.s.count * (r_dc.square() * mask).sum(-1)

    def quadratic(self, v, centred, dc):
        """Same operator with zero data (for power iteration)."""
        warped = _warp_all(v, self.p, self.pad).flatten(-2)
        q_c = torch.einsum("btkn,kmn->btkm", warped, self.s.a_c).square().sum(-1)
        q_dc = torch.einsum("btkn,kn->btk", warped, self.s.abar).square()
        mask = self.mask.to(v.dtype)
        return 0.5 * ((centred * q_c + dc * self.s.count * q_dc) * mask).sum(-1)


def _grad(fn, x):
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        (g,) = torch.autograd.grad(fn(x).sum(), x)
    return g


def _power(obj: PooledObjective, shape, device, dtype, centred, dc, iters=20, seed=0):
    """Largest eigenvalue of the Hessian of obj.quadratic per (b,t)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(shape, generator=gen, dtype=dtype).to(device)
    v = v / torch.linalg.vector_norm(v.flatten(2), dim=-1)[..., None, None]
    lam = torch.ones(shape[:2], device=device, dtype=dtype)
    for _ in range(iters):
        hv = _grad(lambda z: obj.quadratic(z, centred, dc), v)
        lam = torch.linalg.vector_norm(hv.flatten(2), dim=-1)
        v = hv / lam.clamp_min(1e-20)[..., None, None]
    return lam


# ----------------------------------------------------------------------------- physical layer
@dataclass
class LayerConfig:
    iters: int = 100            # FISTA steps per solve (step size 1/L, L from power iteration)
    mu: float = 0.01            # pull towards the coarse image
    tv: float = 0.0             # smoothed TV weight
    tv_eps: float = 0.02
    window: int | None = None   # use frames |k-t| <= window; None = all frames, 0 = own frame only
    start: str = "coarse"       # FISTA start: "coarse" image or "zero"
    pad: int = 20               # margin around each frame's unknown (>= largest relative shift); 0 = frame only


def frame_mask(frames: int, window, device) -> torch.Tensor:
    idx = torch.arange(frames, device=device)
    if window is None:
        return torch.ones(frames, frames, dtype=torch.bool, device=device)
    return (idx[:, None] - idx[None, :]).abs() <= window


def physical_layer(x_co: torch.Tensor, system: System, p: torch.Tensor, cfg: LayerConfig) -> torch.Tensor:
    """x_co (B,T,H,W) coarse frames, p (B,T,2) positions -> refined frames (B,T,H,W) in [0,1]."""
    b, t, h, w = x_co.shape
    pad = cfg.pad
    centre = (Ellipsis, slice(pad, pad + h), slice(pad, pad + w))
    mask = frame_mask(t, cfg.window, x_co.device)
    obj = PooledObjective(system, p, mask, pad=pad)
    shape = (b, t, h + 2 * pad, w + 2 * pad)
    with torch.no_grad():
        l_c = _power(obj, shape, x_co.device, x_co.dtype, 1.0, 0.0)
        l_dc = _power(obj, shape, x_co.device, x_co.dtype, 0.0, 1.0)
        obj.dc_weight = l_c / l_dc.clamp_min(1e-20)            # DC block gets the centred block's curvature
        lipschitz = 2.0 * 1.05 * l_c + cfg.mu + cfg.tv * 8.0 / cfg.tv_eps
        step = (1.0 / lipschitz)[..., None, None]

    def objective(z):
        f = obj.per_frame(z)
        if cfg.mu:                                              # pull only where the coarse image exists
            f = f + 0.5 * cfg.mu * (z[centre] - x_co).square().sum((-2, -1))
        if cfg.tv:
            f = f + cfg.tv * _tv(z, cfg.tv_eps)
        return f

    x = torch.zeros(shape, device=x_co.device, dtype=x_co.dtype)
    if cfg.start == "coarse":
        x[centre] = x_co
    z, tk = x.clone(), 1.0
    for _ in range(cfg.iters):
        x_new = (z - step * _grad(objective, z)).clamp(0.0, 1.0)
        t_new = 0.5 * (1 + math.sqrt(1 + 4 * tk * tk))
        z = x_new + ((tk - 1) / t_new) * (x_new - x)
        x, tk = x_new, t_new
    return x[centre].detach().contiguous()


# ----------------------------------------------------------------------------- motion refinement
@dataclass
class MotionConfig:
    rounds: int = 8
    # (radius, step) in pixels, searched around the current estimate one after the other
    stages: tuple = ((0.75, 0.125), (0.15, 0.03))
    pad: int = 16
    chunk: int = 2048           # max candidate images per batch element processed at once


def reference_canvas(x: torch.Tensor, p: torch.Tensor, pad: int) -> torch.Tensor:
    """Average of all frames moved back to a common canvas (frame pixel (0,0) at canvas (pad,pad))."""
    b, t, h, w = x.shape
    imgs = x.reshape(b * t, 1, h, w)
    back = shift(imgs, -p.reshape(-1, 2), out_hw=(h + 2 * pad, w + 2 * pad), origin=(-pad, -pad))
    cover = shift(torch.ones_like(imgs), -p.reshape(-1, 2), out_hw=(h + 2 * pad, w + 2 * pad), origin=(-pad, -pad))
    back = back.reshape(b, t, h + 2 * pad, w + 2 * pad)
    cover = cover.reshape(b, t, h + 2 * pad, w + 2 * pad)
    return back.sum(1) / cover.sum(1).clamp_min(1e-6)


def _gain_offset_cost(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """min_{g,c} ||g pred + c - y||^2 along the last axis."""
    pc = pred - pred.mean(-1, keepdim=True)
    yc = y - y.mean(-1, keepdim=True)
    return yc.square().sum(-1) - (pc * yc).sum(-1).square() / pc.square().sum(-1).clamp_min(1e-20)


def grid_offsets(radius, step, device, dtype):
    axis = torch.arange(-radius, radius + 1e-9, step, device=device, dtype=torch.float64).to(dtype)
    gx, gy = torch.meshgrid(axis, axis, indexing="xy")
    return torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)          # (G,2) as (dx,dy)


def refine_positions(canvas, origin, patterns_flat, bucket, p, stages, chunk=2048):
    """Per-frame grid search of p_k so that A_k crop(S(p_k) canvas) explains y_k up to gain/offset.

    canvas (B,Hc,Wc); origin (oy,ox) canvas index of frame pixel (0,0) at zero shift;
    patterns_flat (T,M,N) = A_k; bucket (B,T,M); p (B,T,2). Returns new p (B,T,2).
    """
    b, t, _ = p.shape
    h = w = int(round(math.sqrt(patterns_flat.shape[-1])))
    cur = p.clone()
    for radius, step in stages:
        offs = grid_offsets(radius, step, p.device, p.dtype)
        g = offs.shape[0]
        cand = cur[:, :, None, :] + offs[None, None]                         # (B,T,G,2)
        cost = torch.empty(b, t, g, device=p.device, dtype=p.dtype)
        per = max(1, chunk // t)
        for start in range(0, g, per):
            sl = slice(start, min(g, start + per))
            n = sl.stop - sl.start
            crops = shift_many(canvas[:, None], cand[:, :, sl].reshape(b, t * n, 2), out_hw=(h, w), origin=origin)
            pred = torch.einsum("btgn,tmn->btgm", crops.reshape(b, t, n, h * w), patterns_flat)
            cost[:, :, sl] = _gain_offset_cost(pred, bucket[:, :, None])
        best = cost.argmin(-1)
        cur = torch.gather(cand, 2, best[..., None, None].expand(b, t, 1, 2))[:, :, 0]
    return cur


def refine_pairs(x_ref, patterns_flat, bucket, pairs, stages, chunk=2048):
    """Pairwise measurement-domain registration (GPU version of mcgi.mdr_pair).

    For each step t -> t+1, grid-search d so that A_{t+1} S(d) x_ref[t] explains y_{t+1} up to
    gain/offset, starting from `pairs` (B,T-1,2). x_ref (B,T,H,W) is the coarse reconstruction.
    """
    b, t, h, w = x_ref.shape
    cur = pairs.clone()
    ref = x_ref[:, :-1].reshape(b * (t - 1), 1, h, w)
    a_next = patterns_flat[1:]
    y_next = bucket[:, 1:]
    for radius, step in stages:
        offs = grid_offsets(radius, step, pairs.device, pairs.dtype)
        g = offs.shape[0]
        cand = cur[:, :, None, :] + offs[None, None]                          # (B,T-1,G,2)
        cost = torch.empty(b, t - 1, g, device=pairs.device, dtype=pairs.dtype)
        per = max(1, chunk // (t - 1))
        for start in range(0, g, per):
            sl = slice(start, min(g, start + per))
            n = sl.stop - sl.start
            imgs = shift_many(ref, cand[:, :, sl].reshape(b * (t - 1), n, 2), (h, w))
            pred = torch.einsum("bkgn,kmn->bkgm", imgs.reshape(b, t - 1, n, h * w), a_next)
            cost[:, :, sl] = _gain_offset_cost(pred, y_next[:, :, None])
        best = cost.argmin(-1)
        cur = torch.gather(cand, 2, best[..., None, None].expand(b, t - 1, 1, 2))[:, :, 0]
    return cur


@dataclass
class PoolConfig:
    layer: LayerConfig = field(default_factory=LayerConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    refine: bool = True         # False: keep the starting positions (oracle, or no refinement)
    # pairwise registration of the starting pair motion against the coarse frames; () = skip
    pair_stages: tuple = ((4.0, 0.5), (0.5, 0.125), (0.125, 0.03))

    def as_dict(self):
        return asdict(self)


@torch.no_grad()
def pool_reconstruct(x_co, patterns, bucket, start_pairs, cfg: PoolConfig, p_true=None):
    """Alternate physical layer and measurement-domain motion refinement.

    x_co (B,T,H,W); patterns (T,M,H,W) 0/1; bucket (B,T,M); start_pairs (B,T-1,2) per-step motion
    (MotionNet estimate, or the truth for the oracle). Returns dict(x, p, pairs_start, epe_rounds:
    list of (B,) tensors -- MotionNet start, after pairwise registration, then after each round).
    """
    system = System.build(patterns, bucket)
    a_flat = system.a_c + system.abar[:, None]
    epe = []
    if p_true is not None:
        epe.append(motion_epe(positions_from_pairs(start_pairs), p_true))
    pairs = start_pairs
    if cfg.refine and cfg.pair_stages:
        pairs = refine_pairs(x_co, a_flat, bucket, start_pairs, cfg.pair_stages, cfg.motion.chunk)
        if p_true is not None:
            epe.append(motion_epe(positions_from_pairs(pairs), p_true))
    p = positions_from_pairs(pairs)
    x = physical_layer(x_co, system, p, cfg.layer)
    if cfg.refine:
        pad = cfg.motion.pad
        for _ in range(cfg.motion.rounds):
            canvas = reference_canvas(x, p, pad)
            p = refine_positions(canvas, (pad, pad), a_flat, bucket, p, cfg.motion.stages, cfg.motion.chunk)
            p = p - p.mean(1, keepdim=True)
            if p_true is not None:
                epe.append(motion_epe(p, p_true))
            x = physical_layer(x_co, system, p, cfg.layer)
    return {"x": x, "p": p, "epe_rounds": epe}

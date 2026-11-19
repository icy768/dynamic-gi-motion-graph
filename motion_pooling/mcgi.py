"""Feasibility study: motion-compensated multi-frame measurement pooling for dynamic GI.

Pure NumPy/SciPy (no learning). Mirrors the original project's simulation:
64x64 frames, 8 frames, digit upscaled 28->64, bouncing translation within +/-10 px,
binary speckles, A = P / sqrt(4096), bucket y_t = A_t x_t.

Question: if the inter-frame motion were known (or estimated), how much better can a
frame be reconstructed by pooling measurements from neighbouring frames, compared with
using only its own measurements?
"""
from __future__ import annotations

import math
import numpy as np
import scipy.sparse as sp
from scipy import ndimage
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.path import Path as MPath

H = W = 64
N = H * W
T = 8
SPEED_BINS = ((1.0, 2.0), (2.0, 4.0), (4.0, 6.0))
FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
]


def _resolve_fonts(paths):
    """Use the listed fonts when present (Linux, as in the recorded 2026-11-18 results).
    Otherwise (e.g. Windows) fall back to the DejaVu fonts bundled with matplotlib, so the
    code runs anywhere. NOTE: with the fallback the generated digits differ from the recorded
    run, so numbers will be close but not identical."""
    import os
    import warnings
    import matplotlib
    found = [p for p in paths if os.path.exists(p)]
    if len(found) == len(paths):
        return found
    bundled = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    extra = [os.path.join(bundled, n) for n in
             ("DejaVuSans.ttf", "DejaVuSerif.ttf", "DejaVuSans-Bold.ttf",
              "DejaVuSansMono.ttf", "DejaVuSerif-Bold.ttf", "DejaVuSans-Oblique.ttf")]
    extra = [p for p in extra if os.path.exists(p)]
    warnings.warn("Some fonts in FONTS were not found; using matplotlib's bundled DejaVu fonts. "
                  "Generated digits will differ from the recorded results.")
    out = found + [p for p in extra if p not in found]
    return out[:len(paths)]


FONTS = _resolve_fonts(FONTS)


# ----------------------------------------------------------------------------- data
def render_digit(ch: str, font: str, rng: np.random.Generator) -> np.ndarray:
    """MNIST-like 28x28 digit (20x20 box, centred), then bilinear upscale to 64x64."""
    fp = FontProperties(fname=font)
    tp = TextPath((0, 0), ch, size=1.0, prop=fp)
    verts = tp.vertices
    lo, hi = verts.min(0), verts.max(0)
    s = 20.0 / max(hi - lo)
    ang = rng.uniform(-0.25, 0.25)
    shear = rng.uniform(-0.2, 0.2)
    R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    R = R @ np.array([[1.0, shear], [0.0, 1.0]])
    hi_res = 4
    G = 28 * hi_res
    yy, xx = np.mgrid[0:G, 0:G]
    pts = np.stack([(xx.ravel() + 0.5) / hi_res, (yy.ravel() + 0.5) / hi_res], 1)
    # map pixel coords back to glyph coords
    centre = (lo + hi) / 2
    q = (pts - 14.0)
    q[:, 1] = -q[:, 1]
    q = q @ np.linalg.inv(R).T / s + centre
    path = MPath(verts, tp.codes)
    inside = path.contains_points(q).reshape(G, G).astype(np.float64)
    img = inside.reshape(28, hi_res, 28, hi_res).mean((1, 3))
    img = ndimage.gaussian_filter(img, 0.5)
    img = img / max(img.max(), 1e-6)
    up = ndimage.zoom(img, 64 / 28, order=1)
    return np.clip(up[:64, :64], 0, 1)


def trajectory(rng, speed, max_offset=10.0, steps=T, substeps=1):
    """Bouncing positions (dx, dy) at times 0, 1/substeps, ..., matching the project."""
    x, y = rng.uniform(-max_offset, max_offset, 2)
    a = rng.uniform(0, 2 * math.pi)
    vx, vy = speed * math.cos(a), speed * math.sin(a)
    pos = []
    for t in range(steps):
        pos.append((x, y))
        if x + vx < -max_offset or x + vx > max_offset:
            vx = -vx
        if y + vy < -max_offset or y + vy > max_offset:
            vy = -vy
        x += vx
        y += vy
    return np.array(pos), (vx, vy)


def shift_image(img, dx, dy, order=3):
    """Translate content by (+dx, +dy) pixels, zeros outside. Cubic for data generation."""
    return np.clip(ndimage.shift(img, (dy, dx), order=order, mode="constant", cval=0.0), 0, 1)


def make_patterns(M, rng):
    return (rng.random((M, N)) < 0.5).astype(np.float32)


# ----------------------------------------------------------------------------- operators
def shift_matrix(dx, dy):
    """Sparse bilinear operator S with (S x)[i,j] = x[i-dy, j-dx] (content moves by +d)."""
    ii, jj = np.mgrid[0:H, 0:W]
    sy, sx = ii - dy, jj - dx
    y0, x0 = np.floor(sy).astype(int), np.floor(sx).astype(int)
    fy, fx = sy - y0, sx - x0
    rows, cols, vals = [], [], []
    out = (ii * W + jj).ravel()
    for oy, ox, w in ((0, 0, (1 - fy) * (1 - fx)), (0, 1, (1 - fy) * fx),
                      (1, 0, fy * (1 - fx)), (1, 1, fy * fx)):
        yy, xx = y0 + oy, x0 + ox
        ok = ((yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)).ravel()
        rows.append(out[ok]); cols.append((yy * W + xx).ravel()[ok]); vals.append(w.ravel()[ok])
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(N, N))


def warped_operator(A, dx, dy):
    """Rows of A applied to S(d) x  ->  dense (A S)."""
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return A
    S = shift_matrix(dx, dy)
    return np.asarray((S.T @ A.T).T, dtype=np.float32)


# ----------------------------------------------------------------------------- solvers
def grad2(x):
    gx = np.zeros_like(x); gy = np.zeros_like(x)
    gx[:, :-1] = x[:, 1:] - x[:, :-1]
    gy[:-1, :] = x[1:, :] - x[:-1, :]
    return gx, gy


def div2(px, py):
    d = np.zeros_like(px)
    d[:, :-1] += px[:, :-1]; d[:, 1:] -= px[:, :-1]
    d[:-1, :] += py[:-1, :]; d[1:, :] -= py[:-1, :]
    return d


def power_norm(A, iters=30, rng=None):
    v = np.ones(A.shape[1], np.float32) / math.sqrt(A.shape[1])
    for _ in range(iters):
        v = A.T @ (A @ v); n = np.linalg.norm(v); v /= n
    return float(n)


def tv_recon(A, y, lam, iters=300, eps=0.02, x0=None):
    """FISTA on 0.5||Ax-y||^2 + lam * sum sqrt(|grad x|^2 + eps^2), box [0,1]."""
    L = power_norm(A) + lam * 8.0 / eps
    step = 1.0 / L
    x = np.zeros(N, np.float32) if x0 is None else x0.astype(np.float32).ravel().copy()
    z = x.copy(); t = 1.0
    for _ in range(iters):
        r = A @ z - y
        g = A.T @ r
        Z = z.reshape(H, W)
        gx, gy = grad2(Z)
        mag = np.sqrt(gx * gx + gy * gy + eps * eps)
        g = g - lam * div2(gx / mag, gy / mag).ravel()
        x_new = np.clip(z - step * g, 0.0, 1.0)
        t_new = 0.5 * (1 + math.sqrt(1 + 4 * t * t))
        z = x_new + ((t - 1) / t_new) * (x_new - x)
        x, t = x_new, t_new
    return x.reshape(H, W)


def dgi(P, y):
    """Differential ghost imaging per frame, min-max normalised (as in the project)."""
    Pm = P.mean(0)
    s = P.sum(1)
    yn = y - y.mean() / s.mean() * s
    img = ((yn - yn.mean())[:, None] * (P - Pm)).mean(0).reshape(H, W)
    img = img - img.min()
    return img / max(img.max(), 1e-9)


# ----------------------------------------------------------------------------- motion
def phase_shift(a, b, up=20):
    """Estimate (dx, dy) such that b ~= a shifted by (dx, dy). Sub-pixel by local upsampling."""
    Fa, Fb = np.fft.fft2(a - a.mean()), np.fft.fft2(b - b.mean())
    R = Fb * np.conj(Fa); R /= np.abs(R) + 1e-9
    c = np.real(np.fft.ifft2(R))
    py, px = np.unravel_index(np.argmax(c), c.shape)
    # refine with matrix-multiply DFT around the peak
    def upsampled(py0, px0):
        span = 1.5
        ys = py0 + np.linspace(-span, span, int(2 * span * up) + 1)
        xs = px0 + np.linspace(-span, span, int(2 * span * up) + 1)
        ky = np.fft.fftfreq(H); kx = np.fft.fftfreq(W)
        Ey = np.exp(2j * np.pi * np.outer(ys, ky)); Ex = np.exp(2j * np.pi * np.outer(kx, xs))
        cc = np.real(Ey @ R @ Ex) / N
        iy, ix = np.unravel_index(np.argmax(cc), cc.shape)
        return ys[iy], xs[ix]
    sy, sx = upsampled(py, px)
    if sy > H / 2: sy -= H
    if sx > W / 2: sx -= W
    return sx, sy


# ----------------------------------------------------------------------------- metrics
def psnr(x, ref):
    mse = float(np.mean((np.clip(x, 0, 1) - ref) ** 2))
    return 10 * math.log10(1.0 / max(mse, 1e-12))


def meas_residual(A, x, y):
    return float(np.linalg.norm(A @ x.ravel() - y) / np.linalg.norm(y))


# ----------------------------------------------------------------------------- better-conditioned solver
def centred_system(blocks, dc_weight=None):
    """blocks: list of (B_k [M_k x N], y_k). Exact split per block:
    ||B x - y||^2 = ||B_c x - y_c||^2 + M_k (b_k^T x - ybar_k)^2,  B_c = B - 1 b_k^T.
    The DC rows are kept but down-weighted so they do not dominate the step size."""
    Bc, yc, D, d, m = [], [], [], [], []
    for B, y in blocks:
        b = B.mean(0)
        Bc.append(B - b[None, :]); yc.append(y - y.mean())
        D.append(b); d.append(y.mean()); m.append(B.shape[0])
    Bc = np.concatenate(Bc).astype(np.float32); yc = np.concatenate(yc).astype(np.float32)
    D = np.stack(D).astype(np.float32); d = np.array(d, np.float32); m = np.array(m, np.float32)
    Lc = power_norm(Bc)
    if dc_weight is None:
        # scale so the DC block has about the same curvature as the centred block
        dc_weight = Lc / max(power_norm(D * np.sqrt(m)[:, None]), 1e-9)
    wD = (D * np.sqrt(m * dc_weight)[:, None]).astype(np.float32)
    wd = (d * np.sqrt(m * dc_weight)).astype(np.float32)
    return np.concatenate([Bc, wD]), np.concatenate([yc, wd])


def tv_blocks(blocks, lam, iters=400, eps=0.02):
    A, y = centred_system(blocks)
    return tv_recon(A, y, lam, iters=iters, eps=eps)


# ----------------------------------------------------------------------------- canonical-canvas pooling
def tv_recon_shape(A, y, lam, shape, iters=300, eps=0.02):
    """Same FISTA/TV solver, arbitrary image shape."""
    h, w = shape
    n = h * w
    L = power_norm(A) + lam * 8.0 / eps
    step = 1.0 / L
    x = np.zeros(n, np.float32); z = x.copy(); t = 1.0
    for _ in range(iters):
        g = A.T @ (A @ z - y)
        Z = z.reshape(h, w)
        gx, gy = grad2(Z)
        mag = np.sqrt(gx * gx + gy * gy + eps * eps)
        g = g - lam * div2(gx / mag, gy / mag).ravel()
        x_new = np.clip(z - step * g, 0.0, 1.0)
        t_new = 0.5 * (1 + math.sqrt(1 + 4 * t * t))
        z = x_new + ((t - 1) / t_new) * (x_new - x)
        x, t = x_new, t_new
    return x.reshape(h, w)


def crop_shift_matrix(dx, dy, canvas_shape, origin):
    """Sparse (N x n_canvas) operator: frame pixel (i,j) = canvas[i - dy + oy, j - dx + ox] (bilinear).
    origin=(oy, ox) is the canvas index of frame pixel (0,0) when the shift is zero."""
    ch, cw = canvas_shape
    oy, ox = origin
    ii, jj = np.mgrid[0:H, 0:W]
    sy, sx = ii - dy + oy, jj - dx + ox
    y0, x0 = np.floor(sy).astype(int), np.floor(sx).astype(int)
    fy, fx = sy - y0, sx - x0
    rows, cols, vals = [], [], []
    out = (ii * W + jj).ravel()
    for a, b, wgt in ((0, 0, (1 - fy) * (1 - fx)), (0, 1, (1 - fy) * fx),
                      (1, 0, fy * (1 - fx)), (1, 1, fy * fx)):
        yy, xx = y0 + a, x0 + b
        ok = ((yy >= 0) & (yy < ch) & (xx >= 0) & (xx < cw)).ravel()
        rows.append(out[ok]); cols.append((yy * cw + xx).ravel()[ok]); vals.append(wgt.ravel()[ok])
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(N, ch * cw))


def canvas_for(shifts, margin=2):
    """shifts: (T,2) relative displacements (dx,dy) of each frame w.r.t. the canvas reference."""
    dx, dy = shifts[:, 0], shifts[:, 1]
    oy = int(math.ceil(max(0.0, dy.max()))) + margin
    ox = int(math.ceil(max(0.0, dx.max()))) + margin
    ch = oy + H + int(math.ceil(max(0.0, -dy.min()))) + margin
    cw = ox + W + int(math.ceil(max(0.0, -dx.min()))) + margin
    return (ch, cw), (oy, ox)


def pooled_recon(As, ys, shifts, lam, iters=300, frames=None):
    """Reconstruct one canvas from all frames' buckets given relative shifts; return per-frame crops."""
    shape, origin = canvas_for(shifts)
    ops = [crop_shift_matrix(d[0], d[1], shape, origin) for d in shifts]
    use = range(len(As)) if frames is None else frames
    blocks = [(np.asarray((ops[k].T @ As[k].T).T, np.float32), ys[k]) for k in use]
    Ab, yb = centred_system(blocks)
    canvas = tv_recon_shape(Ab, yb, lam, shape, iters=iters)
    crops = np.stack([(op @ canvas.ravel()).reshape(H, W) for op in ops])
    return crops, canvas, ops


def refine_shifts_by_measurement(As, ys, canvas, shape, origin, shifts, radius=1.0, step=0.25):
    """For each frame, grid-search the displacement that best explains its own buckets
    given the current canvas estimate (works directly in the measurement domain)."""
    out = shifts.copy()
    grid = np.arange(-radius, radius + 1e-9, step)
    c = canvas.ravel()
    for k in range(len(As)):
        best, arg = np.inf, shifts[k]
        for gx in grid:
            for gy in grid:
                d = shifts[k] + (gx, gy)
                op = crop_shift_matrix(d[0], d[1], shape, origin)
                r = As[k] @ (op @ c) - ys[k]
                v = float(r @ r)
                if v < best:
                    best, arg = v, d
        out[k] = arg
    return out


# ----------------------------------------------------------------------------- measurement-domain registration
def shifted_frame(img, dx, dy):
    return (shift_matrix(dx, dy) @ img.ravel())


def mdr_pair(A_next, y_next, ref_img, init=(0.0, 0.0), coarse_radius=8, fine=(1.0, 0.1)):
    """Find d so that A_next S(d) ref_img best predicts y_next (least squares, with a free
    gain/offset to absorb the reference image's intensity error)."""
    def cost(d):
        p = A_next @ shifted_frame(ref_img, d[0], d[1])
        X = np.stack([p, np.ones_like(p)], 1)
        coef, *_ = np.linalg.lstsq(X, y_next, rcond=None)
        r = X @ coef - y_next
        return float(r @ r)
    best = (np.inf, np.array(init, float))
    base = np.round(np.array(init))
    for gx in range(-coarse_radius, coarse_radius + 1):
        for gy in range(-coarse_radius, coarse_radius + 1):
            d = base + (gx, gy)
            c = cost(d)
            if c < best[0]:
                best = (c, d)
    d0 = best[1]
    rad, st = fine
    for scale in (rad, rad / 4):
        stp = st if scale == rad else st / 4
        grid = np.arange(-scale, scale + 1e-9, max(stp, 0.025))
        cur = best[1].copy()
        for gx in grid:
            for gy in grid:
                d = cur + (gx, gy)
                c = cost(d)
                if c < best[0]:
                    best = (c, d)
    return best[1]


def refine_against_canvas(As, ys, canvas, shape, origin, shifts, stages=((0.75, 0.125), (0.15, 0.03))):
    """Per-frame measurement-domain refinement against the pooled canvas (with gain/offset)."""
    c = canvas.ravel()
    out = shifts.copy()
    for k in range(len(As)):
        def cost(d):
            p = As[k] @ (crop_shift_matrix(d[0], d[1], shape, origin) @ c)
            X = np.stack([p, np.ones_like(p)], 1)
            coef, *_ = np.linalg.lstsq(X, ys[k], rcond=None)
            r = X @ coef - ys[k]
            return float(r @ r)
        cur = out[k].copy(); bc = cost(cur)
        for rad, st in stages:
            centre = cur.copy()
            for gx in np.arange(-rad, rad + 1e-9, st):
                for gy in np.arange(-rad, rad + 1e-9, st):
                    d = centre + (gx, gy)
                    v = cost(d)
                    if v < bc:
                        bc, cur = v, d
        out[k] = cur
    return out

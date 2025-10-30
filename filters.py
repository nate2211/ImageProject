# filters.py — self-contained warping generator (registers itself)
# -----------------------------------------------------------------------------
# This module provides a WarpGenerator that displaces pixels using procedural
# flow fields (curl noise, FBM, radial, swirl). It is self-contained and hooks
# into your existing CLI via the shared palettes.REGISTRY, just like content.py.
#
# Usage (examples):
#   # gentle curl-noise warp
#   python main.py run \
#     --url input.jpg \
#     --pipeline "warp" \
#     --out out.png \
#     --extra warp.strength_px=8 warp.mode=curl warp.edge_protect=0.75
#
#   # swirl around center with falloff
#   python main.py run \
#     --url input.jpg \
#     --pipeline "warp" \
#     --out out.png \
#     --extra warp.mode=swirl warp.strength_px=12 warp.swirl_turns=1.25
#
#   # radial push outward from 40%/60% image position
#   python main.py run \
#     --url input.jpg \
#     --pipeline "warp" \
#     --out out.png \
#     --extra warp.mode=radial warp.center_x=0.4 warp.center_y=0.6 warp.sign=1
#
# Notes:
# - Warping changes geometry by definition. This generator includes guards
#   (edge_protect, strength clamps, smoothing) to keep it tasteful by default.
# - All parameters are accessible via `warp.PARAM=value` as shown above.
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import math

import numpy as np
from PIL import Image

# Import the shared registry and base generator
from palettes import REGISTRY, BaseGenerator  # type: ignore

__all__ = ["WarpGenerator"]

# ============================ low-level helpers ============================

def _box_blur_gray(field: np.ndarray, k: int) -> np.ndarray:
    """Fast separable box blur for 2D float32 arrays."""
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1
    r = k // 2
    fp = np.pad(field, ((0, 0), (r, r)), mode="reflect")
    c = np.pad(fp, ((0, 0), (1, 0)), mode="constant").cumsum(axis=1, dtype=np.float64)
    horiz = (c[:, k:] - c[:, :-k]) / k
    fp2 = np.pad(horiz, ((r, r), (0, 0)), mode="reflect")
    c2 = np.pad(fp2, ((1, 0), (0, 0)), mode="constant").cumsum(axis=0, dtype=np.float64)
    vert = (c2[k:, :] - c2[:-k, :]) / k
    return vert.astype(np.float32)


def _to_gray01(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    return (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]) / 255.0


def _edge_magnitude(gray01: np.ndarray) -> np.ndarray:
    """Sobel edge strength in [0,1]."""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)

    def conv2d(img: np.ndarray, k: np.ndarray) -> np.ndarray:
        pad = k.shape[0] // 2
        im = np.pad(img, pad, mode="reflect")
        out = np.empty_like(img)
        for y in range(out.shape[0]):
            ys = slice(y, y + 2 * pad + 1)
            row = im[ys]
            for x in range(out.shape[1]):
                xs = slice(x, x + 2 * pad + 1)
                out[y, x] = float((row[:, xs] * k).sum())
        return out

    gx = conv2d(gray01, kx)
    gy = conv2d(gray01, ky)
    mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    m = float(mag.max())
    return (mag / m) if m > 1e-8 else mag


def _fbm(h: int, w: int, *, octaves: int = 5, persistence: float = 0.6, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    acc = np.zeros((h, w), np.float32)
    amp = 1.0
    for o in range(octaves):
        scale = 2 ** (octaves - o - 1)
        hh = max(1, h // scale)
        ww = max(1, w // scale)
        base = rng.random((hh, ww), dtype=np.float32)
        up = np.kron(base, np.ones((int(np.ceil(h / hh)), int(np.ceil(w / ww))), dtype=np.float32))[:h, :w]
        up = _box_blur_gray(up, k=3)
        acc += amp * up
        amp *= persistence
    acc -= acc.min()
    acc /= (acc.max() + 1e-8)
    return acc


def _curl_noise_flow(h: int, w: int, *, octaves: int = 5, persistence: float = 0.6, rng: np.random.Generator | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Divergence-free flow (u,v) via curl of a scalar FBM potential."""
    n = _fbm(h, w, octaves=octaves, persistence=persistence, rng=rng)
    dn_dx = np.zeros_like(n)
    dn_dy = np.zeros_like(n)
    dn_dx[:, 1:-1] = 0.5 * (n[:, 2:] - n[:, :-2])
    dn_dx[:, 0] = n[:, 1] - n[:, 0]
    dn_dx[:, -1] = n[:, -1] - n[:, -2]
    dn_dy[1:-1, :] = 0.5 * (n[2:, :] - n[:-2, :])
    dn_dy[0, :] = n[1, :] - n[0, :]
    dn_dy[-1, :] = n[-1, :] - n[-2, :]
    u = -dn_dy
    v =  dn_dx
    norm = max(np.sqrt(u*u + v*v).mean(), 1e-6)
    return (u / norm).astype(np.float32), (v / norm).astype(np.float32)


def _fbm_flow(h: int, w: int, *, octaves: int = 5, persistence: float = 0.6, rng: np.random.Generator | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Simple FBM-based flow from two independent noises (not divergence-free)."""
    rng = rng or np.random.default_rng()
    u = _fbm(h, w, octaves=octaves, persistence=persistence, rng=rng)
    v = _fbm(h, w, octaves=octaves, persistence=persistence, rng=rng)
    return (2.0 * (u - 0.5)).astype(np.float32), (2.0 * (v - 0.5)).astype(np.float32)


def _bilinear_sample(img_arr: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """Vectorized bilinear resampling of HxWx3 image using float maps (x,y in pixel space)."""
    H, W, _ = img_arr.shape
    x0 = np.floor(map_x).astype(np.int32)
    y0 = np.floor(map_y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    x0 = np.clip(x0, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1)

    wa = (x1 - map_x) * (y1 - map_y)
    wb = (map_x - x0) * (y1 - map_y)
    wc = (x1 - map_x) * (map_y - y0)
    wd = (map_x - x0) * (map_y - y0)

    Ia = img_arr[y0, x0]
    Ib = img_arr[y0, x1]
    Ic = img_arr[y1, x0]
    Id = img_arr[y1, x1]

    out = Ia * wa[..., None] + Ib * wb[..., None] + Ic * wc[..., None] + Id * wd[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_flow_warp(
    img: Image.Image,
    u: np.ndarray,
    v: np.ndarray,
    *,
    strength_px: float,
    edge_mask: np.ndarray | None,
    edge_protect: float,
    smooth_k: int,
) -> Image.Image:
    """Displace pixels using flow (u,v) scaled by strength_px with optional edge protection."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    H, W, _ = arr.shape

    mag = np.sqrt(u*u + v*v) + 1e-8
    u_n = u / mag
    v_n = v / mag

    disp_u = strength_px * u_n
    disp_v = strength_px * v_n

    if smooth_k > 1:
        disp_u = _box_blur_gray(disp_u, k=smooth_k)
        disp_v = _box_blur_gray(disp_v, k=smooth_k)

    if edge_mask is not None and edge_protect > 0:
        # Reduce warp where edges are strong
        protect = 1.0 - edge_protect * np.clip(edge_mask, 0.0, 1.0)  # 1 in flats, ~0 at edges
        disp_u *= protect
        disp_v *= protect

    grid_x, grid_y = np.meshgrid(np.arange(W, dtype=np.float32),
                                 np.arange(H, dtype=np.float32))
    map_x = np.clip(grid_x + disp_u, 0, W - 1)
    map_y = np.clip(grid_y + disp_v, 0, H - 1)
    warped = _bilinear_sample(arr, map_x, map_y)
    return Image.fromarray(warped, "RGB")


# ============================ WarpGenerator ============================

@dataclass
class WarpGenerator(BaseGenerator):
    """
    Procedural pixel warp with safety guards.

    Parameters (extras)
    -------------------
    mode: str                = 'curl'   # 'curl' | 'fbm' | 'radial' | 'swirl'
    strength_px: float       = 8.0      # approximate max pixel displacement
    smooth_k: int            = 3        # blur displacement (odd kernel)
    edge_protect: float      = 0.75     # 0..1, reduce warp near edges
    edges: str               = 'auto'   # 'auto' | 'none'
    octaves: int             = 5        # for noise modes
    persistence: float       = 0.60     # for noise modes

    # radial / swirl parameters
    center_x: float          = 0.5      # in [0,1] relative image width
    center_y: float          = 0.5      # in [0,1] relative image height
    falloff: float           = 1.4      # strength decays with radius^falloff
    sign: int                = 1        # +1 outward / CCW, -1 inward / CW (mode-dependent)
    swirl_turns: float       = 1.0      # number of full rotations at unit radius (swirl)
    swirl_inner: float       = 0.05     # dead zone radius (0..1 of min(width,height))
    swirl_outer: float       = 1.0      # outer radius clamp (0..1 of min(width,height))
    """
    def generate(self, input_image: Image.Image, **kwargs) -> Image.Image:
        img = input_image.convert("RGB")
        W, H = img.size

        mode            = str(kwargs.get("mode", "curl")).lower()
        strength_px     = float(kwargs.get("strength_px", 8.0))
        smooth_k        = int(kwargs.get("smooth_k", 3))
        edge_protect    = float(kwargs.get("edge_protect", 0.75))
        edges           = str(kwargs.get("edges", "auto")).lower()

        octaves         = int(kwargs.get("octaves", 5))
        persistence     = float(kwargs.get("persistence", 0.60))

        cx              = float(kwargs.get("center_x", 0.5)) * (W - 1)
        cy              = float(kwargs.get("center_y", 0.5)) * (H - 1)
        falloff         = float(kwargs.get("falloff", 1.4))
        sign            = int(kwargs.get("sign", 1))
        swirl_turns     = float(kwargs.get("swirl_turns", 1.0))
        swirl_inner     = float(kwargs.get("swirl_inner", 0.05))
        swirl_outer     = float(kwargs.get("swirl_outer", 1.0))

        # Clamp and guards
        strength_px = float(np.clip(strength_px, 0.0, max(W, H) * 0.25))
        edge_protect = float(np.clip(edge_protect, 0.0, 1.0))
        smooth_k = max(1, int(smooth_k) | 1)

        # Edge mask (optional)
        edge_mask = None
        if edges == "auto":
            gray = _to_gray01(img)
            edge_mask = _box_blur_gray(_edge_magnitude(gray), k=3)

        # Build flow
        if mode in ("curl", "fbm"):
            rng = np.random.default_rng(self.seed)
            if mode == "curl":
                u, v = _curl_noise_flow(H, W, octaves=octaves, persistence=persistence, rng=rng)
            else:
                u, v = _fbm_flow(H, W, octaves=octaves, persistence=persistence, rng=rng)
        elif mode in ("radial", "swirl"):
            # coordinate grid centered at (cx, cy)
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            dx = xx - cx
            dy = yy - cy
            r = np.sqrt(dx*dx + dy*dy) + 1e-6
            r_norm = r / (min(W, H) * 0.5)
            r_norm = np.clip(r_norm, 0.0, 1.0)

            fall = (1.0 - r_norm) ** max(0.2, falloff)  # stronger near center, decays to edge
            fall = _box_blur_gray(fall, k=3)

            if mode == "radial":
                # outward/inward along radius (sign = +1/-1)
                u = sign * (dx / r) * fall
                v = sign * (dy / r) * fall
            else:
                # swirl: perpendicular to radius, magnitude ~ turns * radius with inner/outer clamp
                inner = float(np.clip(swirl_inner, 0.0, 1.0))
                outer = float(np.clip(swirl_outer, inner + 1e-6, 1.0))
                mask = (r_norm >= inner) & (r_norm <= outer)
                ang_per_rad = (2.0 * math.pi * swirl_turns)
                # tangential unit vector
                tx = -dy / r
                ty =  dx / r
                # magnitude grows with radius inside [inner, outer]
                g = (np.clip((r_norm - inner) / (outer - inner), 0.0, 1.0)) * ang_per_rad
                g *= fall
                u = np.where(mask, sign * tx * g, 0.0).astype(np.float32)
                v = np.where(mask, sign * ty * g, 0.0).astype(np.float32)
        else:
            # default to curl if unknown
            rng = np.random.default_rng(self.seed)
            u, v = _curl_noise_flow(H, W, octaves=octaves, persistence=persistence, rng=rng)

        # Apply warp
        out = _apply_flow_warp(
            img,
            u, v,
            strength_px=strength_px,
            edge_mask=edge_mask,
            edge_protect=edge_protect,
            smooth_k=smooth_k,
        )
        return out


# Register with the shared registry so `--pipeline warp` works
REGISTRY.register("warp", WarpGenerator)
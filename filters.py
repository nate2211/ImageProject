# filters.py — self-contained warping generator (registers itself)
# -----------------------------------------------------------------------------
# Rewritten with introspection for GUI parameter discovery.
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any
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


def _fbm(h: int, w: int, *, octaves: int = 5, persistence: float = 0.6,
         rng: np.random.Generator | None = None) -> np.ndarray:
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


def _curl_noise_flow(h: int, w: int, *, octaves: int = 5, persistence: float = 0.6,
                     rng: np.random.Generator | None = None) -> Tuple[np.ndarray, np.ndarray]:
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
    v = dn_dx
    norm = max(np.sqrt(u * u + v * v).mean(), 1e-6)
    return (u / norm).astype(np.float32), (v / norm).astype(np.float32)


def _fbm_flow(h: int, w: int, *, octaves: int = 5, persistence: float = 0.6, rng: np.random.Generator | None = None) -> \
Tuple[np.ndarray, np.ndarray]:
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

    mag = np.sqrt(u * u + v * v) + 1e-8
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
    """

    @staticmethod
    def get_params() -> List[Dict[str, Any]]:
        return [
            {
                "name": "mode",
                "type": str,
                "default": "curl",
                "choices": ["curl", "fbm", "radial", "swirl"],
                "help": "Type of flow field used for distortion."
            },
            {
                "name": "strength_px",
                "type": float,
                "default": 8.0,
                "min": 0.0, "max": 100.0,
                "help": "Max pixel displacement distance."
            },
            {
                "name": "edge_protect",
                "type": float,
                "default": 0.75,
                "min": 0.0, "max": 1.0,
                "help": "Reduces warp effect on detected edges to preserve content."
            },
            {
                "name": "smooth_k",
                "type": int,
                "default": 3,
                "min": 1, "max": 15,
                "help": "Blur kernel size for flow smoothing."
            },
            {
                "name": "edges",
                "type": str,
                "default": "auto",
                "choices": ["auto", "none"],
                "help": "Edge detection mode."
            },
            # Noise-specific
            {
                "name": "octaves",
                "type": int,
                "default": 5,
                "min": 1, "max": 8,
                "help": "Noise detail layers (for curl/fbm)."
            },
            {
                "name": "persistence",
                "type": float,
                "default": 0.60,
                "min": 0.0, "max": 1.0,
                "help": "Noise roughness (for curl/fbm)."
            },
            # Radial/Swirl geometry
            {
                "name": "center_x",
                "type": float,
                "default": 0.5,
                "min": 0.0, "max": 1.0,
                "help": "Center X position (0.0=left, 1.0=right)."
            },
            {
                "name": "center_y",
                "type": float,
                "default": 0.5,
                "min": 0.0, "max": 1.0,
                "help": "Center Y position (0.0=top, 1.0=bottom)."
            },
            {
                "name": "falloff",
                "type": float,
                "default": 1.4,
                "min": 0.1, "max": 5.0,
                "help": "Strength decay exponent for radial effects."
            },
            {
                "name": "swirl_turns",
                "type": float,
                "default": 1.0,
                "min": -5.0, "max": 5.0,
                "help": "Number of rotations for swirl mode."
            },
            {
                "name": "swirl_inner",
                "type": float,
                "default": 0.05,
                "min": 0.0, "max": 1.0,
                "help": "Inner dead zone for swirl."
            }
        ]

    def generate(self, input_image: Image.Image, **kwargs) -> Image.Image:
        img = input_image.convert("RGB")
        W, H = img.size

        mode = str(kwargs.get("mode", "curl")).lower()
        strength_px = float(kwargs.get("strength_px", 8.0))
        smooth_k = int(kwargs.get("smooth_k", 3))
        edge_protect = float(kwargs.get("edge_protect", 0.75))
        edges = str(kwargs.get("edges", "auto")).lower()

        octaves = int(kwargs.get("octaves", 5))
        persistence = float(kwargs.get("persistence", 0.60))

        cx = float(kwargs.get("center_x", 0.5)) * (W - 1)
        cy = float(kwargs.get("center_y", 0.5)) * (H - 1)
        falloff = float(kwargs.get("falloff", 1.4))
        sign = int(kwargs.get("sign", 1))
        swirl_turns = float(kwargs.get("swirl_turns", 1.0))
        swirl_inner = float(kwargs.get("swirl_inner", 0.05))
        swirl_outer = float(kwargs.get("swirl_outer", 1.0))

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
            r = np.sqrt(dx * dx + dy * dy) + 1e-6
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
                ty = dx / r
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


# ============================ GlitchGenerator ============================

def _shift_fill(arr: np.ndarray, dx: int, dy: int, *, fill: int = 0) -> np.ndarray:
    """Shift HxWxC array by (dx,dy) with constant fill (no wrap)."""
    H, W, C = arr.shape
    out = np.empty_like(arr)
    out.fill(fill)

    x0_src = max(0, -dx)
    x1_src = min(W, W - dx)  # exclusive
    y0_src = max(0, -dy)
    y1_src = min(H, H - dy)

    x0_dst = max(0, dx)
    x1_dst = min(W, W + dx)
    y0_dst = max(0, dy)
    y1_dst = min(H, H + dy)

    if x1_src <= x0_src or y1_src <= y0_src:
        return out

    out[y0_dst:y1_dst, x0_dst:x1_dst, :] = arr[y0_src:y1_src, x0_src:x1_src, :]
    return out


def _apply_scanlines(arr_f: np.ndarray, *, strength: float, period: int) -> np.ndarray:
    """Darken horizontal scanlines (float32 in [0,255])."""
    if strength <= 0:
        return arr_f
    H = arr_f.shape[0]
    period = max(1, int(period))
    # pattern in [1-strength, 1]
    y = np.arange(H, dtype=np.float32)
    m = 1.0 - strength * (0.5 * (1.0 + np.sin((2.0 * math.pi / period) * y)))
    return arr_f * m[:, None, None]


def _slice_displace(arr: np.ndarray, rng: np.random.Generator, *,
                    slices: int, max_shift_px: int, vertical: bool) -> np.ndarray:
    """Randomly displace horizontal (or vertical) slices with constant fill."""
    H, W, _ = arr.shape
    out = arr.copy()

    slices = max(0, int(slices))
    if slices == 0 or max_shift_px <= 0:
        return out

    max_shift_px = int(max_shift_px)

    if not vertical:
        # Horizontal strips shifted left/right
        for _ in range(slices):
            y0 = int(rng.integers(0, H))
            h = int(rng.integers(max(1, H // 64), max(2, H // 12)))
            y1 = min(H, y0 + h)
            dx = int(rng.integers(-max_shift_px, max_shift_px + 1))
            strip = out[y0:y1, :, :]
            out[y0:y1, :, :] = _shift_fill(strip, dx, 0, fill=0)
    else:
        # Vertical strips shifted up/down
        for _ in range(slices):
            x0 = int(rng.integers(0, W))
            w = int(rng.integers(max(1, W // 64), max(2, W // 12)))
            x1 = min(W, x0 + w)
            dy = int(rng.integers(-max_shift_px, max_shift_px + 1))
            strip = out[:, x0:x1, :]
            out[:, x0:x1, :] = _shift_fill(strip, 0, dy, fill=0)

    return out


def _add_noise(arr_f: np.ndarray, rng: np.random.Generator, *, amount: float) -> np.ndarray:
    """Add uniform noise in [-amount..amount] * 255 to float32 array in [0..255]."""
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0:
        return arr_f
    n = (rng.random(arr_f.shape, dtype=np.float32) - 0.5) * 2.0
    return arr_f + n * (amount * 255.0)


def _parse_color(s: Any, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    Accepts:
      - "255,0,0"
      - "#ff0000"
      - "rgb(255,0,0)"
    Returns (r,g,b) ints 0..255.
    """
    try:
        if s is None:
            return default
        txt = str(s).strip()

        if txt.lower().startswith("rgb"):
            # rgb( r,g,b )
            txt = txt[txt.find("(") + 1 : txt.rfind(")")].strip()

        if txt.startswith("#"):
            h = txt[1:].strip()
            if len(h) == 6:
                r = int(h[0:2], 16)
                g = int(h[2:4], 16)
                b = int(h[4:6], 16)
                return (r, g, b)
            return default

        if "," in txt:
            parts = [p.strip() for p in txt.split(",")]
            if len(parts) != 3:
                return default
            r = int(float(parts[0])); g = int(float(parts[1])); b = int(float(parts[2]))
            r = int(np.clip(r, 0, 255)); g = int(np.clip(g, 0, 255)); b = int(np.clip(b, 0, 255))
            return (r, g, b)

        return default
    except Exception:
        return default

@dataclass
class GlitchGenerator(BaseGenerator):
    """
    Digital glitch effect (same as original):
      - random slice displacement
      - RGB channel splitting (chromatic offset)
      - scanlines
      - additive noise

    NEW:
      - choose the 3 colors used for the split layers
        (layer_from_R -> color_r, layer_from_G -> color_g, layer_from_B -> color_b)
    """

    @staticmethod
    def get_params() -> List[Dict[str, Any]]:
        return [
            {"name": "intensity", "type": float, "default": 0.7, "min": 0.0, "max": 1.0,
             "help": "Overall effect strength scaler."},

            {"name": "slice_count", "type": int, "default": 14, "min": 0, "max": 200,
             "help": "Number of displaced slices."},

            {"name": "slice_shift_px", "type": int, "default": 24, "min": 0, "max": 600,
             "help": "Max pixels to shift each slice."},

            {"name": "slice_vertical", "type": bool, "default": False,
             "help": "Use vertical slices instead of horizontal."},

            {"name": "rgb_shift_px", "type": int, "default": 6, "min": 0, "max": 120,
             "help": "Channel split offset in pixels."},

            # --- NEW: selectable colors for the split layers (defaults = RGB) ---
            {"name": "color_r", "type": "color", "default": "255,0,0",
             "help": "Color for the shifted 'R' layer. Formats: '255,0,0' or '#ff0000' or 'rgb(255,0,0)'."},
            {"name": "color_g", "type": "color", "default": "0,255,0",
             "help": "Color for the 'G' layer. Formats: '0,255,0' or '#00ff00' or 'rgb(0,255,0)'."},
            {"name": "color_b", "type": "color", "default": "0,0,255",
             "help": "Color for the shifted 'B' layer. Formats: '0,0,255' or '#0000ff' or 'rgb(0,0,255)'."},
            # -------------------------------------------------------------------

            {"name": "noise", "type": float, "default": 0.03, "min": 0.0, "max": 0.5,
             "help": "Additive noise amount."},

            {"name": "scanline_strength", "type": float, "default": 0.12, "min": 0.0, "max": 0.8,
             "help": "Scanline darkening strength."},

            {"name": "scanline_period", "type": int, "default": 3, "min": 1, "max": 32,
             "help": "Scanline spacing (rows per cycle)."},

            {"name": "luma_jitter", "type": float, "default": 0.04, "min": 0.0, "max": 0.5,
             "help": "Brightness jitter (like unstable signal gain)."},
        ]

    def generate(self, input_image: Image.Image, **kwargs) -> Image.Image:
        img = input_image.convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
        H, W, _ = arr.shape

        rng = np.random.default_rng(self.seed)

        intensity = float(np.clip(kwargs.get("intensity", 0.7), 0.0, 1.0))

        slice_count = int(kwargs.get("slice_count", 14))
        slice_shift_px = int(kwargs.get("slice_shift_px", 24))
        slice_vertical = bool(kwargs.get("slice_vertical", False))

        rgb_shift_px = int(kwargs.get("rgb_shift_px", 6))
        noise_amt = float(kwargs.get("noise", 0.03))
        scan_s = float(kwargs.get("scanline_strength", 0.12))
        scan_p = int(kwargs.get("scanline_period", 3))
        luma_jitter = float(kwargs.get("luma_jitter", 0.04))

        # NEW: split-layer colors (defaults RGB)
        col_r = _parse_color(kwargs.get("color_r", "255,0,0"), (255, 0, 0))
        col_g = _parse_color(kwargs.get("color_g", "0,255,0"), (0, 255, 0))
        col_b = _parse_color(kwargs.get("color_b", "0,0,255"), (0, 0, 255))

        # Scale effect knobs by intensity (same as original)
        slice_count = int(slice_count * (0.25 + 0.75 * intensity))
        slice_shift_px = int(slice_shift_px * intensity)
        rgb_shift_px = int(rgb_shift_px * intensity)
        noise_amt = float(noise_amt * intensity)
        scan_s = float(scan_s * intensity)
        luma_jitter = float(luma_jitter * intensity)

        # 1) Slice displacement (same as original)
        out = _slice_displace(
            arr, rng,
            slices=slice_count,
            max_shift_px=slice_shift_px,
            vertical=slice_vertical
        )

        # 2) RGB split (same spatial behavior), but recolor the 3 layers
        if rgb_shift_px > 0:
            dx = int(rng.integers(-rgb_shift_px, rgb_shift_px + 1))
            dy = int(rng.integers(-rgb_shift_px, rgb_shift_px + 1))

            # same source layers as original
            r_layer = _shift_fill(out, dx, dy, fill=0)[:, :, 0].astype(np.float32) / 255.0
            g_layer = out[:, :, 1].astype(np.float32) / 255.0
            b_layer = _shift_fill(out, -dx, -dy, fill=0)[:, :, 2].astype(np.float32) / 255.0

            # recolor them (this keeps the "old look" but swaps the hues)
            cr = np.array(col_r, dtype=np.float32) / 255.0
            cg = np.array(col_g, dtype=np.float32) / 255.0
            cb = np.array(col_b, dtype=np.float32) / 255.0

            out_f2 = (
                r_layer[..., None] * cr[None, None, :] +
                g_layer[..., None] * cg[None, None, :] +
                b_layer[..., None] * cb[None, None, :]
            ) * 255.0

            out = np.clip(out_f2, 0.0, 255.0).astype(np.uint8)

        # 3) Convert to float for scanlines/noise/gain jitter (same as original)
        out_f = out.astype(np.float32)

        # 4) Unstable luma gain (same)
        if luma_jitter > 0:
            g = 1.0 + (rng.random((H,), dtype=np.float32) - 0.5) * 2.0 * luma_jitter
            if H > 8:
                g2 = g.copy()
                g2[1:-1] = (g[:-2] + g[1:-1] + g[2:]) / 3.0
                g = g2
            out_f *= g[:, None, None]

        # 5) Scanlines (same)
        out_f = _apply_scanlines(out_f, strength=scan_s, period=scan_p)

        # 6) Noise (same)
        out_f = _add_noise(out_f, rng, amount=noise_amt)

        out_f = np.clip(out_f, 0.0, 255.0)
        return Image.fromarray(out_f.astype(np.uint8), "RGB")


# Register with the shared registry so `--pipeline glitch` works
REGISTRY.register("glitch", GlitchGenerator)

# ============================ DisplaceMapGenerator ============================
# Classic "Displacement Map" / "Bump-map Distortion" (well-known technique)

def _bilinear_sample2(img_arr: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, *, wrap: bool) -> np.ndarray:
    """Vectorized bilinear sampling with optional wrap."""
    H, W, _ = img_arr.shape

    if wrap:
        map_x = np.mod(map_x, W)
        map_y = np.mod(map_y, H)

        x0 = np.floor(map_x).astype(np.int32) % W
        y0 = np.floor(map_y).astype(np.int32) % H
        x1 = (x0 + 1) % W
        y1 = (y0 + 1) % H
    else:
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


def _grad_xy(field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Central-diff gradient for 2D float32 field."""
    gy = np.zeros_like(field)
    gx = np.zeros_like(field)
    gx[:, 1:-1] = 0.5 * (field[:, 2:] - field[:, :-2])
    gx[:, 0] = field[:, 1] - field[:, 0]
    gx[:, -1] = field[:, -1] - field[:, -2]
    gy[1:-1, :] = 0.5 * (field[2:, :] - field[:-2, :])
    gy[0, :] = field[1, :] - field[0, :]
    gy[-1, :] = field[-1, :] - field[-2, :]
    return gx.astype(np.float32), gy.astype(np.float32)


@dataclass
class DisplaceMapGenerator(BaseGenerator):
    """
    Displacement Map / Bump-map Distortion (well-known technique):
      - builds a displacement field from luma (height), luma gradient, or procedural noise
      - displaces pixels and resamples with bilinear filtering

    This is a "legit" classic effect used in many editors + VFX tools.
    """

    @staticmethod
    def get_params() -> List[Dict[str, Any]]:
        return [
            {
                "name": "mode",
                "type": str,
                "default": "luma_grad",
                "choices": ["luma", "luma_grad", "noise_rg", "curl"],
                "help": "Displacement source: luma height, luma gradient (bump-map), noise (RG), or curl flow."
            },
            {
                "name": "strength_x_px",
                "type": float,
                "default": 12.0,
                "min": 0.0, "max": 150.0,
                "help": "Horizontal displacement strength (pixels)."
            },
            {
                "name": "strength_y_px",
                "type": float,
                "default": 12.0,
                "min": 0.0, "max": 150.0,
                "help": "Vertical displacement strength (pixels)."
            },
            {
                "name": "map_blur_k",
                "type": int,
                "default": 5,
                "min": 1, "max": 31,
                "help": "Blur kernel for displacement map (higher = smoother)."
            },
            {
                "name": "field_smooth_k",
                "type": int,
                "default": 3,
                "min": 1, "max": 31,
                "help": "Extra smoothing on dx/dy fields."
            },
            {
                "name": "octaves",
                "type": int,
                "default": 5,
                "min": 1, "max": 8,
                "help": "Noise detail layers (noise_rg/curl)."
            },
            {
                "name": "persistence",
                "type": float,
                "default": 0.6,
                "min": 0.0, "max": 1.0,
                "help": "Noise roughness (noise_rg/curl)."
            },
            {
                "name": "wrap",
                "type": bool,
                "default": False,
                "help": "Wrap sampling at edges (True) instead of clamping (False)."
            },
        ]

    def generate(self, input_image: Image.Image, **kwargs) -> Image.Image:
        img = input_image.convert("RGB")
        arr = np.asarray(img, dtype=np.float32)
        H, W, _ = arr.shape

        mode = str(kwargs.get("mode", "luma_grad")).lower()
        sx = float(kwargs.get("strength_x_px", 12.0))
        sy = float(kwargs.get("strength_y_px", 12.0))
        map_blur_k = int(kwargs.get("map_blur_k", 5))
        field_smooth_k = int(kwargs.get("field_smooth_k", 3))
        octaves = int(kwargs.get("octaves", 5))
        persistence = float(kwargs.get("persistence", 0.6))
        wrap = bool(kwargs.get("wrap", False))

        # Guards
        max_s = max(W, H) * 0.25
        sx = float(np.clip(sx, 0.0, max_s))
        sy = float(np.clip(sy, 0.0, max_s))
        map_blur_k = max(1, int(map_blur_k) | 1)
        field_smooth_k = max(1, int(field_smooth_k) | 1)

        # Build displacement field dx, dy in [-1..1] roughly
        if mode in ("luma", "luma_grad"):
            hmap = _to_gray01(img).astype(np.float32)
            if map_blur_k > 1:
                hmap = _box_blur_gray(hmap, k=map_blur_k)

            if mode == "luma":
                # classic displacement-map: height drives both axes (diagonal-ish push/pull)
                d = (hmap - 0.5) * 2.0
                dx, dy = d, d
            else:
                # bump-map style: displacement follows height gradient (common shader/VFX technique)
                gx, gy = _grad_xy(hmap)
                # normalize
                n = np.sqrt(gx * gx + gy * gy) + 1e-6
                dx = gx / n
                dy = gy / n

        else:
            rng = np.random.default_rng(self.seed)
            if mode == "curl":
                dx, dy = _curl_noise_flow(H, W, octaves=octaves, persistence=persistence, rng=rng)
            else:
                # "noise_rg": independent noise for X and Y (classic displacement map approach)
                dx, dy = _fbm_flow(H, W, octaves=octaves, persistence=persistence, rng=rng)

        # Optional smoothing of fields
        if field_smooth_k > 1:
            dx = _box_blur_gray(dx.astype(np.float32), k=field_smooth_k)
            dy = _box_blur_gray(dy.astype(np.float32), k=field_smooth_k)

        # Build pixel maps and sample
        grid_x, grid_y = np.meshgrid(np.arange(W, dtype=np.float32),
                                     np.arange(H, dtype=np.float32))

        map_x = grid_x + dx.astype(np.float32) * sx
        map_y = grid_y + dy.astype(np.float32) * sy

        out = _bilinear_sample2(arr, map_x, map_y, wrap=wrap)
        return Image.fromarray(out, "RGB")


# Register with the shared registry so `--pipeline displace` works
REGISTRY.register("displace", DisplaceMapGenerator)
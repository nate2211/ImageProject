from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import requests
from PIL import Image

# =============== Registry ===============
class GeneratorRegistry:
    def __init__(self) -> None:
        self._by_name: Dict[str, type[BaseGenerator]] = {}

    def register(self, name: str, cls: type["BaseGenerator"]) -> None:
        key = name.strip().lower()
        self._by_name[key] = cls

    def names(self) -> list[str]:
        return sorted(self._by_name.keys())

    def create(self, name: str, **kwargs) -> "BaseGenerator":
        key = name.strip().lower()
        if key not in self._by_name:
            raise KeyError(f"Unknown generator '{name}'. Available: {', '.join(self.names()) or '(none)'}")
        return self._by_name[key](**kwargs)


REGISTRY = GeneratorRegistry()


# =============== Base & common utils ===============
@dataclass
class BaseGenerator:
    seed: Optional[int] = None
    def generate(self, input_image: Image.Image, **kwargs) -> Image.Image:  # pragma: no cover
        raise NotImplementedError


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed if seed is not None else np.random.SeedSequence().entropy)


def _parse_hex_color(code: str) -> Tuple[float, float, float]:
    s = code.strip().lstrip("#")
    if len(s) == 3:
        s = "".join([c * 2 for c in s])
    if len(s) != 6:
        return (17.0, 17.0, 17.0)
    r = int(s[0:2], 16); g = int(s[2:4], 16); b = int(s[4:6], 16)
    return float(r), float(g), float(b)


# --- Blurs (shape-preserving, no SciPy) ---
def _box_blur_gray(field: np.ndarray, k: int) -> np.ndarray:
    k = max(1, int(k));  k += (k % 2 == 0)
    r = k // 2
    fp = np.pad(field, ((0, 0), (r, r)), mode="reflect")
    c = np.pad(fp, ((0, 0), (1, 0)), mode="constant").cumsum(axis=1, dtype=np.float64)
    horiz = (c[:, k:] - c[:, :-k]) / k
    fp2 = np.pad(horiz, ((r, r), (0, 0)), mode="reflect")
    c2 = np.pad(fp2, ((1, 0), (0, 0)), mode="constant").cumsum(axis=0, dtype=np.float64)
    vert = (c2[k:, :] - c2[:-k, :]) / k
    return vert.astype(np.float32)


def _box_blur(img: Image.Image, k: int) -> Image.Image:
    k = max(1, int(k));  k += (k % 2 == 0)
    r = k // 2
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    fp = np.pad(arr, ((0, 0), (r, r), (0, 0)), mode="reflect")
    c = np.pad(fp, ((0, 0), (1, 0), (0, 0)), mode="constant").cumsum(axis=1, dtype=np.float64)
    horiz = (c[:, k:, :] - c[:, :-k, :]) / k
    fp2 = np.pad(horiz, ((r, r), (0, 0), (0, 0)), mode="reflect")
    c2 = np.pad(fp2, ((1, 0), (0, 0), (0, 0)), mode="constant").cumsum(axis=0, dtype=np.float64)
    vert = (c2[k:, :, :] - c2[:-k, :, :]) / k
    return Image.fromarray(np.clip(vert, 0, 255).astype(np.uint8), "RGB")


def _to_gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float32) / 255.0


# --- Palette helpers ---
def _extract_palette_kmeans(img: Image.Image, k: int, rng: np.random.Generator, sample_px: int = 50_000) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32).reshape(-1, 3)
    n = arr.shape[0]
    data = arr[rng.choice(n, size=min(sample_px, n), replace=False)] if n > sample_px else arr
    centers = data[rng.choice(data.shape[0], size=k, replace=False)]
    for _ in range(8):
        d2 = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(2)
        labels = np.argmin(d2, axis=1)
        for ci in range(k):
            mask = labels == ci
            centers[ci] = data[mask].mean(0) if mask.any() else data[rng.integers(0, data.shape[0])]
    return centers


def _fbm(h: int, w: int, octaves: int, persistence: float, rng: np.random.Generator) -> np.ndarray:
    acc = np.zeros((h, w), np.float32)
    amp = 1.0
    for o in range(octaves):
        scale = 2 ** (octaves - o - 1)
        hh, ww = max(1, h // scale), max(1, w // scale)
        base = rng.random((hh, ww), dtype=np.float32)
        up = np.kron(base, np.ones((int(np.ceil(h / hh)), int(np.ceil(w / ww))), dtype=np.float32))[:h, :w]
        up = _box_blur_gray(up, k=3)
        acc += amp * up
        amp *= persistence
    acc -= acc.min()
    acc /= (acc.max() + 1e-8)
    return acc


def _map_field_to_palette(field01: np.ndarray, palette: np.ndarray) -> np.ndarray:
    lum = (0.2126 * palette[:, 0] + 0.7152 * palette[:, 1] + 0.0722 * palette[:, 2])
    pal = palette[np.argsort(lum)]
    h, w = field01.shape
    x = field01.reshape(-1)
    n = pal.shape[0]
    pos = x * (n - 1 - 1e-6)
    idx = np.floor(pos).astype(int)
    frac = (pos - idx)[..., None]
    c0 = pal[idx]
    c1 = pal[np.clip(idx + 1, 0, n - 1)]
    out = (c0 * (1 - frac) + c1 * frac).reshape(h, w, 3)
    return np.clip(out, 0, 255)

def _open_image_any(src: str) -> Image.Image:
    if src.lower().startswith(("http://", "https://")):
        r = requests.get(src, timeout=30); r.raise_for_status()
        im = Image.open(io.BytesIO(r.content)); im.load(); return im.convert("RGB")
    if src.lower().startswith("file://"):
        from urllib.parse import urlparse, unquote
        p = unquote(urlparse(src).path)
        if os.name == "nt" and p.startswith("/"): p = p[1:]
        im = Image.open(p); im.load(); return im.convert("RGB")
    im = Image.open(src); im.load(); return im.convert("RGB")
def _down(img: Image.Image, w: int) -> Image.Image:
    r = w / max(1, img.width)
    h = max(1, int(img.height * r))
    return img.resize((w, h), Image.Resampling.LANCZOS).convert("RGB")

def _photo_elements(img: Image.Image, *, bins: int = 64) -> dict:
    """Very low-freq features from the real photo (no fine structure)."""
    small = _down(img, 256)
    arr = np.asarray(small, np.float32)
    gray = (0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2]) / 255.0

    # --- color ramp by luminance bins (avg RGB per bin) ---
    q = np.clip((gray * (bins-1)).astype(np.int32), 0, bins-1)
    ramp = np.zeros((bins,3), np.float32); counts = np.bincount(q.ravel(), minlength=bins).astype(np.float32)+1e-6
    for c in range(3):
        ramp[:,c] = np.bincount(q.ravel(), weights=arr[...,c].ravel(), minlength=bins).astype(np.float32)/counts

    # --- orientation via gradient histogram (dominant direction) ---
    gy, gx = np.gradient(gray)
    mag = np.hypot(gx, gy) + 1e-8
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi  # [0,pi)
    hist, edges = np.histogram(ang, bins=36, weights=mag)
    dom_idx = int(hist.argmax()); dom_theta = 0.5*(edges[dom_idx]+edges[dom_idx+1])  # radians

    # --- horizon guess: row of max horizontal energy (blurred) ---
    horiz_energy = np.mean(np.abs(gy), axis=1)
    horiz_energy = (horiz_energy - horiz_energy.min()) / (np.ptp(horiz_energy) + 1e-8)
    if horiz_energy.max() > 0.25:
        y_hat = int(np.argmax(_box_blur_gray(horiz_energy[None,:], 9).ravel()))
        horizon = y_hat / (gray.shape[0]-1)
    else:
        horizon = None

    # --- soft composition density (edges + luma dev), then blur hard ---
    edge_mag = (mag / mag.max()).astype(np.float32)
    luma_dev = np.abs(gray - gray.mean()); luma_dev /= (luma_dev.max()+1e-8)
    density = 0.6*edge_mag + 0.4*luma_dev
    density = _box_blur_gray(density, 31)  # heavy blur → only low-freq layout
    density /= (density.sum()+1e-8)

    return {"ramp": ramp, "theta": float(dom_theta), "horizon": horizon, "density": density, "gray": gray}

def _map_with_ramp(field01: np.ndarray, ramp: np.ndarray) -> np.ndarray:
    idx = np.clip((field01 * (len(ramp)-1)).astype(np.int32), 0, len(ramp)-1)
    return ramp[idx]
# =============== Generators ===============
@dataclass
class EdgeArtGenerator(BaseGenerator):
    """Sobel edges + tinted composite."""
    def generate(self, input_image: Image.Image, **kwargs) -> Image.Image:
        thresh = float(kwargs.get("thresh", 0.20))
        tint = str(kwargs.get("tint", "#00ffff"))
        alpha = float(kwargs.get("alpha", 0.80))
        invert = str(kwargs.get("invert", "false")).lower() == "true"
        mix = float(kwargs.get("mix", 0.35))
        bg_mode = str(kwargs.get("bg", "black")).lower()

        src = input_image.convert("RGB")
        gray = _to_gray(src)

        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)

        def conv2d(img: np.ndarray, k: np.ndarray) -> np.ndarray:
            pad = k.shape[0] // 2
            im = np.pad(img, pad, mode="reflect")
            out = np.empty_like(img)
            for y in range(out.shape[0]):
                ys = slice(y, y + 2 * pad + 1)
                for x in range(out.shape[1]):
                    xs = slice(x, x + 2 * pad + 1)
                    out[y, x] = float((im[ys, xs] * k).sum())
            return out

        gx = conv2d(gray, kx); gy = conv2d(gray, ky)
        mag = np.sqrt(gx * gx + gy * gy)
        mag /= (mag.max() + 1e-8)
        edges = (1.0 - (mag > thresh).astype(np.float32)) if invert else (mag > thresh).astype(np.float32)

        src_arr = np.asarray(src, np.float32)
        if bg_mode == "white":
            bg = np.full_like(src_arr, 255.0)
        elif bg_mode == "avg":
            mean = src_arr.reshape(-1, 3).mean(axis=0, keepdims=True)
            bg = np.broadcast_to(mean, src_arr.shape).copy()
        else:
            bg = np.zeros_like(src_arr)

        base = mix * src_arr + (1 - mix) * bg
        r, g, b = _parse_hex_color(tint)
        edge_rgb = np.dstack([edges * r, edges * g, edges * b])
        comp = np.clip(base * (1 - alpha) + edge_rgb * alpha, 0, 255).astype(np.uint8)
        return Image.fromarray(comp, "RGB")


@dataclass
class MosaicGenerator(BaseGenerator):
    """Pixel mosaic (fast, tiny memory)."""
    def generate(self, input_image: Image.Image, **kwargs) -> Image.Image:
        tile = max(2, int(kwargs.get("tile", 16)))
        softness = float(kwargs.get("softness", 0.15))
        img = input_image.convert("RGB")
        if softness > 0:
            k = max(1, int(softness * 6));  k += (k % 2 == 0)
            img = _box_blur(img, k)
        w, h = img.size
        down = (max(1, w // tile), max(1, h // tile))
        return img.resize(down, Image.Resampling.BILINEAR).resize((w, h), Image.Resampling.NEAREST)


@dataclass
class PaletteFBMGenerator(BaseGenerator):
    """FBM noise recolored with palette from the source."""
    def generate(self, input_image: Image.Image, **kwargs) -> Image.Image:
        rng = _rng(self.seed)
        out_w = int(kwargs.get("width", input_image.width))
        out_h = int(kwargs.get("height", input_image.height))
        k_colors = max(2, int(kwargs.get("colors", 5)))
        octaves = max(1, int(kwargs.get("octaves", 6)))
        persistence = float(kwargs.get("persistence", 0.55))
        smooth = float(kwargs.get("smooth", 0.0))

        palette = _extract_palette_kmeans(input_image, k=k_colors, rng=rng, sample_px=60_000).astype(np.float32)
        field = _fbm(out_h, out_w, octaves, persistence, rng)
        if smooth > 0:
            field = _box_blur_gray(field, k=max(1, int(2 * smooth + 1)))
        img_arr = _map_field_to_palette(field, palette)
        return Image.fromarray(img_arr.astype(np.uint8), "RGB")

# ================== PaletteVoronoi (low-RAM, PyCharm-friendly) ==================
@dataclass
class PaletteVoronoiGenerator(BaseGenerator):
    """
    Content-aware Voronoi photomosaic (tiling; f16 distances; GC nudges).
    Memory knobs: work_mp, max_ram_mb, precision('f16'|'f32'), ssaa, min_grid, labels_dtype.
    """
    seed: Optional[int] = None

    def generate(self, input_image: Image.Image, **kwargs) -> Image.Image:
        import gc

        rng = _rng(self.seed)

        # ---------- user params ----------
        out_w = int(kwargs.get("width", input_image.width))
        out_h = int(kwargs.get("height", input_image.height))
        base_sites = max(8, int(kwargs.get("sites", 600)))
        grid_w_req = max(128, int(kwargs.get("grid", 960)))

        guided = str(kwargs.get("guided", "both")).lower()
        guide_edges_w = float(kwargs.get("guide_edges", 0.7))
        guide_luma_w  = float(kwargs.get("guide_luma", 0.3))
        relax = max(0, int(kwargs.get("relax", 2)))
        jitter = float(kwargs.get("jitter", 0.08))
        edge_adherence = float(kwargs.get("edge_adherence", 0.60))
        color_mode = str(kwargs.get("color_mode", "mean")).lower()
        shade = float(kwargs.get("shade", 0.25))
        borders = max(0, int(kwargs.get("borders", 1)))
        border_hex = str(kwargs.get("border_color", "#111111"))
        adaptive_borders = str(kwargs.get("adaptive_borders", "true")).lower() == "true"
        blend = float(kwargs.get("blend", 0.25))
        smooth_src = max(1, int(kwargs.get("smooth_src", 3)))
        edge_keep = float(kwargs.get("edge_keep", 0.35))
        clarity = float(kwargs.get("clarity", 0.25))
        local_contrast = float(kwargs.get("local_contrast", 0.20))
        saturation = float(kwargs.get("saturation", 1.0))
        site_scale = float(kwargs.get("site_scale", 1.0))

        # Photoreal/AA extras
        smoothness    = float(kwargs.get("smoothness", 0.0))
        clarity_boost = float(kwargs.get("clarity_boost", 0.0))
        line_strength = float(kwargs.get("line_strength", 1.0))
        ssaa          = float(kwargs.get("ssaa", 1.5))  # 1..3

        # Anti-lag / memory knobs
        work_mp     = float(kwargs.get("work_mp", 1.2))
        min_grid    = int(kwargs.get("min_grid", 640))
        max_ram_mb  = float(kwargs.get("max_ram_mb", 256.0))
        precision   = str(kwargs.get("precision", "f16")).lower()
        labels_dt   = str(kwargs.get("labels_dtype", "auto")).lower()

        # ---------- working grid (auto-capped by work_mp) ----------
        W0 = grid_w_req
        H0 = max(1, int(round(out_h * (W0 / max(out_w, 1)))))
        ssaa = max(1.0, min(3.0, ssaa))
        W = int(round(W0 * ssaa))
        H = int(round(H0 * ssaa))
        mp = (W * H) / 1e6
        if mp > work_mp:
            s = (work_mp / mp) ** 0.5
            W = max(min_grid, int(W * s))
            H = max(1, int(H * s))
        target_W, target_H = W0, H0

        # ---------- load & pre-smooth ----------
        src_small = input_image.resize((W, H), Image.Resampling.LANCZOS).convert("RGB")
        if smooth_src > 1:
            src_small = self._box_blur_rgb(src_small, k=self._odd(smooth_src))
        src_arr = np.asarray(src_small, dtype=np.float32)
        gray = _to_gray(src_small)

        # ---------- guidance fields ----------
        edges_imp = self._edge_importance_from_gray(gray)
        luma_imp  = self._luma_importance_from_gray(gray)
        if guided == "edges":
            guide = edges_imp
        elif guided == "luma":
            guide = luma_imp
        elif guided == "both":
            g = guide_edges_w * edges_imp + guide_luma_w * luma_imp
            m = g.max()
            guide = (g / m) if m > 1e-8 else np.ones_like(g)
        else:
            guide = np.ones((H, W), dtype=np.float32)

        edge_map = edges_imp.copy()
        if edge_map.max() > 1e-8:
            edge_map /= edge_map.max()
        if local_contrast > 0:
            guide = self._local_contrast_boost(guide, amount=local_contrast)
        guide = _box_blur_gray(guide, k=5)
        guide /= (guide.sum() + 1e-8)

        # ---------- site count ----------
        if site_scale > 0:
            edge_energy = float(edges_imp.mean())
            area_scale = (out_w * out_h) / float(input_image.width * input_image.height)
            n_sites = int(base_sites * (1.0 + 0.6 * edge_energy) * area_scale * site_scale)
            n_sites = max(8, n_sites)
        else:
            n_sites = base_sites

        if labels_dt == "auto":
            lab_dtype = np.int16 if n_sites <= 32767 else np.int32
        else:
            lab_dtype = np.int16 if labels_dt == "i16" else np.int32

        # ---------- sample & relax ----------
        sites_yx = self._sample_sites_from_density(guide, n_sites, rng)
        sites = np.empty((n_sites, 2), dtype=np.float32)
        sites[:, 0] = sites_yx[:, 1] / (W - 1 + 1e-8)
        sites[:, 1] = sites_yx[:, 0] / (H - 1 + 1e-8)
        if jitter > 0:
            sites += (rng.random(sites.shape, dtype=np.float32) - 0.5) * 2 * jitter
            sites = np.clip(sites, 0, 1)

        # row tiling auto-size from max_ram_mb & precision
        bytes_per_val = 2 if precision == "f16" else 4
        row_chunk = max(72, int((max_ram_mb * (1024**2)) / (max(1, W) * bytes_per_val * 3)))
        row_chunk = min(row_chunk, 512)
        row_chunk = max(64, row_chunk)

        for _ in range(relax):
            labels = self._voronoi_labels_streaming(
                H, W, sites, edge_map, edge_adherence,
                row_chunk=row_chunk, want_second=False,
                precision=precision, labels_dtype=lab_dtype, gc_every=3
            )
            cy, cx = self._centroids_from_labels_weighted(labels, guide)
            sites[:, 0] = cx / (W - 1 + 1e-8)
            sites[:, 1] = cy / (H - 1 + 1e-8)

        labels, d0, d1 = self._voronoi_labels_streaming(
            H, W, sites, edge_map, edge_adherence,
            row_chunk=row_chunk, want_second=True,
            precision=precision, labels_dtype=lab_dtype, gc_every=3
        )

        # ---------- color per cell ----------
        if color_mode == "medoid":
            small_rgb = self._cell_color_medoid_fast(src_arr, labels.astype(np.int32), n_sites)
        else:
            small_rgb = self._cell_color_mean(src_arr, labels.astype(np.int32), n_sites)

        # ---------- shading ----------
        if shade > 0:
            gap = np.sqrt(np.maximum(d1.astype(np.float32) - d0.astype(np.float32), 0.0))
            gmax = gap.max()
            if gmax > 1e-8:
                gap /= gmax
            shade_map = (1.0 - gap)
            shade_map = shade_map * (1.0 - shade) + (1.0 - shade_map) * shade
            small_rgb = np.clip(small_rgb * shade_map[..., None], 0, 255)
        seam_feather = float(kwargs.get("seam_feather", 0.0))  # 0..1, 0 = off

        # --- after the "edge-aware mix-back" block, insert:
        if seam_feather > 0:
            # Use label edges as a matte, blur it, and blend back the original
            edge_mask = self._label_edges(labels.astype(np.int32)).astype(np.float32)
            # Feather the matte so the transition is invisible
            k = max(3, int(7 + 10 * seam_feather)) | 1  # wider feather as value grows
            matte = _box_blur_gray(edge_mask, k=k)  # 0..~1
            matte = np.clip(matte * (0.6 + 0.4 * seam_feather), 0.0, 1.0)[..., None]
            # Blend: replace just along seams with a softly blurred original
            src_soft = np.asarray(self._box_blur_rgb(
                Image.fromarray(src_arr.astype(np.uint8), "RGB"), k=5), dtype=np.float32)
            small_rgb = np.clip((1.0 - matte) * small_rgb + matte * src_soft, 0, 255)
        # ---------- edge-aware mix-back ----------
        if edge_keep > 0:
            ek = np.clip(edge_map, 0, 1)[..., None]
            small_rgb = np.clip((1.0 - edge_keep * ek) * small_rgb + (edge_keep * ek) * src_arr, 0, 255)

        # ---------- cleanup ----------
        if smoothness > 0:
            small_rgb = self._guided_smooth_rgb(small_rgb, guide=gray, amount=smoothness)
        if clarity > 0:
            small_rgb = self._unsharp_rgb(small_rgb, strength=clarity, radius=3)
        if clarity_boost > 0:
            small_rgb = self._unsharp_rgb(small_rgb, strength=clarity_boost, radius=2)
        if local_contrast > 0:
            Y, U, V = self._rgb_to_yuv(small_rgb)
            Yb = self._local_contrast_boost(Y, amount=local_contrast)
            small_rgb = self._yuv_to_rgb(Yb, U, V)
        if abs(saturation - 1.0) > 1e-6:
            Y, U, V = self._rgb_to_yuv(small_rgb)
            U *= saturation; V *= saturation
            small_rgb = self._yuv_to_rgb(Y, U, V)

        # ---------- borders ----------
        if borders > 0 and line_strength > 0:
            if adaptive_borders:
                cell_mean = self._cell_color_mean(src_arr, labels.astype(np.int32), n_sites)
                border_rgb = np.clip(cell_mean * 0.35, 0, 255)
            else:
                bc = np.array(_parse_hex_color(border_hex), dtype=np.float32)
                border_rgb = np.broadcast_to(bc, small_rgb.shape)

            edge_mask = self._label_edges(labels.astype(np.int32))
            if borders > 1:
                edge_mask = self._thicken_mask(edge_mask, k=borders)
            if line_strength < 1.0:
                small_rgb[edge_mask] = ((1.0 - line_strength) * small_rgb[edge_mask] +
                                        line_strength * border_rgb[edge_mask])
            else:
                small_rgb[edge_mask] = border_rgb[edge_mask]

        # ---------- optional final blend ----------
        if blend > 0:
            sm = self._box_blur_rgb(Image.fromarray(src_arr.astype(np.uint8), "RGB"), k=self._odd(3))
            sm_arr = np.asarray(sm, dtype=np.float32)
            small_rgb = np.clip((1.0 - blend) * small_rgb + blend * sm_arr, 0, 255)

        # ---------- SSAA downsample ----------
        if (W, H) != (target_W, target_H):
            small_img = Image.fromarray(small_rgb.astype(np.uint8), "RGB")
            small_img = small_img.resize((target_W, target_H), Image.Resampling.LANCZOS)
            small_rgb = np.asarray(small_img, dtype=np.float32)

        out = Image.fromarray(small_rgb.astype(np.uint8), "RGB")
        if (target_W, target_H) != (out_w, out_h):
            out = out.resize((out_w, out_h), Image.Resampling.LANCZOS)

        gc.collect()
        return out

    # ---- streaming Voronoi (tiny peak RAM) ----
    @staticmethod
    def _voronoi_labels_streaming(
        H: int,
        W: int,
        sites01: np.ndarray,
        edge_map: np.ndarray,
        edge_w: float,
        *,
        row_chunk: int = 256,
        want_second: bool = False,
        precision: str = "f16",
        labels_dtype=np.int16,
        gc_every: int = 0,
    ):
        import gc
        use_f16 = (precision == "f16")
        f_dtype = np.float16 if use_f16 else np.float32

        warp_full = (1.0 + edge_w * np.clip(edge_map, 0.0, 1.0)).astype(f_dtype)
        best_idx = np.zeros((H, W), dtype=labels_dtype)
        best_d2  = np.full((H, W), np.inf, dtype=f_dtype)
        second_d2 = np.full((H, W), np.inf, dtype=f_dtype) if want_second else None

        xx = np.linspace(0, 1, W, dtype=f_dtype)[None, :]
        n_sites = int(sites01.shape[0])

        tile_counter = 0
        for y0 in range(0, H, row_chunk):
            y1 = min(H, y0 + row_chunk)
            h  = y1 - y0
            yy = np.linspace(y0/(H-1+1e-8), (y1-1)/(H-1+1e-8), h, dtype=f_dtype)[:, None]
            warp = warp_full[y0:y1, :]

            tile_best_d2  = np.full((h, W), np.inf, dtype=f_dtype)
            tile_best_idx = np.zeros((h, W), dtype=labels_dtype)
            tile_second_d2 = np.full((h, W), np.inf, dtype=f_dtype) if want_second else None

            for i in range(n_sites):
                sx, sy = sites01[i, 0].astype(f_dtype), sites01[i, 1].astype(f_dtype)
                dx = xx - sx
                dy = yy - sy
                if use_f16:
                    d2 = (dx.astype(np.float32) * dx.astype(np.float32) +
                          dy.astype(np.float32) * dy.astype(np.float32)).astype(np.float32)
                    d2 = (d2 * warp.astype(np.float32)).astype(f_dtype)
                else:
                    d2 = (dx * dx + dy * dy) * warp

                better = d2 < tile_best_d2
                if want_second:
                    tile_second_d2 = np.where(better,
                                              np.minimum(tile_second_d2, tile_best_d2),
                                              np.minimum(tile_second_d2, d2))
                tile_best_idx = np.where(better, i, tile_best_idx)
                tile_best_d2  = np.minimum(tile_best_d2, d2)

            best_idx[y0:y1, :] = tile_best_idx
            best_d2 [y0:y1, :] = tile_best_d2
            if want_second:
                second_d2[y0:y1, :] = tile_second_d2

            tile_counter += 1
            if gc_every and (tile_counter % gc_every == 0):
                gc.collect()

        if want_second:
            return best_idx, best_d2, second_d2
        return best_idx

    # ----------------------- helpers -----------------------
    @staticmethod
    def _rgb_to_yuv(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        U = -0.14713 * R - 0.28886 * G + 0.436 * B
        V = 0.615 * R - 0.51499 * G - 0.10001 * B
        return Y, U, V

    @staticmethod
    def _yuv_to_rgb(Y: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
        R = Y + 1.13983 * V
        G = Y - 0.39465 * U - 0.58060 * V
        B = Y + 2.03211 * U
        out = np.stack([R, G, B], axis=-1)
        return np.clip(out, 0, 255)

    @staticmethod
    def _odd(k: int) -> int:
        return k if k % 2 == 1 else k + 1

    @classmethod
    def _box_blur_rgb(cls, img: Image.Image, k: int) -> Image.Image:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
        r = _box_blur_gray(arr[:, :, 0], k)
        g = _box_blur_gray(arr[:, :, 1], k)
        b = _box_blur_gray(arr[:, :, 2], k)
        out = np.stack([r, g, b], axis=-1)
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")

    @classmethod
    def _edge_importance_from_gray(cls, g: np.ndarray) -> np.ndarray:
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
        gx = cls._conv_valid(g, kx); gy = cls._conv_valid(g, ky)
        mag = np.sqrt(gx * gx + gy * gy)
        if mag.max() > 1e-8: mag /= mag.max()
        return _box_blur_gray(mag, k=5)

    @classmethod
    def _luma_importance_from_gray(cls, g: np.ndarray) -> np.ndarray:
        imp = np.abs(g - g.mean())
        if imp.max() > 1e-8: imp /= imp.max()
        return _box_blur_gray(imp, k=7)

    @staticmethod
    def _conv_valid(img: np.ndarray, k: np.ndarray) -> np.ndarray:
        pad = k.shape[0] // 2
        im = np.pad(img, pad, mode="reflect")
        out = np.empty_like(img)
        for y in range(out.shape[0]):
            ys = slice(y, y + 2*pad + 1)
            for x in range(out.shape[1]):
                xs = slice(x, x + 2*pad + 1)
                out[y, x] = float((im[ys, xs] * k).sum())
        return out

    @staticmethod
    def _sample_sites_from_density(density: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        H, W = density.shape
        flat = density.reshape(-1)
        flat /= (flat.sum() + 1e-8)
        idx = rng.choice(flat.size, size=n, replace=False if n <= flat.size else True, p=flat)
        y = (idx // W).astype(np.float32)
        x = (idx %  W).astype(np.float32)
        return np.stack([y, x], axis=1)

    @staticmethod
    def _centroids_from_labels_weighted(labels: np.ndarray, weight: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        H, W = labels.shape
        n = int(labels.max()) + 1
        y = np.repeat(np.arange(H)[:, None], W, axis=1).astype(np.float32)
        x = np.repeat(np.arange(W)[None, :], H, axis=0).astype(np.float32)
        w = weight
        sum_w = np.bincount(labels.ravel().astype(np.int32), weights=w.ravel(), minlength=n).astype(np.float32) + 1e-8
        sum_y = np.bincount(labels.ravel().astype(np.int32), weights=(y*w).ravel(), minlength=n).astype(np.float32)
        sum_x = np.bincount(labels.ravel().astype(np.int32), weights=(x*w).ravel(), minlength=n).astype(np.float32)
        cy = sum_y / sum_w
        cx = sum_x / sum_w
        return cy, cx

    @staticmethod
    def _label_edges(labels: np.ndarray) -> np.ndarray:
        H, W = labels.shape
        m = np.zeros((H, W), dtype=bool)
        m[1:, :] |= (labels[1:, :] != labels[:-1, :])
        m[:-1, :] |= (labels[:-1, :] != labels[1:,  :])
        m[:, 1:] |= (labels[:, 1:] != labels[:, :-1])
        m[:, :-1] |= (labels[:, :-1] != labels[:, 1:])
        return m

    @classmethod
    def _thicken_mask(cls, mask: np.ndarray, k: int = 2) -> np.ndarray:
        f = mask.astype(np.float32)
        b = _box_blur_gray(f, k=max(1, int(k) | 1))
        return b > 0.01

    # ---------- per-cell color ----------
    @staticmethod
    def _cell_color_mean(src: np.ndarray, labels: np.ndarray, n_sites: int) -> np.ndarray:
        lab = labels.ravel().astype(np.int32)
        r = src[..., 0].ravel(); g = src[..., 1].ravel(); b = src[..., 2].ravel()
        count = np.bincount(lab, minlength=n_sites).astype(np.float32) + 1e-8
        sum_r = np.bincount(lab, weights=r, minlength=n_sites).astype(np.float32)
        sum_g = np.bincount(lab, weights=g, minlength=n_sites).astype(np.float32)
        sum_b = np.bincount(lab, weights=b, minlength=n_sites).astype(np.float32)
        means = np.stack([sum_r/count, sum_g/count, sum_b/count], axis=1)
        return np.clip(means[labels.astype(np.int32)], 0, 255)

    @staticmethod
    def _cell_color_medoid_fast(src: np.ndarray, labels: np.ndarray, n_sites: int) -> np.ndarray:
        # Visual proxy: cell mean (pairs well with guided smoothing).
        return PaletteVoronoiGenerator._cell_color_mean(src, labels, n_sites)

    # ---------- sharpening / local contrast ----------
    @classmethod
    def _unsharp_rgb(cls, arr: np.ndarray, strength: float, radius: int = 3) -> np.ndarray:
        Y, U, V = cls._rgb_to_yuv(arr)
        Yb = _box_blur_gray(Y, k=max(1, int(radius) | 1))
        hi = Y - Yb
        denom = (np.max(np.abs(hi)) + 1e-8)
        Y2 = np.clip(Y + strength * 255.0 * hi / denom, 0, 255)
        return cls._yuv_to_rgb(Y2, U, V)

    @staticmethod
    def _local_contrast_boost(field: np.ndarray, amount: float) -> np.ndarray:
        k1 = 3; k2 = 9
        f = field.astype(np.float32)
        mx = f.max(); mn = f.min()
        s = (f - mn) / (mx - mn + 1e-8)
        b1 = _box_blur_gray(s, k=max(1, k1 | 1))
        b2 = _box_blur_gray(s, k=max(1, k2 | 1))
        dog = s + amount * (s - b2) + 0.5 * amount * (s - b1)
        dog = np.clip(dog, 0, 1)
        return dog * (mx - mn) + mn

    # ---------- guided smoothing ----------
    @classmethod
    def _guided_smooth_rgb(cls, arr: np.ndarray, guide: np.ndarray, amount: float) -> np.ndarray:
        if amount <= 0:
            return arr
        r = max(1, int(2 + 6 * amount)) | 1
        eps = (0.02 + 0.18 * amount) ** 2
        I = guide.astype(np.float32)

        mean_I = _box_blur_gray(I, r)
        mean_II = _box_blur_gray(I * I, r)
        var_I = np.maximum(mean_II - mean_I * mean_I, 0.0)

        out = np.empty_like(arr)
        for ch in range(3):
            p = arr[:, :, ch] / 255.0
            mean_p  = _box_blur_gray(p, r)
            mean_Ip = _box_blur_gray(I * p, r)
            cov_Ip = mean_Ip - mean_I * mean_p
            a = cov_Ip / (var_I + eps)
            b = mean_p - a * mean_I
            mean_a = _box_blur_gray(a, r)
            mean_b = _box_blur_gray(b, r)
            q = mean_a * I + mean_b
            out[:, :, ch] = np.clip(q * 255.0, 0, 255)
        alpha = 0.65 * amount
        return np.clip((1.0 - alpha) * arr + alpha * out, 0, 255)

@dataclass
class PaletteNovelGenerator(BaseGenerator):
    """
    Procedural, non-derivative image that borrows *global* cues from the photo
    for realism: color ramp vs luminance, soft composition density, dominant
    orientation, and optional horizon. Geometry is synthesized (no structure
    from the photo is reused).
    """
    seed: Optional[int] = None

    def generate(self, input_image: Image.Image, **kwargs) -> Image.Image:
        rng = _rng(self.seed)

        # ---------- user params ----------
        out_w = int(kwargs.get("width",  input_image.width))
        out_h = int(kwargs.get("height", input_image.height))

        # content/style
        motif      = str(kwargs.get("motif", "cells")).lower()   # cells|waves|stripes|clouds
        sites      = max(128, int(kwargs.get("sites", 1200)))    # for 'cells'
        shade      = float(kwargs.get("shade", 0.20))             # 0..1
        smooth     = float(kwargs.get("smooth", 0.18))            # pre-smoothing of tonal field
        smoothness = float(kwargs.get("smoothness", 0.20))        # guided smoothing post
        clarity    = float(kwargs.get("clarity", 0.16))           # luminance-only unsharp
        saturation = float(kwargs.get("saturation", 1.02))
        coherence  = float(kwargs.get("coherence", 0.90))         # 1.0 = perfectly coherent colors
        relax      = max(0, int(kwargs.get("relax", 2)))          # Lloyd steps
        seed_jit   = float(kwargs.get("seed_jitter", 0.18))
        aa         = float(kwargs.get("aa", 1.35))

        # extra “substance” knobs
        q_levels   = max(0, int(kwargs.get("q_levels", 7)))       # luminance quantization (0=off)
        contrast   = float(kwargs.get("contrast", 0.18))          # local contrast in tonal field
        substance  = float(kwargs.get("substance", 0.55))         # intra-cell form
        edge_glow  = float(kwargs.get("edge_glow", 0.12))         # soft delineation
        vignette   = float(kwargs.get("vignette", 0.08))

        # palette / ref image
        k_colors   = max(3, int(kwargs.get("colors", 9)))
        ref        = kwargs.get("ref", None)
        src_for_palette = _open_image_any(ref) if ref else input_image
        palette = _extract_palette_kmeans(src_for_palette, k=k_colors, rng=rng, sample_px=60_000).astype(np.float32)

        # photo-real cues
        photo_enable   = str(kwargs.get("photo_enable", "true")).lower() == "true"
        photo_bins     = int(kwargs.get("photo_bins", 64))
        photo_density  = float(kwargs.get("photo_density", 0.55))  # 0..1 weight for composition density
        photo_orient   = float(kwargs.get("photo_orient", 0.65))   # 0..1 weight for orientation alignment
        auto_horizon   = str(kwargs.get("auto_horizon", "true")).lower() == "true"
        horizon        = kwargs.get("horizon", None)

        photo = _photo_elements(src_for_palette, bins=photo_bins) if photo_enable else None
        if auto_horizon and horizon is None and photo and photo["horizon"] is not None:
            horizon = float(photo["horizon"])

        # ---------- render size (SSAA) ----------
        W = max(8, int(out_w * max(1.0, min(2.0, aa))))
        H = max(8, int(out_h * max(1.0, min(2.0, aa))))

        # ---------- global procedural fields (no input structure) ----------
        f1 = _fbm(H, W, octaves=6, persistence=0.56, rng=rng)
        f2 = _fbm(H, W, octaves=5, persistence=0.62, rng=rng)

        if horizon is not None:
            hpos = float(horizon)
            yy = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
            sky = 1.0 - np.clip((yy - 0.0) / max(hpos, 1e-6), 0, 1)
            ground = np.clip((yy - hpos) / max(1.0 - hpos, 1e-6), 0, 1)
            base = 0.65 * sky + 0.35 * ground
        else:
            base = np.full((H, W), 0.5, np.float32)

        # tonal backbone
        tonal = np.clip(0.55 * f1 + 0.35 * f2 + 0.10 * base, 0, 1)
        if smooth > 0:
            tonal = _box_blur_gray(tonal, k=max(1, int(3 + 6 * smooth) | 1))
        if contrast > 0:
            tonal = PaletteVoronoiGenerator._local_contrast_boost(tonal, amount=contrast)
            tonal = np.clip(tonal, 0, 1)
        if q_levels > 0:
            q = np.linspace(0, 1, q_levels, dtype=np.float32)
            idx = np.searchsorted(q, tonal, side="right") - 1
            tonal = q[np.clip(idx, 0, q_levels - 1)]

        # choose color mapping source
        def _colorize(field01: np.ndarray) -> np.ndarray:
            if photo_enable and photo is not None:
                return _map_with_ramp(field01, photo["ramp"])
            return _map_field_to_palette(field01, palette)

        # ---------- motifs ----------
        if motif == "cells":
            # Site layout: start from low-discrepancy; optionally bias by photo density
            sites01 = self._halton_sites(sites)
            if photo_enable and photo is not None and photo_density > 0:
                dens_small = photo["density"]
                dens_img = Image.fromarray(
                    (255 * dens_small / (dens_small.max() + 1e-8)).astype(np.uint8), "L"
                ).resize((W, H), Image.Resampling.LANCZOS)
                dens = np.asarray(dens_img, np.float32)
                dens /= (dens.sum() + 1e-8)
                dens = (1.0 - photo_density) * (np.ones_like(dens) / dens.size) + photo_density * dens
                yx = PaletteVoronoiGenerator._sample_sites_from_density(dens, sites, rng)
                sites01 = np.stack([yx[:, 1] / (W - 1 + 1e-8), yx[:, 0] / (H - 1 + 1e-8)], axis=1).astype(np.float32)

            if seed_jit > 0:
                sites01 = np.clip(sites01 + (rng.random(sites01.shape, dtype=np.float32) - 0.5) * seed_jit, 0, 1)

            # Voronoi labels + (optional) Lloyd relaxation for coherence
            edge_map = np.zeros((H, W), np.float32)
            labels, d0, d1 = PaletteVoronoiGenerator._voronoi_labels_streaming(
                H, W, sites01, edge_map, 0.0, row_chunk=256, want_second=True, precision="f16", labels_dtype=np.int32
            )
            for _ in range(relax):
                cy, cx = PaletteVoronoiGenerator._centroids_from_labels_weighted(labels, np.ones_like(tonal))
                sites01[:, 0] = np.clip(cx / (W - 1 + 1e-8), 0, 1)
                sites01[:, 1] = np.clip(cy / (H - 1 + 1e-8), 0, 1)
                labels = PaletteVoronoiGenerator._voronoi_labels_streaming(
                    H, W, sites01, edge_map, 0.0, row_chunk=256, want_second=False, precision="f16", labels_dtype=np.int32
                )

            # Global color coherence via mapped tonal; then per-cell mean
            mapped = _colorize(tonal)
            n_sites = sites01.shape[0]
            cell_rgb = PaletteVoronoiGenerator._cell_color_mean(mapped.astype(np.float32),
                                                                labels.astype(np.int32), n_sites)

            # Tiny alternate to avoid perfectly flat cells
            if coherence < 1.0:
                alt = _colorize(np.clip(tonal * (0.95 + 0.1 * f2), 0, 1))
                cell_alt = PaletteVoronoiGenerator._cell_color_mean(alt.astype(np.float32),
                                                                    labels.astype(np.int32), n_sites)
                rgb = np.clip((coherence) * cell_rgb + (1.0 - coherence) * cell_alt, 0, 255)
            else:
                rgb = cell_rgb

            # Intra-cell form (substance) using distance fields
            if isinstance(d0, np.ndarray) and substance > 0:
                r = np.sqrt(np.maximum(d0.astype(np.float32), 0.0))
                if isinstance(d1, np.ndarray):
                    gap = np.sqrt(np.maximum(d1.astype(np.float32) - d0.astype(np.float32), 0.0))
                    inset = 1.0 - gap / (gap.max() + 1e-8)
                else:
                    inset = r / (r.max() + 1e-8)
                core = 1.0 - (r / (r.max() + 1e-8))
                core_boost = 0.18 * substance
                edge_shade = 0.22 * substance
                rgb = np.clip(
                    rgb * (1.0 - edge_shade * inset[..., None]) + 255.0 * core_boost * (core[..., None] ** 1.2),
                    0, 255
                )
            if edge_glow > 0 and isinstance(d1, np.ndarray):
                gap = np.sqrt(np.maximum(d1.astype(np.float32) - d0.astype(np.float32), 0.0))
                edge = 1.0 - np.clip(gap / (gap.max() + 1e-8), 0, 1)
                k = max(3, int(5 + 10 * edge_glow)) | 1
                glow = _box_blur_gray(edge, k=k)[..., None]
                rgb = np.clip(rgb + 30.0 * edge_glow * glow, 0, 255)

        elif motif == "waves":
            # Align bands with dominant photo orientation
            yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
            xx = np.linspace(0, 1, W, dtype=np.float32)[None, :]
            if photo_enable and photo is not None:
                theta = photo["theta"]
                s = (np.cos(theta) * xx + np.sin(theta) * yy)
                s = (1.0 - photo_orient) * yy + photo_orient * s
            else:
                s = yy
            phase = 2.0 * np.pi * (3.0 * s + 1.25 * f2)
            bands = 0.5 + 0.5 * np.sin(phase)
            lum = np.clip(0.50 * tonal + 0.50 * bands, 0, 1)
            rgb = _colorize(lum)

        elif motif == "stripes":
            yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
            xx = np.linspace(0, 1, W, dtype=np.float32)[None, :]
            if photo_enable and photo is not None:
                theta = photo["theta"]
                s = (np.cos(theta) * xx + np.sin(theta) * yy)
                s = (1.0 - photo_orient) * xx + photo_orient * s
            else:
                s = xx
            phase = 2.0 * np.pi * (5.0 * s + 0.85 * f1)
            bands = 0.5 + 0.5 * np.sin(phase)
            lum = np.clip(0.50 * tonal + 0.50 * bands, 0, 1)
            rgb = _colorize(lum)

        else:  # clouds
            rgb = _colorize(tonal)

        # ---------- post: coherence-preserving cleanup ----------
        if smoothness > 0:
            rgb = PaletteVoronoiGenerator._guided_smooth_rgb(rgb, guide=tonal, amount=smoothness)

        if clarity > 0:
            Y, U, V = PaletteVoronoiGenerator._rgb_to_yuv(rgb)
            Yb = _box_blur_gray(Y, k=3)
            hi = Y - Yb
            denom = (np.max(np.abs(hi)) + 1e-8)
            Y2 = np.clip(Y + clarity * 285.0 * hi / denom, 0, 255)
            rgb = PaletteVoronoiGenerator._yuv_to_rgb(Y2, U, V)

        if abs(saturation - 1.0) > 1e-6:
            Y, U, V = PaletteVoronoiGenerator._rgb_to_yuv(rgb)
            U *= saturation; V *= saturation
            rgb = PaletteVoronoiGenerator._yuv_to_rgb(Y, U, V)

        if vignette > 0:
            yy2 = (np.linspace(-1, 1, H, dtype=np.float32)[:, None]) ** 2
            xx2 = (np.linspace(-1, 1, W, dtype=np.float32)[None, :]) ** 2
            v = np.clip(1.0 - vignette * (xx2 + yy2), 0.85, 1.0)[..., None]
            rgb = np.clip(rgb * v, 0, 255)

        img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")
        if (W, H) != (out_w, out_h):
            img = img.resize((out_w, out_h), Image.Resampling.LANCZOS)
        return img

    @staticmethod
    def _halton_sites(n: int) -> np.ndarray:
        def halton(i: int, b: int) -> float:
            f = 1.0; r = 0.0
            while i > 0:
                f /= b
                r += f * (i % b)
                i //= b
            return r
        pts = np.empty((n, 2), np.float32)
        for i in range(n):
            pts[i, 0] = halton(i + 1, 2)
            pts[i, 1] = halton(i + 1, 3)
        return pts

@dataclass
class PaletteContextGenerator(BaseGenerator):
    """
    Pure contextual generator:
      - Geometry is fully procedural (no reuse of image structure).
      - Borrows only global cues (color ramp vs luminance, dominant orientation,
        soft composition density, and optional horizon).
      - Ignores/overrides 'photo_enable' and 'ref' so output is always contextual-only.
      - Accepts stylistic knobs via **kwargs, but keeps the above guarantees.
    """
    seed: Optional[int] = None

    def generate(self, input_image: Image.Image, **kwargs) -> Image.Image:
        rng = _rng(self.seed)

        # ---------- lock to contextual-only ----------
        # Disallow any attempt to bypass the contextual constraint
        kwargs.pop("photo_enable", None)
        kwargs.pop("ref", None)

        # ---------- user params (style only; safe defaults) ----------
        out_w = int(kwargs.get("width",  input_image.width))
        out_h = int(kwargs.get("height", input_image.height))

        motif      = str(kwargs.get("motif", "cells")).lower()   # cells|waves|stripes|clouds
        sites      = max(128, int(kwargs.get("sites", 1200)))
        shade      = float(kwargs.get("shade", 0.20))
        smooth     = float(kwargs.get("smooth", 0.18))
        smoothness = float(kwargs.get("smoothness", 0.20))
        clarity    = float(kwargs.get("clarity", 0.16))
        saturation = float(kwargs.get("saturation", 1.02))
        coherence  = float(kwargs.get("coherence", 0.90))
        relax      = max(0, int(kwargs.get("relax", 2)))
        seed_jit   = float(kwargs.get("seed_jitter", 0.18))
        aa         = float(kwargs.get("aa", 1.35))

        q_levels   = max(0, int(kwargs.get("q_levels", 7)))
        contrast   = float(kwargs.get("contrast", 0.18))
        substance  = float(kwargs.get("substance", 0.55))
        edge_glow  = float(kwargs.get("edge_glow", 0.12))
        vignette   = float(kwargs.get("vignette", 0.08))

        # weights for contextual cues
        photo_bins     = int(kwargs.get("photo_bins", 64))
        photo_density  = float(kwargs.get("photo_density", 0.55))  # density influence (0..1)
        photo_orient   = float(kwargs.get("photo_orient", 0.65))   # orientation influence (0..1)
        auto_horizon   = str(kwargs.get("auto_horizon", "true")).lower() == "true"
        horizon        = kwargs.get("horizon", None)

        # ---------- extract global cues from the source image ----------
        # Only global descriptors: ramp/orientation/horizon/density
        photo = _photo_elements(input_image, bins=photo_bins)
        if auto_horizon and horizon is None and photo["horizon"] is not None:
            horizon = float(photo["horizon"])

        # ---------- render size (SSAA) ----------
        W = max(8, int(out_w * max(1.0, min(2.0, aa))))
        H = max(8, int(out_h * max(1.0, min(2.0, aa))))

        # ---------- global procedural fields (no input structure) ----------
        f1 = _fbm(H, W, octaves=6, persistence=0.56, rng=rng)
        f2 = _fbm(H, W, octaves=5, persistence=0.62, rng=rng)

        if horizon is not None:
            hpos = float(horizon)
            yy = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
            sky = 1.0 - np.clip((yy - 0.0) / max(hpos, 1e-6), 0, 1)
            ground = np.clip((yy - hpos) / max(1.0 - hpos, 1e-6), 0, 1)
            base = 0.65 * sky + 0.35 * ground
        else:
            base = np.full((H, W), 0.5, np.float32)

        # tonal backbone
        tonal = np.clip(0.55 * f1 + 0.35 * f2 + 0.10 * base, 0, 1)
        if smooth > 0:
            tonal = _box_blur_gray(tonal, k=max(1, int(3 + 6 * smooth) | 1))
        if contrast > 0:
            tonal = PaletteVoronoiGenerator._local_contrast_boost(tonal, amount=contrast)
            tonal = np.clip(tonal, 0, 1)
        if q_levels > 0:
            q = np.linspace(0, 1, q_levels, dtype=np.float32)
            idx = np.searchsorted(q, tonal, side="right") - 1
            tonal = q[np.clip(idx, 0, q_levels - 1)]

        # colorize ONLY from the luminance ramp (global)
        def _colorize(field01: np.ndarray) -> np.ndarray:
            ramp = photo["ramp"]
            return _map_with_ramp(field01, ramp)

        # ---------- motifs ----------
        if motif == "cells":
            # Start with low-discrepancy; bias by soft composition density
            sites01 = self._halton_sites(sites)
            if photo_density > 0:
                dens_small = photo["density"]
                dens_img = Image.fromarray(
                    (255 * dens_small / (dens_small.max() + 1e-8)).astype(np.uint8), "L"
                ).resize((W, H), Image.Resampling.LANCZOS)
                dens = np.asarray(dens_img, np.float32)
                dens /= (dens.sum() + 1e-8)
                dens = (1.0 - photo_density) * (np.ones_like(dens) / dens.size) + photo_density * dens

                # Density sampling for contextual layout
                yx = PaletteVoronoiGenerator._sample_sites_from_density(dens, sites, rng)
                sites01 = np.stack([yx[:, 1] / (W - 1 + 1e-8), yx[:, 0] / (H - 1 + 1e-8)], axis=1).astype(np.float32)

            if seed_jit > 0:
                sites01 = np.clip(sites01 + (rng.random(sites01.shape, dtype=np.float32) - 0.5) * seed_jit, 0, 1)

            # Voronoi labels + (optional) Lloyd relaxation (purely over procedural space)
            edge_map = np.zeros((H, W), np.float32)
            labels, d0, d1 = PaletteVoronoiGenerator._voronoi_labels_streaming(
                H, W, sites01, edge_map, 0.0, row_chunk=256, want_second=True, precision="f16", labels_dtype=np.int32
            )
            for _ in range(relax):
                cy, cx = PaletteVoronoiGenerator._centroids_from_labels_weighted(labels, np.ones_like(tonal))
                sites01[:, 0] = np.clip(cx / (W - 1 + 1e-8), 0, 1)
                sites01[:, 1] = np.clip(cy / (H - 1 + 1e-8), 0, 1)
                labels = PaletteVoronoiGenerator._voronoi_labels_streaming(
                    H, W, sites01, edge_map, 0.0, row_chunk=256, want_second=False, precision="f16", labels_dtype=np.int32
                )

            # Global color coherence via mapped tonal; then per-cell mean
            mapped = _colorize(tonal)
            n_sites = sites01.shape[0]
            cell_rgb = PaletteVoronoiGenerator._cell_color_mean(mapped.astype(np.float32),
                                                                labels.astype(np.int32), n_sites)

            if coherence < 1.0:
                alt = _colorize(np.clip(tonal * (0.95 + 0.1 * f2), 0, 1))
                cell_alt = PaletteVoronoiGenerator._cell_color_mean(alt.astype(np.float32),
                                                                    labels.astype(np.int32), n_sites)
                rgb = np.clip(coherence * cell_rgb + (1.0 - coherence) * cell_alt, 0, 255)
            else:
                rgb = cell_rgb

            # Intra-cell form (substance) using distance fields (procedural)
            if isinstance(d0, np.ndarray) and substance > 0:
                r = np.sqrt(np.maximum(d0.astype(np.float32), 0.0))
                if isinstance(d1, np.ndarray):
                    gap = np.sqrt(np.maximum(d1.astype(np.float32) - d0.astype(np.float32), 0.0))
                    inset = 1.0 - gap / (gap.max() + 1e-8)
                else:
                    inset = r / (r.max() + 1e-8)
                core = 1.0 - (r / (r.max() + 1e-8))
                core_boost = 0.18 * substance
                edge_shade = 0.22 * substance
                rgb = np.clip(
                    rgb * (1.0 - edge_shade * inset[..., None]) + 255.0 * core_boost * (core[..., None] ** 1.2),
                    0, 255
                )
            if edge_glow > 0 and isinstance(d1, np.ndarray):
                gap = np.sqrt(np.maximum(d1.astype(np.float32) - d0.astype(np.float32), 0.0))
                edge = 1.0 - np.clip(gap / (gap.max() + 1e-8), 0, 1)
                k = max(3, int(5 + 10 * edge_glow)) | 1
                glow = _box_blur_gray(edge, k=k)[..., None]
                rgb = np.clip(rgb + 30.0 * edge_glow * glow, 0, 255)

        elif motif == "waves":
            # Align bands with dominant photo orientation (contextual influence only)
            yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
            xx = np.linspace(0, 1, W, dtype=np.float32)[None, :]
            theta = photo["theta"]
            s = (np.cos(theta) * xx + np.sin(theta) * yy)
            s = (1.0 - photo_orient) * yy + photo_orient * s
            phase = 2.0 * np.pi * (3.0 * s + 1.25 * f2)
            bands = 0.5 + 0.5 * np.sin(phase)
            lum = np.clip(0.50 * tonal + 0.50 * bands, 0, 1)
            rgb = _colorize(lum)

        elif motif == "stripes":
            yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
            xx = np.linspace(0, 1, W, dtype=np.float32)[None, :]
            theta = photo["theta"]
            s = (np.cos(theta) * xx + np.sin(theta) * yy)
            s = (1.0 - photo_orient) * xx + photo_orient * s
            phase = 2.0 * np.pi * (5.0 * s + 0.85 * f1)
            bands = 0.5 + 0.5 * np.sin(phase)
            lum = np.clip(0.50 * tonal + 0.50 * bands, 0, 1)
            rgb = _colorize(lum)

        else:  # clouds
            rgb = _colorize(tonal)

        # ---------- post: coherence-preserving cleanup ----------
        if smoothness > 0:
            rgb = PaletteVoronoiGenerator._guided_smooth_rgb(rgb, guide=tonal, amount=smoothness)

        if clarity > 0:
            Y, U, V = PaletteVoronoiGenerator._rgb_to_yuv(rgb)
            Yb = _box_blur_gray(Y, k=3)
            hi = Y - Yb
            denom = (np.max(np.abs(hi)) + 1e-8)
            Y2 = np.clip(Y + clarity * 285.0 * hi / denom, 0, 255)
            rgb = PaletteVoronoiGenerator._yuv_to_rgb(Y2, U, V)

        if abs(saturation - 1.0) > 1e-6:
            Y, U, V = PaletteVoronoiGenerator._rgb_to_yuv(rgb)
            U *= saturation; V *= saturation
            rgb = PaletteVoronoiGenerator._yuv_to_rgb(Y, U, V)

        if shade > 0:
            # gentle global shading from tonal for depth (still procedural)
            sh = np.clip(1.0 - 0.5 * (tonal - tonal.mean()), 0.75, 1.25)[..., None]
            rgb = np.clip(rgb * (1.0 - 0.15 * shade + 0.15 * shade * sh), 0, 255)

        if vignette > 0:
            yy2 = (np.linspace(-1, 1, H, dtype=np.float32)[:, None]) ** 2
            xx2 = (np.linspace(-1, 1, W, dtype=np.float32)[None, :]) ** 2
            v = np.clip(1.0 - vignette * (xx2 + yy2), 0.85, 1.0)[..., None]
            rgb = np.clip(rgb * v, 0, 255)

        img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")
        if (W, H) != (out_w, out_h):
            img = img.resize((out_w, out_h), Image.Resampling.LANCZOS)
        return img

    # Small, local Halton sequence to avoid dependency on other classes for sites.
    @staticmethod
    def _halton_sites(n: int) -> np.ndarray:
        def halton(i: int, b: int) -> float:
            f = 1.0; r = 0.0
            while i > 0:
                f /= b
                r += f * (i % b)
                i //= b
            return r
        pts = np.empty((n, 2), np.float32)
        for i in range(n):
            pts[i, 0] = halton(i + 1, 2)
            pts[i, 1] = halton(i + 1, 3)
        return pts
# ================== PhotoEnhance (no geometry; photo intact) ==================
@dataclass
class PhotoEnhanceGenerator(BaseGenerator):
    """
    High-definition, content-safe photo enhancer.
    Strategy:
      • SSAA upsample → enhance in high-res space → LANCZOS back down (perceived sharpness ↑).
      • Noise-aware, edge-protected multi-band clarity (HF/MF/LF) + mild decon-like boost.
      • Robust AWB, guarded dehaze, local contrast, vibrance w/ hue rolloff, tone shaping.
      • Highlight/shadow protection, skin guard, gamut guard.
    """
    seed: Optional[int] = None

    def generate(self, input_image: Image.Image, **kwargs) -> Image.Image:
        # ---------- Potent defaults (still safe) ----------
        # SSAA (perceived “higher-res”): 1.6–3.0 recommended
        ssaa              = float(kwargs.get("ssaa", 2.0))                 # 1.0..3.0
        ssaa              = max(1.0, min(3.0, ssaa))

        # Base knobs
        denoise           = float(kwargs.get("denoise", 0.12))
        clarity           = float(kwargs.get("clarity", 0.40))              # stronger default
        local_contrast    = float(kwargs.get("local_contrast", 0.30))
        saturation        = float(kwargs.get("saturation", 1.05))
        protect_high      = float(kwargs.get("protect_highlights", 0.14))
        protect_shadow    = float(kwargs.get("protect_shadows", 0.08))

        # Potency & guards
        auto_wb           = str(kwargs.get("auto_wb", "true")).lower() == "true"
        wb_mode           = str(kwargs.get("wb_mode", "bright")).lower()    # bright favors neutral highlights
        wb_strength       = float(kwargs.get("wb_strength", 0.95))

        dehaze_amt        = float(kwargs.get("dehaze", 0.10))
        dehaze_radius     = int(kwargs.get("dehaze_radius", 7))
        dehaze_edge_refine= float(kwargs.get("dehaze_edge_refine", 0.40))

        chroma_denoise    = float(kwargs.get("chroma_denoise", 0.12))
        noise_floor       = float(kwargs.get("noise_floor", 0.14))
        lc_mid_bias       = float(kwargs.get("lc_mid_bias", 0.58))
        microcontrast     = float(kwargs.get("microcontrast", 0.22))

        clarity_halo_guard= float(kwargs.get("clarity_halo_guard", 0.48))
        sharpen_thresh    = float(kwargs.get("sharpen_threshold", 0.10))
        structure_boost   = float(kwargs.get("structure_boost", 0.26))
        texture_boost     = float(kwargs.get("texture_boost", 0.20))

        tone              = str(kwargs.get("tone", "soft")).lower()         # off|soft|filmic|auto
        tone_strength     = float(kwargs.get("tone_strength", 0.62))
        shadow_lift       = float(kwargs.get("shadow_lift", 0.08))
        highlight_rec     = float(kwargs.get("highlight_recovery", 0.12))

        vibrance          = float(kwargs.get("vibrance", 0.12))
        vibrance_hue_roll = float(kwargs.get("vibrance_hue_roll", 0.35))
        protect_skin      = float(kwargs.get("protect_skin", 0.15))

        gamut_guard       = float(kwargs.get("gamut_guard", 0.70))
        chroma_guard      = float(kwargs.get("chroma_guard", 0.40))

        # Decon-like extra bite (small & safe)
        decon_gain        = float(kwargs.get("decon_gain", 0.18))           # 0..0.35
        decon_radius_px   = int(kwargs.get("decon_radius_px", 1))           # kernel ~3x3/5x5 effort

        # --- new resample knobs (final stage) ---
        resample_spec = kwargs.get("resample", None)  # e.g. "2400x1600" or "1.5x"
        resample_factor = float(kwargs.get("resample_factor", 1.0))
        resample_width = kwargs.get("resample_width", None)  # int or None
        resample_height = kwargs.get("resample_height", None)  # int or None
        resample_keep_aspect = str(kwargs.get("resample_keep_aspect", "true")).lower() == "true"
        resample_clarity = float(kwargs.get("resample_clarity", 0.16))  # 0..0.6
        resample_radius = int(kwargs.get("resample_radius", 3))  # unsharp radius
        # ---------- Load & SSAA upsample ----------
        base = input_image.convert("RGB")
        in_w, in_h = base.size
        if ssaa > 1.0:
            up_w = max(8, int(round(in_w * ssaa)))
            up_h = max(8, int(round(in_h * ssaa)))
            img = base.resize((up_w, up_h), Image.Resampling.LANCZOS)
        else:
            img = base

        arr = np.asarray(img, dtype=np.float32)

        def _to_gray01(a: np.ndarray) -> np.ndarray:
            return (0.299*a[...,0] + 0.587*a[...,1] + 0.114*a[...,2]) / 255.0

        gray01 = _to_gray01(arr)

        # ---------- 0) Robust Auto-WB ----------
        if auto_wb and wb_strength > 0:
            eps = 1e-6
            if wb_mode == "bright":
                mask = gray01 > np.percentile(gray01, 65.0)
                means = np.maximum((arr[mask].reshape(-1,3).mean(0) if mask.any() else arr.reshape(-1,3).mean(0)), eps)
            elif wb_mode == "mean":
                means = np.maximum(arr.reshape(-1,3).mean(0), eps)
            else:  # median
                means = np.maximum(np.median(arr.reshape(-1,3), axis=0), eps)
            m = means.mean()
            gains = np.clip(m / means, 0.7, 1.35)
            gains = (1.0 - wb_strength) * np.array([1,1,1], np.float32) + wb_strength * gains
            arr = np.clip(arr * gains[None, None, :], 0, 255)

        # ---------- 0.5) Guarded Dehaze ----------
        if dehaze_amt > 1e-6:
            k = max(3, dehaze_radius | 1)
            min3 = np.minimum.reduce([arr[:,:,0], arr[:,:,1], arr[:,:,2]])
            dark = _box_blur_gray(min3, k=k)
            A = np.array([np.percentile(arr[:,:,c], 99.5) for c in range(3)], dtype=np.float32) + 1e-6
            t0 = np.clip(1.0 - 0.95 * (dark / (np.max(dark) + 1e-6)), 0.12, 1.0)
            if dehaze_edge_refine > 0:
                t_b = _box_blur_gray(t0, k=5)
                edge = PaletteVoronoiGenerator._edge_importance_from_gray(gray01.astype(np.float32))
                edge = edge / (edge.max() + 1e-8)
                t = np.clip((1.0 - dehaze_edge_refine*edge)*t0 + (dehaze_edge_refine*edge)*t_b, 0.12, 1.0)
            else:
                t = t0
            t = (1.0 - dehaze_amt) + dehaze_amt * t
            for c in range(3):
                arr[:,:,c] = np.clip((arr[:,:,c] - A[c]) / t + A[c], 0, 255)

        # ---------- 1) YUV split & chroma denoise ----------
        Y, U, V = PaletteVoronoiGenerator._rgb_to_yuv(arr)
        if chroma_denoise > 0:
            k = max(1, int(3 + 8 * chroma_denoise)) | 1
            U = _box_blur_gray(U, k)
            V = _box_blur_gray(V, k)

        # ---------- 2) Guided luma denoise (noise map) ----------
        if denoise > 0:
            base_dn = PaletteVoronoiGenerator._guided_smooth_rgb(arr, guide=gray01, amount=denoise)
            Y_dn, _, _ = PaletteVoronoiGenerator._rgb_to_yuv(base_dn)
            Yb = _box_blur_gray(Y, k=3)
            noise = np.clip((np.abs(Y - Yb) / 18.0), 0.0, 1.0)
            w_dn = np.clip(noise / (noise + noise_floor), 0.0, 1.0)
            Y = (1.0 - w_dn) * Y + w_dn * Y_dn

        # ---------- 3) Local contrast (two-scale, mid emphasis) ----------
        if local_contrast > 0:
            Y1 = _box_blur_gray(Y, k=3)
            Y2 = _box_blur_gray(Y, k=11)
            dog = Y1 - Y2
            y01 = np.clip(Y / 255.0, 0, 1)
            mids = np.exp(-((y01 - lc_mid_bias) ** 2) / 0.03)
            hi = np.clip((Y - 210) / 45.0, 0, 1)
            sh = np.clip((110 - Y) / 45.0, 0, 1)
            safe = (1.0 - protect_high * hi) * (1.0 - protect_shadow * sh)
            amt = local_contrast * mids * safe
            Y = np.clip(Y + 255.0 * amt * dog / (np.max(np.abs(dog)) + 1e-8), 0, 255)

        # ---------- 3.5) Multi-band clarity + decon-style bite ----------
        if (clarity > 0) or (structure_boost > 0) or (texture_boost > 0) or (microcontrast > 0) or (decon_gain > 0):
            # bands
            Yb3  = _box_blur_gray(Y, k=3)
            Yb7  = _box_blur_gray(Y, k=7)
            Yb13 = _box_blur_gray(Y, k=13)
            hf = Y - Yb3
            mf = Yb3 - Yb7
            lf = Yb7 - Yb13

            # edge/noise guards
            eg = PaletteVoronoiGenerator._edge_importance_from_gray(gray01.astype(np.float32))
            eg = eg / (eg.max() + 1e-8)
            edge_mask = np.clip(1.0 - clarity_halo_guard * eg, 0.0, 1.0)

            def _th(x, t):
                m = np.max(np.abs(x)) + 1e-8
                thr = t * m
                return np.where(np.abs(x) > thr, x, 0.0)

            hf = _th(hf, sharpen_thresh)
            mf = _th(mf, 0.5*sharpen_thresh)
            bf = _th(Y - Yb13, 0.25*sharpen_thresh)  # broader band

            g_hf = texture_boost + 0.50*clarity + 0.70*microcontrast
            g_mf = structure_boost + 0.35*clarity
            g_lf = 0.20*clarity + 0.25*microcontrast
            mix  = g_hf*hf + g_mf*mf + g_lf*bf

            # small decon-like impulse: difference of Gaussians with tighter k
            if decon_gain > 0:
                k_small = max(1, int(2*decon_radius_px) | 1)        # 3..5
                k_large = max(3, int(2*k_small+1) | 1)              # 5..9
                Ys = _box_blur_gray(Y, k=k_small)
                Yl = _box_blur_gray(Y, k=k_large)
                decon = Ys - Yl
                mix += decon_gain * decon

            if np.max(np.abs(mix)) > 1e-8:
                mix = 255.0 * mix / (np.max(np.abs(mix)) + 1e-8)

            y01 = np.clip(Y / 255.0, 0, 1)
            extreme_guard = (0.85 - 0.7*np.abs(y01-0.5))
            mix *= np.clip(extreme_guard, 0.25, 1.0)
            Y = np.clip(Y + edge_mask * mix, 0, 255)

        # ---------- 4) Tone curve & HL/Shadow shaping ----------
        if (tone != "off" and tone_strength > 0) or (shadow_lift > 0) or (highlight_rec > 0):
            y01 = np.clip(Y / 255.0, 0, 1)
            if tone == "soft":
                mapped = y01 / (1.0 + y01)
            elif tone == "filmic":
                A,B,C,D,E,F = 0.22, 0.30, 0.10, 0.20, 0.01, 0.30
                def hable(x): return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F)) - E/F
                mapped = np.clip(hable(2.0*y01), 0, 1)
            elif tone == "auto":
                mid = float(np.median(y01))
                c = 0.6 + 0.8*(mid - 0.5)
                mapped = 1/(1+np.exp(-6*c*(y01-0.5)))
            else:
                mapped = y01
            y_tone = (1.0 - tone_strength) * y01 + tone_strength * mapped
            if shadow_lift > 0:
                y_tone = y_tone + shadow_lift * (1.0 - np.exp(-4.0*y_tone)) * (1.0 - y_tone)
            if highlight_rec > 0:
                y_tone = y_tone - highlight_rec * (1.0 - np.exp(-4.0*(1.0 - y_tone))) * (y_tone)
            Y = np.clip(255.0 * y_tone, 0, 255)

        # ---------- 5) Vibrance/saturation with hue rolloff + chroma guard ----------
        if abs(vibrance) > 1e-6 or abs(saturation - 1.0) > 1e-6:
            hue = np.arctan2(V, U)
            red_mask = np.exp(-0.5*((hue/np.pi*3.0) - 0.0)**2)
            sat_uv = np.sqrt(U**2 + V**2) + 1e-8
            vib_w = np.clip(1.0 - (sat_uv / (np.percentile(sat_uv, 95.0) + 1e-6)), 0.0, 1.0)
            vib_roll = 1.0 - vibrance_hue_roll * red_mask
            vib_scale = 1.0 + vibrance * vib_w * vib_roll
            U *= vib_scale; V *= vib_scale
            y01 = np.clip(Y/255.0, 0, 1.0)
            extreme = np.clip(np.maximum(np.abs(y01-0.0), np.abs(y01-1.0)), 0.0, 1.0)
            guard = 1.0 - chroma_guard * (0.6*extreme + 0.4*(1.0 - vib_w))
            U *= guard; V *= guard
            U *= saturation; V *= saturation

        # ---------- 6) Optional skin protection ----------
        if protect_skin > 0:
            r,g,b = arr[...,0], arr[...,1], arr[...,2]
            maxc = np.maximum.reduce([r,g,b]) + 1e-6
            minc = np.minimum.reduce([r,g,b])
            v = maxc / 255.0
            s = (maxc - minc) / maxc
            skin_hints = (r > g) & (g > b) & (s > 0.15) & (v > 0.20)
            if skin_hints.any():
                Y = np.where(skin_hints, 0.90*Y + 0.10*_box_blur_gray(Y, k=5), Y)

        # ---------- 6.5) Highlight reconstruction ----------
        if highlight_rec > 1e-6:
            clipped = (arr[:,:,0] > 250) | (arr[:,:,1] > 250) | (arr[:,:,2] > 250)
            if clipped.any():
                Y_lin = np.clip(Y, 0, 255) + 1e-6
                scale = (Y_lin / (np.maximum(arr.mean(axis=2), 1e-3)))
                scale = np.clip(scale, 0.85, 1.10)
                for c in range(3):
                    arr[:,:,c] = np.where(clipped, np.clip(arr[:,:,c]*scale, 0, 255), arr[:,:,c])

        # ---------- 7) Recompose & gamut guard ----------
        arr = PaletteVoronoiGenerator._yuv_to_rgb(Y, U, V)
        if gamut_guard > 0:
            over = np.maximum(arr - 255.0, 0.0).max()
            under = np.maximum(0.0 - arr, 0.0).max()
            if over > 1e-6 or under > 1e-6:
                Yc, Uc, Vc = PaletteVoronoiGenerator._rgb_to_yuv(np.clip(arr, 0, 255))
                scale = 1.0 - gamut_guard * np.clip((over+under)/64.0, 0.0, 1.0)
                Uc *= scale; Vc *= scale
                arr = PaletteVoronoiGenerator._yuv_to_rgb(Yc, Uc, Vc)

        out_hi = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

        # Step A: bring back to base size if SSAA was used (keeps perceived sharpness)
        if ssaa > 1.0:
            out = out_hi.resize((in_w, in_h), Image.Resampling.LANCZOS)
        else:
            out = out_hi

        # Step B: optional clarity-aware final RESAMPLE
        # Accepts:
        #  • photo_enhance.resample="WxH"  (e.g., "2400x1600")
        #  • photo_enhance.resample="1.5x" (factor)
        #  • photo_enhance.resample_width / resample_height (with keep_aspect toggle)
        #  • photo_enhance.resample_factor
        want_resample = False
        target_w = target_h = None

        if isinstance(resample_spec, str) and resample_spec:
            s = resample_spec.strip().lower()
            if s.endswith("x") and len(s) > 1:
                # factor, e.g. "1.5x"
                try:
                    resample_factor = float(s[:-1])
                    want_resample = resample_factor != 1.0
                except Exception:
                    pass
            else:
                wh = self._parse_wh(s)
                if wh:
                    target_w, target_h = wh
                    want_resample = True

        # explicit width/height override spec string
        if resample_width is not None or resample_height is not None:
            ow, oh = out.size
            if resample_width is not None and resample_height is not None:
                target_w, target_h = int(resample_width), int(resample_height)
            elif resample_width is not None:
                target_w = int(resample_width)
                if resample_keep_aspect:
                    target_h = max(1, int(round(oh * (target_w / ow))))
                else:
                    target_h = oh
            else:  # height only
                target_h = int(resample_height)
                if resample_keep_aspect:
                    target_w = max(1, int(round(ow * (target_h / oh))))
                else:
                    target_w = ow
            want_resample = True

        # factor fallback
        if not want_resample and resample_factor and abs(resample_factor - 1.0) > 1e-6:
            ow, oh = out.size
            target_w = max(1, int(round(ow * resample_factor)))
            target_h = max(1, int(round(oh * resample_factor)))
            want_resample = True

        if want_resample and target_w and target_h:
            # Lanczos resize
            out = out.resize((int(target_w), int(target_h)), Image.Resampling.LANCZOS)

            # tiny clarity pass to keep things crisp, guarded to avoid halos
            if resample_clarity > 0:
                arr_rs = np.asarray(out, dtype=np.float32)
                arr_rs = self._unsharp_yuv(arr_rs, strength=float(resample_clarity), radius=int(resample_radius))
                out = Image.fromarray(np.clip(arr_rs, 0, 255).astype(np.uint8), "RGB")

        return out

    @staticmethod
    def _parse_wh(s: str) -> Optional[Tuple[int, int]]:
        try:
            w, h = s.lower().split("x", 1)
            return max(1, int(w)), max(1, int(h))
        except Exception:
            return None

    @staticmethod
    def _unsharp_yuv(arr: np.ndarray, strength: float, radius: int = 3) -> np.ndarray:
        # small, halo-guarded unsharp on Y only
        Y, U, V = PaletteVoronoiGenerator._rgb_to_yuv(arr)
        Yb = _box_blur_gray(Y, k=max(1, int(radius) | 1))
        hi = Y - Yb
        denom = (np.max(np.abs(hi)) + 1e-8)
        if denom > 0:
            Y = np.clip(Y + strength * 255.0 * hi / denom, 0, 255)
        return PaletteVoronoiGenerator._yuv_to_rgb(Y, U, V)

@dataclass
class PreviewGenerator(BaseGenerator):
    """
    Pure pass-through: preserves pixels 1:1 and (optionally) draws a tiny HUD.
    Useful as the first stage so you know you're previewing the capture as-is.
    """
    def generate(self, input_image: Image.Image, **kwargs) -> Image.Image:
        # No resampling; just return the frame
        return input_image.convert("RGB")

# ---- Register defaults at import time ----
REGISTRY.register("preview", PreviewGenerator)
REGISTRY.register("palette_context", PaletteContextGenerator)
REGISTRY.register("edge_art", EdgeArtGenerator)
REGISTRY.register("mosaic", MosaicGenerator)
REGISTRY.register("palette_fbm", PaletteFBMGenerator)
REGISTRY.register("palette_voronoi", PaletteVoronoiGenerator)
REGISTRY.register("palette_novel", PaletteNovelGenerator)
REGISTRY.register("photo_enhance", PhotoEnhanceGenerator)
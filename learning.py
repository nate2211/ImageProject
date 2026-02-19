# learning.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
from PIL import Image

# Reuse helpers and registry from your palettes module
from palettes import (
    REGISTRY,
    BaseGenerator,
    _rng,
    _box_blur_gray,
    _map_with_ramp,
    _extract_palette_kmeans,
    _photo_elements,
    PaletteVoronoiGenerator,  # for guided smoothing / yuv helpers
)

# ----------------------------- Profile I/O ----------------------------- #

@dataclass(frozen=True)
class LearnedProfile:
    """Compact bundle of global, non-derivative cues, plus clarity guards."""
    palette: np.ndarray           # (K,3) float32, 0..255
    ramp: np.ndarray              # (bins,3) float32, 0..255; avg RGB per luminance bin
    theta: float                  # dominant orientation in radians [0, pi)
    horizon: Optional[float]      # 0..1 row position or None
    density: np.ndarray           # (H,W) float32 soft composition density, sums to ~1
    bins: int                     # luminance bin count used for ramp
    src_wh: Tuple[int, int]       # (width, height) of image profile was learned from
    wb_gains: Tuple[float, float, float]         # gray-world gains to keep overlay neutral
    edge_soft: np.ndarray                         # (H,W) float32, blurred edge strength (0..1)
    lum_cdf: np.ndarray                           # (256,) float32 luminance CDF (for tone awareness)
    sat_p95: float                                 # 95th-percentile saturation of source (for chroma matching)

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            palette=self.palette.astype(np.float32),
            ramp=self.ramp.astype(np.float32),
            theta=np.float32(self.theta),
            horizon=(np.array([-1.0], np.float32) if self.horizon is None else np.array([self.horizon], np.float32)),
            density=self.density.astype(np.float32),
            bins=np.int32(self.bins),
            src_wh=np.int32(list(self.src_wh)),
            wb_gains=np.array(self.wb_gains, np.float32),
            edge_soft=self.edge_soft.astype(np.float32),
            lum_cdf=self.lum_cdf.astype(np.float32),
            sat_p95=np.float32(self.sat_p95),
        )

    @staticmethod
    def load(path: str) -> "LearnedProfile":
        z = np.load(path, allow_pickle=False)
        horiz = float(z["horizon"][0])
        # Backward-compat guards
        wb_gains = z["wb_gains"].astype(np.float32) if "wb_gains" in z else np.array([1.0, 1.0, 1.0], np.float32)
        edge_soft = z["edge_soft"].astype(np.float32) if "edge_soft" in z else np.zeros_like(z["density"], np.float32)
        lum_cdf = z["lum_cdf"].astype(np.float32) if "lum_cdf" in z else np.linspace(0, 1, 256, dtype=np.float32)
        sat_p95 = float(z["sat_p95"]) if "sat_p95" in z else 0.0
        return LearnedProfile(
            palette=z["palette"].astype(np.float32),
            ramp=z["ramp"].astype(np.float32),
            theta=float(z["theta"]),
            horizon=None if horiz < 0 else float(horiz),
            density=z["density"].astype(np.float32),
            bins=int(z["bins"]),
            src_wh=(int(z["src_wh"][0]), int(z["src_wh"][1])),
            wb_gains=(float(wb_gains[0]), float(wb_gains[1]), float(wb_gains[2])),
            edge_soft=edge_soft.astype(np.float32),
            lum_cdf=lum_cdf.astype(np.float32),
            sat_p95=sat_p95,
        )


# ----------------------------- Generators ----------------------------- #

@dataclass
class LearnProfileGenerator(BaseGenerator):
    """
    Learn global, non-derivative cues from the input and save them to disk.
    Returns the input image unchanged (so it can be piped).
    """

    @staticmethod
    def get_params() -> List[Dict[str, Any]]:
        return [
            {
                "name": "out",
                "type": str,
                "default": "profile.npz",
                "help": "Path to save the learned profile (.npz)."
            },
            {
                "name": "colors",
                "type": int,
                "default": 9,
                "min": 3, "max": 32,
                "help": "Number of dominant colors to extract (k-means)."
            },
            {
                "name": "photo_bins",
                "type": int,
                "default": 64,
                "min": 16, "max": 256,
                "help": "Resolution of the luminance ramp."
            },
            {
                "name": "sample_px",
                "type": int,
                "default": 60000,
                "min": 1000, "max": 500000,
                "help": "Number of pixels to sample for color analysis."
            }
        ]

    def generate(self, input_image: Image.Image, **kwargs) -> Image.Image:
        out_path = (kwargs.get("out") or kwargs.get("learn_profile.out") or "").strip()
        if not out_path:
            raise ValueError("learn_profile: --extra learn_profile.out=PROFILE.npz is required")

        rng = _rng(self.seed)
        k_colors     = max(3, int(kwargs.get("colors", 9)))
        photo_bins   = max(16, int(kwargs.get("photo_bins", 64)))
        sample_px    = max(10_000, int(kwargs.get("sample_px", 60_000)))
        _ = max(128, int(kwargs.get("density_down", 256)))  # density already downsampled in _photo_elements

        # Palette via k-means over sampled pixels
        palette = _extract_palette_kmeans(input_image, k=k_colors, rng=rng, sample_px=sample_px)

        # Global photo cues (ramp/orient/horizon/soft density)
        elems = _photo_elements(input_image, bins=photo_bins)

        # Robust gray-world gains (median-based) for neutral overlay later
        arr = np.asarray(input_image.convert("RGB"), dtype=np.float32)
        med = np.maximum(np.median(arr.reshape(-1, 3), axis=0), 1e-6)
        m = float(np.mean(med))
        wb_gains = tuple(np.clip(m / med, 0.7, 1.3).astype(np.float32).tolist())

        # Soft edge/texture map to protect detail during overlay
        gray = (0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2]) / 255.0
        edge = PaletteVoronoiGenerator._edge_importance_from_gray(gray.astype(np.float32))
        edge = edge / (edge.max() + 1e-8)
        # Slight refinement: mix with local variance (texture) to better protect feathers/hair/foliage
        g_blur = _box_blur_gray(gray.astype(np.float32), k=5)
        local_var = np.clip((gray - g_blur)**2 * 6.0, 0.0, 1.0)
        edge_soft = _box_blur_gray(np.clip(0.7*edge + 0.3*local_var, 0.0, 1.0), k=7)

        # Luminance histogram CDF for tone awareness (0..255 bins)
        Y, U, V = PaletteVoronoiGenerator._rgb_to_yuv(arr)
        Y_clamped = np.clip(Y, 0, 255).astype(np.int32)
        hist = np.bincount(Y_clamped.ravel(), minlength=256).astype(np.float32)
        cdf = np.cumsum(hist) / (hist.sum() + 1e-8)

        # Saturation 95th percentile to guide chroma normalization in overlay
        sat = np.sqrt(U**2 + V**2)
        sat_p95 = float(np.percentile(sat, 95.0))

        prof = LearnedProfile(
            palette=palette.astype(np.float32),
            ramp=elems["ramp"].astype(np.float32),
            theta=float(elems["theta"]),
            horizon=(None if elems["horizon"] is None else float(elems["horizon"])),
            density=elems["density"].astype(np.float32),
            bins=int(photo_bins),
            src_wh=(input_image.width, input_image.height),
            wb_gains=wb_gains,
            edge_soft=edge_soft.astype(np.float32),
            lum_cdf=cdf.astype(np.float32),
            sat_p95=sat_p95,
        )
        prof.save(out_path)
        return input_image  # passthrough


@dataclass
class ProfileContextGenerator(BaseGenerator):
    """
    Context-only renderer that *loads* a learned profile and applies it to the
    current input. Geometry is procedural; only global cues from the profile
    are used (color ramp, orientation, horizon, composition density).
    """

    @staticmethod
    def get_params() -> List[Dict[str, Any]]:
        return [
            # Inputs
            {
                "name": "profile",
                "type": str,
                "default": "profile.npz",
                "help": "Path to the learned profile (.npz)."
            },
            # Pattern
            {
                "name": "motif",
                "type": str,
                "default": "cells",
                "choices": ["cells", "waves", "stripes", "clouds"],
                "help": "Visual pattern style."
            },
            {
                "name": "sites",
                "type": int,
                "default": 1200,
                "min": 100, "max": 5000,
                "help": "Number of Voronoi sites (for 'cells' motif)."
            },
            {
                "name": "relax",
                "type": int,
                "default": 2,
                "min": 0, "max": 10,
                "help": "Lloyd relaxation steps for cell uniformity."
            },
            # Style
            {
                "name": "smoothness",
                "type": float,
                "default": 0.20,
                "min": 0.0, "max": 1.0,
                "help": "Overall pattern smoothing."
            },
            {
                "name": "substance",
                "type": float,
                "default": 0.55,
                "min": 0.0, "max": 1.0,
                "help": "Cell core/edge contrast strength."
            },
            {
                "name": "edge_glow",
                "type": float,
                "default": 0.12,
                "min": 0.0, "max": 1.0,
                "help": "Glowing edges between cells."
            },
            {
                "name": "photo_density",
                "type": float,
                "default": 0.55,
                "min": 0.0, "max": 1.0,
                "help": "How much the photo structure guides cell distribution."
            },
            # Composition
            {
                "name": "compose",
                "type": str,
                "default": "overlay",
                "choices": ["overlay", "replace"],
                "help": "Mix mode: 'overlay' blends with photo, 'replace' shows pattern only."
            },
            {
                "name": "overlay_alpha",
                "type": float,
                "default": 0.35,
                "min": 0.0, "max": 1.0,
                "help": "Opacity of the pattern overlay."
            },
            {
                "name": "edge_protect",
                "type": float,
                "default": 0.70,
                "min": 0.0, "max": 1.0,
                "help": "Reduces overlay on sharp edges to preserve detail."
            },
            {
                "name": "preserve_details",
                "type": bool,
                "default": True,
                "help": "If True, keeps photo luminance (Luma) and only blends Color."
            },
            {
                "name": "chroma_match",
                "type": bool,
                "default": True,
                "help": "Match pattern saturation to the original photo."
            }
        ]

    def generate(self, input_image: Image.Image, **kwargs) -> Image.Image:
        rng = _rng(self.seed)

        # ---------- required profile ----------
        prof_path = str(kwargs.get("profile", "")).strip()
        if not prof_path:
            raise ValueError("profile_context: --extra profile_context.profile=PROFILE.npz is required")
        prof = LearnedProfile.load(prof_path)

        # ---------- user params ----------
        out_w = int(kwargs.get("width",  input_image.width))
        out_h = int(kwargs.get("height", input_image.height))

        motif      = str(kwargs.get("motif", "cells")).lower()
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

        photo_density = float(kwargs.get("photo_density", 0.55))
        photo_orient  = float(kwargs.get("photo_orient", 0.65))

        # clarity-preserving compose knobs
        compose = str(kwargs.get("compose", "overlay")).lower()         # overlay | replace
        overlay_alpha = float(kwargs.get("overlay_alpha", 0.35))
        edge_protect  = float(kwargs.get("edge_protect", 0.70))
        preserve_details = str(kwargs.get("preserve_details", "true")).lower() == "true"
        preserve_blacks = float(kwargs.get("preserve_blacks", 0.40))
        protect_highlights = float(kwargs.get("protect_highlights", 0.20))
        chroma_match = str(kwargs.get("chroma_match", "true")).lower() == "true"
        chroma_clip  = float(kwargs.get("chroma_clip", 1.25))

        # ---------- render size (SSAA) ----------
        W = max(8, int(out_w * max(1.0, min(2.0, aa))))
        H = max(8, int(out_h * max(1.0, min(2.0, aa))))

        # ---------- procedural tonal fields ----------
        f1 = _fbm_like(H, W, octaves=6, persistence=0.56, rng=rng)
        f2 = _fbm_like(H, W, octaves=5, persistence=0.62, rng=rng)

        if prof.horizon is not None:
            hpos = float(prof.horizon)
            yy = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
            sky = 1.0 - np.clip((yy - 0.0) / max(hpos, 1e-6), 0, 1)
            ground = np.clip((yy - hpos) / max(1.0 - hpos, 1e-6), 0, 1)
            base = 0.65 * sky + 0.35 * ground
        else:
            base = np.full((H, W), 0.5, np.float32)

        tonal = np.clip(0.55 * f1 + 0.35 * f2 + 0.10 * base, 0, 1)
        if smooth > 0:
            k = max(1, int(3 + 6 * smooth) | 1)
            tonal = _box_blur_gray(tonal, k=k)
        if contrast > 0:
            tonal = PaletteVoronoiGenerator._local_contrast_boost(tonal, amount=contrast)
            tonal = np.clip(tonal, 0, 1)
        if q_levels > 0:
            q = np.linspace(0, 1, q_levels, dtype=np.float32)
            idx = np.searchsorted(q, tonal, side="right") - 1
            tonal = q[np.clip(idx, 0, q_levels - 1)]

        # colorize via learned luminance ramp (not per-pixel reuse)
        def _colorize(field01: np.ndarray) -> np.ndarray:
            return _map_with_ramp(field01, prof.ramp)

        # ---------- motif rendering ----------
        if motif == "cells":
            # Start from low-discrepancy; bias by learned composition density
            sites01 = _halton_sites_local(sites)
            if photo_density > 0:
                dens_small = prof.density
                dens_img = Image.fromarray(
                    (255 * dens_small / (dens_small.max() + 1e-8)).astype(np.uint8), "L"
                ).resize((W, H), Image.Resampling.LANCZOS)
                dens = np.asarray(dens_img, np.float32)
                dens /= (dens.sum() + 1e-8)
                dens = (1.0 - photo_density) * (np.ones_like(dens) / dens.size) + photo_density * dens
                yx = PaletteVoronoiGenerator._sample_sites_from_density(dens, sites, _rng(self.seed))
                sites01 = np.stack([yx[:, 1] / (W - 1 + 1e-8), yx[:, 0] / (H - 1 + 1e-8)], axis=1).astype(np.float32)

            if seed_jit > 0:
                sites01 = np.clip(sites01 + (_rng(self.seed).random(sites01.shape, dtype=np.float32) - 0.5) * seed_jit, 0, 1)

            # Voronoi labels (+ optional Lloyd)
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

            # Intra-cell form
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

        elif motif in ("waves", "stripes"):
            yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
            xx = np.linspace(0, 1, W, dtype=np.float32)[None, :]
            theta = float(prof.theta)
            if motif == "waves":
                s = (np.cos(theta) * xx + np.sin(theta) * yy)
                s = (1.0 - photo_orient) * yy + photo_orient * s
                phase = 2.0 * np.pi * (3.0 * s + 1.25 * f2)
            else:
                s = (np.cos(theta) * xx + np.sin(theta) * yy)
                s = (1.0 - photo_orient) * xx + photo_orient * s
                phase = 2.0 * np.pi * (5.0 * s + 0.85 * f1)
            bands = 0.5 + 0.5 * np.sin(phase)
            lum = np.clip(0.50 * tonal + 0.50 * bands, 0, 1)
            rgb = _colorize(lum)

        else:  # clouds
            rgb = _colorize(tonal)

        # ---------- post cleanup (context layer) ----------
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
            sh = np.clip(1.0 - 0.5 * (tonal - tonal.mean()), 0.75, 1.25)[..., None]
            rgb = np.clip(rgb * (1.0 - 0.15 * shade + 0.15 * shade * sh), 0, 255)
        if vignette > 0:
            yy2 = (np.linspace(-1, 1, H, dtype=np.float32)[:, None]) ** 2
            xx2 = (np.linspace(-1, 1, W, dtype=np.float32)[None, :]) ** 2
            v = np.clip(1.0 - vignette * (xx2 + yy2), 0.85, 1.0)[..., None]
            rgb = np.clip(rgb * v, 0, 255)

        # ---------- size match ----------
        ctx_img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")
        if (W, H) != (out_w, out_h):
            ctx_img = ctx_img.resize((out_w, out_h), Image.Resampling.LANCZOS)

        if compose == "replace":
            # Pure contextual output
            return ctx_img

        # ---------- overlay mode: CONTENT-PRESERVING APPLY ----------
        base = input_image.convert("RGB").resize((out_w, out_h), Image.Resampling.LANCZOS)
        base_arr = np.asarray(base, np.float32)
        ctx_arr  = np.asarray(ctx_img,  np.float32)

        # Neutralize context via learned gray-world gains to avoid color casts
        gw = np.array(prof.wb_gains, dtype=np.float32)[None, None, :]
        ctx_arr = np.clip(ctx_arr * gw, 0, 255)

        # Edge protection weight from learned profile (resized)
        es = prof.edge_soft
        es_img = Image.fromarray((255.0 * np.clip(es, 0, 1)).astype(np.uint8), "L").resize(
            (out_w, out_h), Image.Resampling.LANCZOS
        )
        es_arr = np.asarray(es_img, np.float32) / 255.0  # shape (H,W), 0..1

        # Final overlay weight: lower where edges are strong (keep as 2-D)
        w = overlay_alpha * (1.0 - edge_protect * es_arr)  # (H,W)
        w = np.clip(w, 0.0, 1.0)

        # Extra shadow/highlight protection masks from base luminance (2-D)
        Yb, Ub, Vb = PaletteVoronoiGenerator._rgb_to_yuv(base_arr)
        sh_mask = np.clip((110.0 - Yb) / 55.0, 0.0, 1.0)     # strong in deep shadows
        hi_mask = np.clip((Yb - 210.0) / 55.0, 0.0, 1.0)     # strong in bright highlights
        w = w * (1.0 - preserve_blacks * sh_mask) * (1.0 - protect_highlights * hi_mask)  # still (H,W)

        if preserve_details:
            # Blend only color (U/V); keep luminance from the photo to avoid any “muddiness”
            Yc, Uc, Vc = PaletteVoronoiGenerator._rgb_to_yuv(ctx_arr)

            # Optional chroma normalization: match context saturation to the photo’s
            if chroma_match:
                # Compute current context saturation p95 and base saturation p95
                sat_ctx = np.sqrt(Uc**2 + Vc**2)
                sat_ctx_p95 = float(np.percentile(sat_ctx, 95.0))
                # Prefer live base stat; fall back to learned profile if needed
                sat_base = np.sqrt(Ub**2 + Vb**2)
                sat_base_p95 = float(np.percentile(sat_base, 95.0)) if sat_base.size else max(1.0, prof.sat_p95)
                scale = sat_base_p95 / (sat_ctx_p95 + 1e-8)
                scale = float(np.clip(scale, 1.0 / max(chroma_clip, 1e-6), chroma_clip))
                Uc *= scale
                Vc *= scale

            # broadcast-safe 2-D blends (w is (H,W))
            Uo = (1.0 - w) * Ub + w * Uc
            Vo = (1.0 - w) * Vb + w * Vc
            out = PaletteVoronoiGenerator._yuv_to_rgb(Yb, Uo, Vo)
        else:
            # Full RGB overlay (softer details). Expand w to (H,W,3) only here.
            w3 = w[..., None]
            out = (1.0 - w3) * base_arr + w3 * ctx_arr

        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


# ----------------------------- Small helpers ----------------------------- #

def _fbm_like(h: int, w: int, *, octaves: int, persistence: float, rng: np.random.Generator) -> np.ndarray:
    # identical to palettes._fbm but kept local to avoid import cycles
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


def _halton_sites_local(n: int) -> np.ndarray:
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


# ----------------------------- Registry hooks ----------------------------- #

REGISTRY.register("learn_profile", LearnProfileGenerator)
REGISTRY.register("profile_context", ProfileContextGenerator)
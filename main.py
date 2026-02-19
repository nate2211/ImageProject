from __future__ import annotations

import argparse
import hashlib
import io
import logging
import mimetypes
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, unquote

import numpy as np
import requests
import subprocess
import shutil
import threading
from collections import deque

from PIL import Image, ImageOps

# Import registry & generators (registration happens at import time)
# Make sure the file is named palettes.py (not pallettes.py)
from palettes import REGISTRY  # noqa: F401
import content
import learning
import filters

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass

from time import perf_counter, sleep

try:
    import mss  # fast, cross-platform screen capture
except Exception:
    mss = None

# Optional live preview / recording (recommended):
try:
    import cv2
except Exception:
    cv2 = None

# Windows window-rect lookup (optional; only used when --window / listing is set)
try:
    import win32gui
    import win32con
    import ctypes
except Exception:
    win32gui = None
    win32con = None
    ctypes = None
from time import perf_counter
# =============== Logging ===============
log = logging.getLogger("imagegen")


def setup_logging(verbosity: int = 0) -> None:
    level = [logging.WARNING, logging.INFO, logging.DEBUG][min(verbosity, 2)]
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

class FFMpegWriter:
    def __init__(self, path: str, w: int, h: int, fps: float, crf: int = 18, preset: str = "veryfast", ffmpeg_bin: str = "ffmpeg"):
        ffmpeg_path = _resolve_ffmpeg(ffmpeg_bin)
        self.proc = subprocess.Popen(
            [ffmpeg_path, "-y",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
             "-an",
             "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
             "-movflags", "+faststart",
             path],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
        self.w, self.h = w, h

    def write_bgr(self, frame_bgr: "np.ndarray") -> None:
        self.proc.stdin.write(frame_bgr.tobytes())

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        self.proc.wait(timeout=5)

class _PipeDrainer:
    def __init__(self, stream):
        self._stream = stream
        self._buf = bytearray()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def start(self): self._t.start()

    def _run(self):
        try:
            while not self._stop.is_set():
                chunk = self._stream.read(8192)
                if not chunk:
                    break
                self._buf[:] = (self._buf + chunk)[-65536:]
        except Exception:
            pass

    def stop(self):
        self._stop.set()
        try: self._t.join(timeout=0.5)
        except Exception: pass

    def last_text(self):
        try: return self._buf.decode(errors="ignore")
        except Exception: return ""
# =============== Core: Fetcher & Loader ===============
class FileFetcher:
    """Fetch bytes from http(s) / file:// / local path with a tiny, safe cache."""

    def __init__(self, cache_dir: Optional[Path] = None, timeout: float = 20.0) -> None:
        self.timeout = timeout
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "imagegen_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "imagegen/1.0 (+https://local)"})

    def fetch(self, src: str) -> Tuple[bytes, Optional[str]]:
        parsed = urlparse(src)
        scheme = (parsed.scheme or "").lower()
        if scheme in ("http", "https"):
            return self._fetch_http_cached(src)
        if scheme == "file":
            local_path = unquote(parsed.path)
            if os.name == "nt" and local_path.startswith("/"):
                local_path = local_path[1:]
            return self._fetch_local(local_path)
        if scheme == "":
            return self._fetch_local(src)
        raise ValueError(f"Unsupported URL scheme: {scheme}")

    def _cache_key(self, url: str) -> Path:
        h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
        return self.cache_dir / f"{h}.bin"

    def _fetch_http_cached(self, url: str) -> Tuple[bytes, Optional[str]]:
        key = self._cache_key(url)
        if key.exists():
            try:
                raw = key.read_bytes()
                ctype = mimetypes.guess_type(url)[0]
                log.info("Cache hit: %s", key.name)
                return raw, ctype
            except Exception:
                pass
        log.info("Fetching: %s", url)
        r = self._session.get(url, timeout=self.timeout, stream=True)
        r.raise_for_status()
        raw = r.content
        try:
            key.write_bytes(raw)
        except Exception:
            pass
        return raw, r.headers.get("Content-Type")

    def _fetch_local(self, path_str: str) -> Tuple[bytes, Optional[str]]:
        p = Path(path_str)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"Input file not found: {p}")
        raw = p.read_bytes()
        return raw, mimetypes.guess_type(p.name)[0]


class ImageLoader:
    """Decode bytes → RGB Pillow image. Optional max-size for speed/RAM."""

    def load(self, raw: bytes, content_type: Optional[str], *, max_size: Optional[int] = None) -> Image.Image:
        try:
            img = Image.open(io.BytesIO(raw))
            img.load()
        except Exception as e:
            raise ValueError(f"Failed to decode image: {e}") from e

        if img.mode == "L":
            img = ImageOps.colorize(img, "black", "white").convert("RGB")
        elif img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        else:
            img = img.convert("RGB")

        if max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return img



def _is_video_path_or_url(src: str, content_type: Optional[str]) -> bool:
    if content_type and content_type.lower().startswith("video/"):
        return True
    p = urlparse(src)
    path = (p.path or src).lower()
    return any(path.endswith(ext) for ext in (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"))

def _is_video_output_path(p: Path) -> bool:
    return p.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")

def _resolve_ffprobe(ffmpeg_bin: str) -> str:
    ffmpeg_path = _resolve_ffmpeg(ffmpeg_bin)
    # If ffmpeg is ".../ffmpeg(.exe)" assume ffprobe sits next to it; else fallback to PATH.
    ffprobe = str(Path(ffmpeg_path).with_name("ffprobe.exe" if os.name == "nt" else "ffprobe"))
    if Path(ffprobe).exists():
        return ffprobe
    hit = shutil.which("ffprobe")
    if hit:
        return hit
    raise FileNotFoundError("ffprobe not found (needed for video input). Install FFmpeg full build (ffmpeg+ffprobe).")

def _ffprobe_size_and_fps(src: str, ffmpeg_bin: str = "ffmpeg") -> Tuple[int, int, float]:
    ffprobe = _resolve_ffprobe(ffmpeg_bin)
    # width,height,r_frame_rate
    cmd = [
        ffprobe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "default=nw=1:nk=1",
        src
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8", errors="ignore").strip().splitlines()
    if len(out) < 3:
        raise RuntimeError(f"ffprobe failed to read stream info for: {src}")
    w = int(out[0].strip())
    h = int(out[1].strip())
    fr = out[2].strip()
    # r_frame_rate like "30000/1001"
    if "/" in fr:
        a, b = fr.split("/", 1)
        fps = float(a) / float(b) if float(b) != 0 else 30.0
    else:
        fps = float(fr) if fr else 30.0
    return w, h, fps

def _read_exact(stream, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            break
        buf += chunk
    return bytes(buf)

def _iter_video_frames_raw_bgr24(
    src: str,
    *,
    ffmpeg_bin: str = "ffmpeg",
    max_size: Optional[int] = None,
    target_fps: Optional[float] = None,
):
    ffmpeg_path = _resolve_ffmpeg(ffmpeg_bin)
    in_w, in_h, in_fps = _ffprobe_size_and_fps(src, ffmpeg_bin=ffmpeg_bin)

    # Decide output size up front (and FORCE ffmpeg to match it)
    out_w, out_h = in_w, in_h
    if max_size and max_size > 0 and max(in_w, in_h) > max_size:
        if in_w >= in_h:
            out_w = int(max_size)
            out_h = int(round(in_h * (out_w / in_w)))
        else:
            out_h = int(max_size)
            out_w = int(round(in_w * (out_h / in_h)))

        # make even (important for many encoders + consistency)
        out_w = max(2, (out_w // 2) * 2)
        out_h = max(2, (out_h // 2) * 2)

    out_fps = float(target_fps) if (target_fps and target_fps > 0) else float(in_fps or 30.0)

    vf_parts = []
    if target_fps and target_fps > 0:
        vf_parts.append(f"fps={float(target_fps)}")
    # force exact scale if requested (or if you want, always force scale=out_w:out_h)
    if (out_w, out_h) != (in_w, in_h):
        vf_parts.append(f"scale={out_w}:{out_h}")

    vf = ",".join(vf_parts) if vf_parts else None

    cmd = [ffmpeg_path, "-hide_banner", "-nostats", "-loglevel", "error", "-i", src]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-pix_fmt", "bgr24", "-f", "rawvideo", "-an", "-sn", "-dn", "-"]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
        bufsize=0,
    )
    if proc.stdout is None or proc.stderr is None:
        raise RuntimeError("Failed to start ffmpeg video decode (no pipes).")

    drainer = _PipeDrainer(proc.stderr)
    drainer.start()

    frame_bytes = out_w * out_h * 3

    def gen():
        try:
            while True:
                buf = _read_exact(proc.stdout, frame_bytes)
                if not buf:
                    break
                if len(buf) < frame_bytes:
                    err = drainer.last_text()
                    rc = proc.poll()
                    raise RuntimeError(
                        f"Short read from ffmpeg (got {len(buf)} of {frame_bytes}). rc={rc}\n"
                        f"ffmpeg stderr:\n{err}"
                    )
                frame = np.frombuffer(buf, np.uint8).reshape(out_h, out_w, 3)  # BGR
                yield Image.fromarray(frame[:, :, ::-1])
        finally:
            try:
                proc.kill()
            except Exception:
                pass
            drainer.stop()

    return out_w, out_h, out_fps, gen()
def _run_video_pipeline(
    *,
    src: str,
    out_path: Path,
    stages: List[str],
    stage_extras: List[Dict[str, Any]],
    seed: Optional[int],
    max_size: Optional[int],
    scale: int,
    ffmpeg_bin: str = "ffmpeg",
    video_fps: Optional[float] = None,
    crf: int = 18,
    preset: str = "veryfast",
) -> None:
    gens = [REGISTRY.create(name, seed=seed) for name in stages]

    w, h, fps, frames = _iter_video_frames_raw_bgr24(
        src, ffmpeg_bin=ffmpeg_bin, max_size=max_size, target_fps=video_fps
    )

    # We may upscale AFTER pipeline (your existing semantics).
    # Writer expects final output size.
    writer = None
    riter = None
    t0 = perf_counter()
    last_log = t0
    n = 0

    try:
        for img in frames:
            out = _run_pipeline_instances(img, gens, stage_extras)

            if scale and scale > 1:
                out = out.resize((out.width * scale, out.height * scale), Image.Resampling.LANCZOS)

            if writer is None:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                writer = FFMpegWriter(
                    str(out_path), out.width, out.height, fps,
                    crf=crf, preset=preset, ffmpeg_bin=ffmpeg_bin
                )
                print(f"Video: {out.width}x{out.height} @ {fps:.3f} fps → {out_path}")

            writer.write_bgr(_pil_to_bgr(out))

            n += 1
            now = perf_counter()
            if now - last_log >= 1.0:  # log once per second
                elapsed = now - t0
                eff_fps = n / elapsed if elapsed > 0 else 0.0
                print(f"Processed {n} frames | {eff_fps:.2f} fps (effective)")
                last_log = now

        if writer is None:
            raise RuntimeError("No frames decoded from input video.")
    finally:
        if writer is not None:
            writer.close()
            print(f"Done. Wrote {n} frames to {out_path}")
# =============== Small CLI helpers ===============
def _coerce(v: str) -> Any:
    if v.isdigit():
        return int(v)
    try:
        return float(v)
    except ValueError:
        low = v.lower()
        if low in ("true", "false"):
            return low == "true"
    return v


def _parse_kv_pairs(pairs: Optional[List[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not pairs:
        return out
    for p in pairs:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k.strip()] = _coerce(v.strip())
    return out


def _infer_format_from_path(p: Path) -> str:
    ext = p.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return "JPEG"
    if ext == ".png":
        return "PNG"
    if ext == ".webp":
        return "WEBP"
    return "PNG"


# ======= Pipeline helpers (multi-generator) =======
def _parse_pipeline(spec: Optional[str], fallback_generator: Optional[str]) -> List[str]:
    if spec:
        stages = [s.strip().lower() for s in spec.split("|") if s.strip()]
        if not stages:
            raise ValueError("Empty --pipeline. Example: edge_art|palette_voronoi")
        return stages
    if fallback_generator:
        return [fallback_generator.strip().lower()]
    raise ValueError("Provide --pipeline 'g1|g2|...' or --generator NAME.")


def _split_stage_extras(stages: List[str], raw_extras: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extras can be:
      - Unprefixed:        key=val          (applies to ALL stages unless overridden)
      - By name:           gen.key=val      (applies to the stage whose name matches 'gen')
      - By index (0-based) 0.key=val        (applies to stage at index 0)
      - 'all.key=val'      applies to all (alias of unprefixed)
    Merge order per stage: (unprefixed/all) -> (by-name) -> (by-index)
    """
    stage_extras = [dict() for _ in stages]
    global_extras: Dict[str, Any] = {}
    name_targets: Dict[str, Dict[str, Any]] = {}
    index_targets: Dict[int, Dict[str, Any]] = {}

    for k, v in raw_extras.items():
        if "." not in k:
            global_extras[k] = v
            continue
        prefix, key = k.split(".", 1)
        prefix = prefix.strip().lower()
        key = key.strip()
        if prefix == "all":
            global_extras[key] = v
        elif prefix.isdigit():
            idx = int(prefix)
            if 0 <= idx < len(stages):
                index_targets.setdefault(idx, {})[key] = v
        else:
            name_targets.setdefault(prefix, {})[key] = v

    for i, name in enumerate(stages):
        merged = {}
        merged.update(global_extras)
        if name in name_targets:
            merged.update(name_targets[name])
        if i in index_targets:
            merged.update(index_targets[i])
        stage_extras[i] = merged
    return stage_extras


def _run_pipeline(
    img: Image.Image,
    stages: List[str],
    stage_extras: List[Dict[str, Any]],
    seed: Optional[int],
) -> Image.Image:
    out = img
    for i, name in enumerate(stages):
        gen = REGISTRY.create(name, seed=seed)
        extras = stage_extras[i]
        log.info("Stage %d/%d: %s extras=%s", i + 1, len(stages), name, {k: extras[k] for k in sorted(extras)})
        out = gen.generate(out, **extras)
    return out


# =============== CLI ===============
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="URL → Image generator with pipeline support (low-RAM)")
    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv).")

    sub = p.add_subparsers(dest="cmd", required=True)

    lp = sub.add_parser("list", help="List generators.")
    lp.set_defaults(func=cmd_list)

    sp = sub.add_parser("stream", help="Live-capture the screen/window and process frames through a generator pipeline.")
    sp.add_argument("--monitor", type=int, default=1, help="Monitor index (1-based per mss).")
    sp.add_argument("--region", type=str, default=None, help="Crop region as x,y,w,h (overrides monitor capture).")
    sp.add_argument("--fps", type=float, default=20.0, help="Target frames per second.")
    sp.add_argument("--dur", type=float, default=0.0, help="Duration in seconds (0 = until Ctrl+C).")
    sp.add_argument("--pipeline", required=True, help="Pipeline as 'g1|g2|g3'.")
    sp.add_argument("--generator", choices=REGISTRY.names(), help="(Optional) Single generator (deprecated).")
    sp.add_argument("--max-size", type=int, default=None, help="Downscale input longest side before processing.")
    sp.add_argument("--scale", type=int, default=1, help="Final upscale factor via Lanczos (1=off).")
    sp.add_argument("--seed", type=int, default=None, help="RNG seed.")
    sp.add_argument("--extra", nargs="*", help="Extra k=v pairs (same format as run).")
    sp.add_argument("--preview", action="store_true", help="Show live preview window (requires OpenCV).")
    sp.add_argument("--out-video", type=Path, help="Optional path to save output video (mp4/avi).")
    sp.add_argument("--video-fps", type=float, default=None, help="Video save FPS (default = capture FPS).")
    sp.add_argument("--window", type=str, default=None,
                    help="Capture a specific application window by title substring or exact caption (Windows).")
    sp.add_argument("--follow-window", action="store_true",
                    help="Recompute the window rectangle each frame (track moves/resizes).")
    sp.add_argument("--window-trim", type=int, default=8,
                    help="Trim (px) to remove window borders/titlebar (applied on all sides, Windows only).")
    sp.add_argument("--preview-x", type=int, default=None, help="Force preview window X position (screen coords).")
    sp.add_argument("--preview-y", type=int, default=None, help="Force preview window Y position (screen coords).")
    sp.add_argument("--no-autoplace", action="store_true", help="Don’t try to auto-place the preview window.")
    sp.add_argument("--window-native", action="store_true",
                    help="Capture a specific application window natively via FFmpeg (gdigrab) instead of screen.")
    sp.add_argument("--ffmpeg-bin", type=str, default="ffmpeg",
                    help="Path to ffmpeg executable or folder (default: find 'ffmpeg' on PATH).")
    sp.add_argument("--list-windows", action="store_true",
                    help="List visible window titles (Windows) and exit.")
    sp.add_argument("--preview-zoom", type=float, default=1.0,
                    help="Scale the preview window only (does not change image pixels).")
    sp.add_argument("--lock-base-size", action="store_true",
                    help="Force the pipeline output to match the base capture size exactly.")
    sp.set_defaults(func=cmd_stream)

    rp = sub.add_parser("run", help="Run one or more generators (pipeline).")
    rp.add_argument("--url", required=True, help="HTTP(S) URL, file:// URL, or local path.")
    rp.add_argument("--pipeline", help="Pipe generators as 'g1|g2|g3'. (Quote on PowerShell)")
    rp.add_argument("--generator", choices=REGISTRY.names(), help="(Deprecated) Single generator.")
    rp.add_argument("--out", type=Path, required=True, help="Output image file (png/jpg/webp).")
    rp.add_argument("--seed", type=int, default=None, help="RNG seed (optional).")
    rp.add_argument("--max-size", type=int, default=None, help="Downscale input longest side before processing.")
    rp.add_argument("--scale", type=int, default=1, help="Final upscale factor via Lanczos (1=off).")
    rp.add_argument(
        "--extra",
        nargs="*",
        help=(
            "Extra k=v pairs. Unprefixed apply to all stages. "
            "Use name.key=val or index.key=val for per-stage (e.g., palette_voronoi.sites=900 or 1.tile=12). "
            "Common low-RAM keys for palette_voronoi: work_mp, max_ram_mb, precision=f16."
        ),
    )
    rp.set_defaults(func=cmd_run)

    bp = sub.add_parser("bench", help="Micro-benchmark a pipeline (or single generator).")
    bp.add_argument("--url", required=True)
    bp.add_argument("--pipeline", help="Pipe generators as 'g1|g2|g3'.")
    bp.add_argument("--generator", choices=REGISTRY.names(), help="(Deprecated) Single generator.")
    bp.add_argument("--runs", type=int, default=3)
    bp.add_argument("--extra", nargs="*")
    bp.set_defaults(func=cmd_bench)

    return p


# =============== Commands ===============
def cmd_list(_args: argparse.Namespace) -> int:
    print("Available generators:", ", ".join(REGISTRY.names()) or "(none)")
    return 0


def _normalize_src_for_ffmpeg(src: str) -> str:
    p = urlparse(src)
    if (p.scheme or "").lower() == "file":
        local_path = unquote(p.path)
        if os.name == "nt" and local_path.startswith("/"):
            local_path = local_path[1:]
        return local_path
    return src

def cmd_run(args: argparse.Namespace) -> int:
    try:
        # Parse pipeline first (so errors show early)
        stages = _parse_pipeline(args.pipeline, args.generator)
        unknown = [s for s in stages if s not in REGISTRY.names()]
        if unknown:
            raise SystemExit(f"Unknown generator(s) in pipeline: {', '.join(unknown)}")

        raw_extras = _parse_kv_pairs(args.extra)
        stage_extras = _split_stage_extras(stages, raw_extras)

        # Decide whether this is video mode
        # NOTE: content_type is unknown here unless you fetch; use extension + out suffix.
        video_out = _is_video_output_path(args.out)
        video_in = _is_video_path_or_url(args.url, content_type=None)
        if video_in or video_out:
            src_for_ffmpeg = _normalize_src_for_ffmpeg(args.url)

            # Optional: allow these via --extra if you want (simple defaults here)
            ffmpeg_bin = str(raw_extras.get("ffmpeg_bin", "ffmpeg"))
            video_fps = raw_extras.get("video_fps", None)
            video_fps = float(video_fps) if video_fps is not None else None
            crf = int(raw_extras.get("crf", 18))
            preset = str(raw_extras.get("preset", "veryfast"))

            _run_video_pipeline(
                src=src_for_ffmpeg,
                out_path=args.out,
                stages=stages,
                stage_extras=stage_extras,
                seed=args.seed,
                max_size=int(args.max_size) if args.max_size else None,
                scale=int(args.scale) if args.scale else 1,
                ffmpeg_bin=ffmpeg_bin,
                video_fps=video_fps,
                crf=crf,
                preset=preset,
            )
            log.info("Saved video %s", args.out)
            return 0

        # ---- IMAGE MODE (your existing behavior) ----
        fetcher = FileFetcher()
        loader = ImageLoader()

        raw, ctype = fetcher.fetch(args.url)
        src_img = loader.load(raw, ctype, max_size=args.max_size)

        out_img = _run_pipeline(src_img, stages, stage_extras, seed=args.seed)

        if args.scale and args.scale > 1:
            w, h = out_img.size
            out_img = out_img.resize((w * args.scale, h * args.scale), Image.Resampling.LANCZOS)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        fmt = _infer_format_from_path(args.out)
        out_img.save(args.out, format=fmt, optimize=True)
        log.info("Saved %s (%dx%d)", args.out, *out_img.size)
        return 0

    except MemoryError:
        log.error("MemoryError: try lowering palette_voronoi work_mp, max_ram_mb, or grid; and set precision=f16.")
        return 1
    except Exception as e:
        log.exception("Failed: %s", e)
        return 1


def cmd_bench(args: argparse.Namespace) -> int:
    try:
        fetcher = FileFetcher()
        loader = ImageLoader()
        raw, ctype = fetcher.fetch(args.url)
        src_img = loader.load(raw, ctype)

        stages = _parse_pipeline(args.pipeline, args.generator)
        unknown = [s for s in stages if s not in REGISTRY.names()]
        if unknown:
            raise SystemExit(f"Unknown generator(s) in pipeline: {', '.join(unknown)}")

        raw_extras = _parse_kv_pairs(args.extra)
        stage_extras = _split_stage_extras(stages, raw_extras)

        times = []
        for _ in range(max(1, args.runs)):
            t0 = time.perf_counter()
            _ = _run_pipeline(src_img, stages, stage_extras, seed=None)
            times.append(time.perf_counter() - t0)
        avg = sum(times) / len(times)
        print(
            f"{'|'.join(stages)}: {args.runs} run(s) — avg {avg*1000:.2f} ms, "
            f"min {min(times)*1000:.2f} ms, max {max(times)*1000:.2f} ms"
        )
        return 0
    except Exception as e:
        log.exception("Bench failed: %s", e)
        return 1


# =============== Win helpers & UI placement ===============
def _parse_region(s: Optional[str]) -> Optional[Dict[str, int]]:
    if not s:
        return None
    try:
        x, y, w, h = [int(v.strip()) for v in s.split(",")]
        return {"left": x, "top": y, "width": w, "height": h}
    except Exception:
        raise SystemExit("Invalid --region, expected 'x,y,w,h' (integers)")


def _pil_to_bgr(img: Image.Image) -> "np.ndarray":
    arr = np.asarray(img.convert("RGB"))
    return arr[:, :, ::-1].copy()


def _set_process_dpi_aware_windows() -> None:
    try:
        if ctypes is None or not hasattr(ctypes, "windll"):
            return
        user32 = ctypes.windll.user32
        try:
            # Windows 10+: per-monitor v2 (best)
            # -4 == DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
            DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4 & 0xFFFFFFFFFFFFFFFF)
            user32.SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
            return
        except Exception:
            pass
        try:
            # Fallback: per-monitor v1
            shcore = ctypes.windll.shcore
            # 2 == PROCESS_PER_MONITOR_DPI_AWARE
            shcore.SetProcessDpiAwareness(2)
            return
        except Exception:
            pass
        try:
            # Last resort: system aware
            user32.SetProcessDPIAware()
        except Exception:
            pass
    except Exception:
        pass


def list_visible_windows() -> List[str]:
    titles: List[str] = []
    if win32gui is None:
        return titles

    def enum_cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            t = (win32gui.GetWindowText(hwnd) or "").strip()
            if t:
                titles.append(t)

    try:
        win32gui.EnumWindows(enum_cb, None)
    except Exception:
        pass
    # unique & sorted
    seen = set()
    out = []
    for t in titles:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return sorted(out)


def _resolve_window_title_like(query: str) -> Optional[str]:
    if win32gui is None:
        return None
    titles = list_visible_windows()
    for t in titles:  # exact first
        if t == query:
            return t
    q = query.lower()
    for t in titles:  # then substring
        if q in t.lower():
            return t
    return None


def _get_window_bbox_windows(title_substr: str, trim: int = 0) -> Optional[Dict[str, int]]:
    if win32gui is None:
        return None

    title_sub = title_substr.lower()
    found_rect = None

    def enum_cb(hwnd, _):
        nonlocal found_rect
        if found_rect is not None:
            return
        if not win32gui.IsWindowVisible(hwnd) or win32gui.IsIconic(hwnd):
            return
        title = win32gui.GetWindowText(hwnd) or ""
        if title_sub in title.lower():
            try:
                l, t, r, b = win32gui.GetWindowRect(hwnd)
                l += trim; t += trim; r -= trim; b -= trim
                w = max(1, r - l); h = max(1, b - t)
                found_rect = {"left": int(l), "top": int(t), "width": int(w), "height": int(h)}
            except Exception:
                pass

    try:
        win32gui.EnumWindows(enum_cb, None)
    except Exception:
        return None
    return found_rect


def _virtual_bounds(monitors: List[Dict[str, int]]) -> Tuple[int, int, int, int]:
    L = min(m["left"] for m in monitors)
    T = min(m["top"] for m in monitors)
    R = max(m["left"] + m["width"] for m in monitors)
    B = max(m["top"] + m["height"] for m in monitors)
    return L, T, R, B


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(v, hi))


def _rects_overlap(a: Dict[str, int], b: Dict[str, int]) -> bool:
    ax2, ay2 = a["left"] + a["width"], a["top"] + a["height"]
    bx2, by2 = b["left"] + b["width"], b["top"] + b["height"]
    return not (ax2 <= b["left"] or bx2 <= a["left"] or ay2 <= b["top"] or by2 <= a["top"])


def _autoplace_preview(bbox: Dict[str, int], pw: int, ph: int, mons: List[Dict[str, int]]) -> Tuple[int, int]:
    L, T, R, B = _virtual_bounds(mons)
    candidates = [
        (bbox["left"] + bbox["width"] + 20, bbox["top"] + 20),
        (bbox["left"] - pw - 20, bbox["top"] + 20),
        (bbox["left"] + 20, bbox["top"] + bbox["height"] + 20),
        (bbox["left"] + 20, bbox["top"] - ph - 20),
        (R - pw - 20, T + 20),
    ]
    for x, y in candidates:
        x = _clamp(x, L, R - pw)
        y = _clamp(y, T, B - ph)
        test = {"left": x, "top": y, "width": pw, "height": ph}
        if not _rects_overlap(test, bbox):
            return x, y
    return _clamp(R - pw - 20, L, R - pw), _clamp(T + 20, T, B - ph)


# =============== FFmpeg (gdigrab) native window capture ===============
def _resolve_ffmpeg(bin_hint: str) -> str:
    p = Path(bin_hint)
    if p.is_dir():
        for name in ("ffmpeg.exe", "ffmpeg"):
            cand = p / name
            if cand.exists():
                return str(cand)
    if p.exists() and p.is_file():
        return str(p)
    hit = shutil.which(bin_hint)
    if hit:
        return hit
    raise FileNotFoundError(f"FFmpeg not found: {bin_hint}")


def _iter_ffmpeg_window_mjpeg(
    *,
    ffmpeg_bin: str,
    fps: float,
    max_width: Optional[int],
    title: Optional[str] = None,
    region: Optional[Dict[str, int]] = None,
    hwnd: Optional[int] = None,
):
    """
    Fast, low-latency capture with auto fallback:
      1) ddagrab (preferred) → RAW bgr24 if region size is known
      2) ddagrab → MJPEG (even-sized)
      3) gdigrab → MJPEG (even-sized)
    Yields PIL RGB images.
    """
    ffmpeg_path = _resolve_ffmpeg(ffmpeg_bin)

    def build_in_opts(api: str) -> List[str]:
        if hwnd is not None and api == "gdigrab":
            return ["-f","gdigrab","-framerate",f"{max(1.0,float(fps))}","-draw_mouse","0","-i",f"hwnd={_format_hwnd(hwnd)}"]
        if title and api == "gdigrab":
            return ["-f","gdigrab","-framerate",f"{max(1.0,float(fps))}","-draw_mouse","0","-i",f"title={title}"]
        if region:
            x,y,w,h = region["left"],region["top"],region["width"],region["height"]
            if api == "gdigrab":
                return ["-f","gdigrab","-framerate",f"{max(1.0,float(fps))}","-draw_mouse","0",
                        "-offset_x",str(int(x)),"-offset_y",str(int(y)),
                        "-video_size",f"{int(w)}x{int(h)}","-i","desktop"]
            else:
                return ["-f","ddagrab","-framerate",f"{max(1.0,float(fps))}","-draw_mouse","0",
                        "-offset_x",str(int(x)),"-offset_y",str(int(y)),
                        "-video_size",f"{int(w)}x{int(h)}","-i","desktop"]
        return (["-f","gdigrab","-framerate",f"{max(1.0,float(fps))}","-draw_mouse","0","-i","desktop"]
                if api=="gdigrab" else
                ["-f","ddagrab","-framerate",f"{max(1.0,float(fps))}","-draw_mouse","0","-i","desktop"])

    def build_filters(even_only: bool = False) -> Optional[str]:
        vf = []
        if max_width:
            vf.append(f"scale='min(iw,{int(max_width)})':-1")
        if even_only:
            vf.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")
        return ",".join(vf) if vf else None

    def launch(cmd: List[str]):
        # Strongly reduce ffmpeg chatter; -nostats prevents periodic progress lines
        base = [ffmpeg_path, "-hide_banner", "-nostats", "-loglevel", "error"]
        full = base + cmd
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        log.debug("Spawning ffmpeg: %s", " ".join(full))
        proc = subprocess.Popen(
            full, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=flags, bufsize=0  # unbuffered pipes
        )
        if proc.stderr is None or proc.stdout is None:
            raise RuntimeError("Failed to start ffmpeg (no pipes).")
        drainer = _PipeDrainer(proc.stderr)
        drainer.start()
        return proc, drainer

    candidates: List[Tuple[str,str,List[str],str]] = []

    # Region → raw (fastest)
    if region:
        in_dd = build_in_opts("ddagrab")
        raw_cmd = [*in_dd, "-use_wallclock_as_timestamps","1","-vsync","0",
                   "-pix_fmt","bgr24","-f","rawvideo","-"]
        candidates.append(("ddagrab","raw",raw_cmd,"ddagrab→raw"))

    # ddagrab → mjpeg
    in_dd = build_in_opts("ddagrab")
    vf_dd = build_filters(even_only=True)
    dd_mjpeg = [*in_dd,"-use_wallclock_as_timestamps","1","-vsync","0"]
    if vf_dd: dd_mjpeg += ["-vf", vf_dd]
    dd_mjpeg += ["-vcodec","mjpeg","-q:v","6","-f","image2pipe","-"]
    candidates.append(("ddagrab","mjpeg",dd_mjpeg,"ddagrab→mjpeg"))

    # gdigrab → mjpeg
    in_gd = build_in_opts("gdigrab")
    vf_gd = build_filters(even_only=True)
    gd_mjpeg = [*in_gd,"-use_wallclock_as_timestamps","1","-vsync","0"]
    if vf_gd: gd_mjpeg += ["-vf", vf_gd]
    gd_mjpeg += ["-vcodec","mjpeg","-q:v","6","-f","image2pipe","-"]
    candidates.append(("gdigrab","mjpeg",gd_mjpeg,"gdigrab→mjpeg"))

    last_err = ""
    for api, mode, cmd, desc in candidates:
        proc = drainer = None
        try:
            proc, drainer = launch(cmd)
        except Exception as e:
            log.warning("ffmpeg launch failed for %s: %s", desc, e)
            continue

        try:
            last_ok = time.time()
            if mode == "raw":
                w, h = int(region["width"]), int(region["height"])
                frame_bytes = w * h * 3
                while True:
                    buf = proc.stdout.read(frame_bytes)
                    if not buf or len(buf) < frame_bytes:
                        last_err = (drainer.last_text() if drainer else "") or "raw stream ended"
                        break
                    frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3)  # BGR
                    img = Image.fromarray(frame[:, :, ::-1], "RGB")
                    yield img
                    last_ok = time.time()

            else:
                buf = bytearray()
                SOI, EOI = b"\xFF\xD8", b"\xFF\xD9"
                while True:
                    chunk = proc.stdout.read(1 << 20)
                    if not chunk:
                        last_err = (drainer.last_text() if drainer else "") or "mjpeg stream ended"
                        break
                    buf += chunk
                    while True:
                        i = buf.find(SOI)
                        if i < 0:
                            if len(buf) > 2: buf[:] = buf[-2:]
                            break
                        j = buf.find(EOI, i + 2)
                        if j < 0:
                            if i > 0: del buf[:i]
                            break
                        frame = bytes(buf[i:j+2]); del buf[:j+2]
                        try:
                            img = Image.open(io.BytesIO(frame)); img.load()
                            if img.mode != "RGB": img = img.convert("RGB")
                            yield img
                            last_ok = time.time()
                        except Exception:
                            continue

                    if time.time() - last_ok > 3.0:
                        if proc.poll() is not None:
                            last_err = (drainer.last_text() if drainer else "") or f"{desc} ended"
                            break
                        last_ok = time.time()
        finally:
            try:
                if proc:
                    proc.kill()
            except Exception:
                pass
            if drainer:
                drainer.stop()

    if last_err:
        log.error("ffmpeg capture failed: %s", last_err)
    else:
        log.error("ffmpeg capture failed with no output.")

def _find_hwnd_by_title_like(query: str) -> Optional[int]:
    if win32gui is None:
        return None

    q_lower = query.lower()
    exact_hwnd: Optional[int] = None
    substr_hwnd: Optional[int] = None

    def enum_cb(hwnd, _):
        nonlocal exact_hwnd, substr_hwnd
        if exact_hwnd is not None:
            return  # Found exact match, stop searching
        if not win32gui.IsWindowVisible(hwnd) or win32gui.IsIconic(hwnd):
            return

        t = (win32gui.GetWindowText(hwnd) or "")
        if not t:
            return

        if t == query:  # Prioritize exact match
            exact_hwnd = hwnd
            return

        if substr_hwnd is None and q_lower in t.lower():  # Store first substring match
            substr_hwnd = hwnd

    try:
        win32gui.EnumWindows(enum_cb, None)
    except Exception:
        return None

    return exact_hwnd or substr_hwnd  # Return exact match if found, else substring
def _format_hwnd(hwnd: int) -> str:
    return f"0x{hwnd:08x}"
def _run_pipeline_instances(
    img: Image.Image,
    gens: List[Any],
    stage_extras: List[Dict[str, Any]],
) -> Image.Image:
    out = img
    for gen, extras in zip(gens, stage_extras):
        out = gen.generate(out, **extras)
    return out
# =============== Streaming ===============
def cmd_stream(args: argparse.Namespace) -> int:
    """Capture → pipeline → preview (and optional video write) with stable high FPS and correct lifetimes."""
    # ---- Quick utility: list windows ----
    if getattr(args, "list_windows", False):
        for t in list_visible_windows():
            print(t)
        return 0

    # ---- Dependencies ----
    if not getattr(args, "window_native", False) and mss is None:
        log.error("Missing dependency: mss. Install with 'pip install mss'")
        return 1

    _set_process_dpi_aware_windows()

    # ---- Helpers ----
    def _open_preview_once(win_name: str, img: Image.Image, *, zoom: float,
                           px: Optional[int], py: Optional[int]) -> None:
        if cv2 is None:
            return
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        win_w, win_h = max(1, int(img.width * zoom)), max(1, int(img.height * zoom))
        try:
            cv2.resizeWindow(win_name, win_w, win_h)
        except Exception:
            pass
        if px is not None and py is not None:
            try:
                cv2.moveWindow(win_name, int(px), int(py))
            except Exception:
                pass

    # ---- Build pipeline ----
    try:
        stages = _parse_pipeline(args.pipeline, args.generator)

        unknown = [s for s in stages if s not in REGISTRY.names()]
        if unknown:
            raise SystemExit(f"Unknown generator(s): {', '.join(unknown)}")
        raw_extras = _parse_kv_pairs(args.extra)
        stage_extras = _split_stage_extras(stages, raw_extras)

    except Exception as e:
        log.exception("Pipeline build failed: %s", e)
        return 1

    # ---- Runtime values ----
    target_fps = max(1.0, float(args.fps))
    frame_dt = 1.0 / target_fps
    max_size = int(args.max_size) if args.max_size and args.max_size > 0 else None
    seed = args.seed
    t_end = (time.time() + float(args.dur)) if args.dur and args.dur > 0 else None
    gens = [REGISTRY.create(name, seed=seed) for name in stages]
    preview_name = "imagegen stream"
    show = bool(args.preview and cv2 is not None)
    zoom = float(args.preview_zoom or 1.0)
    px = int(args.preview_x) if args.preview_x is not None else None
    py = int(args.preview_y) if args.preview_y is not None else None

    # Video writer (lazy init)
    writer = None
    writer_is_ffmpeg = False
    base_size_ref: List[Tuple[int, int]] = []  # set on first processed frame

    def process_frame(img: Image.Image, *, lock_to_base: bool) -> Image.Image:
        # Downscale before pipeline for speed
        proc = img
        if max_size:
            proc = proc.copy()
            proc.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        out = _run_pipeline_instances(proc, gens, stage_extras)
        if not base_size_ref:
            base_size_ref.append(out.size)

        if (lock_to_base or writer is not None) and out.size != base_size_ref[0]:
            out = out.resize(base_size_ref[0], Image.Resampling.LANCZOS)

        if args.scale and args.scale > 1:
            w, h = out.size
            out = out.resize((w * args.scale, h * args.scale), Image.Resampling.LANCZOS)

        return out

    # =================== Native (FFmpeg ddagrab/gdigrab) path ===================
    if getattr(args, "window_native", False):
        if not args.window:
            log.error("--window-native requires --window <title substring or exact caption>.")
            return 1

        hwnd = _find_hwnd_by_title_like(args.window) if args.window else None
        win_rect = _get_window_bbox_windows(args.window, trim=int(args.window_trim)) if args.window else None
        resolved_title = _resolve_window_title_like(args.window) if args.window else None

        if hwnd is not None:
            log.info("Native window capture by HWND: %s", _format_hwnd(hwnd))
        elif resolved_title:
            log.info("Native window capture by title: %r", resolved_title)
        else:
            log.warning("Could not resolve HWND or exact title from %r; will try region fallback.", args.window)

        if win_rect is None:
            log.error("Could not find a visible window rectangle for: %r. Ensure it is visible and not minimized.",
                      args.window)
            return 1

        def frames_by_hwnd():
            return _iter_ffmpeg_window_mjpeg(ffmpeg_bin=args.ffmpeg_bin, fps=float(args.fps),
                                             max_width=max_size, hwnd=hwnd)

        def frames_by_title():
            return _iter_ffmpeg_window_mjpeg(ffmpeg_bin=args.ffmpeg_bin, fps=float(args.fps),
                                             max_width=max_size, title=resolved_title)

        def frames_by_region():
            return _iter_ffmpeg_window_mjpeg(ffmpeg_bin=args.ffmpeg_bin, fps=float(args.fps),
                                             max_width=max_size, region=win_rect)

        attempts = []
        if hwnd is not None:
            attempts.append(frames_by_hwnd)
        if resolved_title:
            attempts.append(frames_by_title)
        attempts.append(frames_by_region)

        log.info("Streaming via FFmpeg @ %.1f FPS. Ctrl+C to stop.", target_fps)

        got_any = False  # <-- define outside loop to avoid lifetime bug
        try:
            for attempt in attempts:
                try:
                    frames = attempt()
                except Exception as e:
                    log.exception("Failed to start ffmpeg capture: %s", e)
                    continue

                next_t = perf_counter()
                try:
                    for img in frames:
                        got_any = True
                        if t_end and time.time() >= t_end:
                            break

                        out = process_frame(img, lock_to_base=True)

                        # Lazy-init preview
                        if show and len(base_size_ref) == 1:
                            _open_preview_once(preview_name, out, zoom=zoom, px=px, py=py)

                        # Lazy-init FFmpeg writer
                        if args.out_video and writer is None:
                            writer = FFMpegWriter(
                                str(args.out_video), out.width, out.height,
                                float(args.video_fps or args.fps), crf=16, preset="veryfast"
                            )
                            writer_is_ffmpeg = True

                        if show:
                            cv2.imshow(preview_name, _pil_to_bgr(out))
                            cv2.waitKey(1)

                        if writer is not None:
                            if writer_is_ffmpeg:
                                writer.write_bgr(_pil_to_bgr(out))
                            else:
                                # Should not happen in native mode, but keep safe
                                writer.write(_pil_to_bgr(out))
                        now = perf_counter()
                        if now < next_t:
                            sleep(min(0.002, next_t - now))
                            # keep the newest frame by continuing; OR you can process anyway.
                            # I'd continue to keep timing stable:
                            continue

                        # If we're *very* behind, drop frames until we're close again
                        if now - next_t > 2 * frame_dt:
                            next_t = now  # jump forward; effectively drops backlog

                        # pacing
                        now = perf_counter()
                        next_t += frame_dt
                        dt = next_t - now
                        if dt > 0.001:
                            sleep(dt)
                        elif dt < -0.2:
                            next_t = now
                finally:
                    if got_any:
                        break

            if not got_any:
                log.error("FFmpeg did not deliver any frames via HWND, title, or region capture.")
                return 1
            return 0

        except KeyboardInterrupt:
            return 0
        except Exception as e:
            log.exception("Streaming failed: %s", e)
            return 1
        finally:
            try:
                if writer is not None:
                    if writer_is_ffmpeg:
                        writer.close()
                    else:
                        writer.release()
            except Exception:
                pass
            writer = None
            if show:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

    # =================== MSS (screen/region) path ===================
    mon_idx = max(1, int(args.monitor))
    region = _parse_region(args.region)
    window_bbox = None
    if args.window:
        window_bbox = _get_window_bbox_windows(args.window, trim=int(args.window_trim))
        if window_bbox is None:
            log.error("Window not found (title contains): %r", args.window)
            return 1

    with mss.mss() as sct:
        if window_bbox:
            bbox = dict(window_bbox)
            region_desc = f"window:'{args.window}'"
        elif region:
            bbox = region
            region_desc = "custom-region"
        else:
            mons = sct.monitors
            if mon_idx < 1 or mon_idx >= len(mons):
                raise SystemExit(f"Monitor index {mon_idx} out of range. Available: 1..{len(mons)-1}")
            bbox = mons[mon_idx]
            region_desc = f"monitor:{mon_idx}"

        # Prime capture & base size
        raw = sct.grab(bbox)
        first_img = Image.frombytes("RGB", (raw.width, raw.height), raw.rgb)
        out = process_frame(first_img, lock_to_base=bool(args.lock_base_size))
        base_size = out.size  # tuple

        if show:
            _open_preview_once(preview_name, out, zoom=zoom, px=px, py=py)

        # OpenCV writer for MSS path
        if args.out_video:
            if cv2 is None:
                log.error("--out-video needs OpenCV. Install with 'pip install opencv-python'")
                return 1
            fourcc = cv2.VideoWriter_fourcc(*("mp4v" if str(args.out_video).lower().endswith(".mp4") else "XVID"))
            vid_fps = float(args.video_fps or target_fps)
            writer = cv2.VideoWriter(str(args.out_video), fourcc, vid_fps, base_size)
            writer_is_ffmpeg = False

        log.info("Streaming %s @ %.1f FPS. Ctrl+C to stop.", region_desc, target_fps)

        try:
            # ---- inside cmd_stream, MSS path ----

            latest = deque(maxlen=1)
            stop_ev = threading.Event()

            def capture_loop():
                # Capture as fast as possible (or you can pace this too)
                while not stop_ev.is_set():
                    try:
                        raw = sct.grab(bbox)
                        # Keep newest only (deque maxlen=1)
                        latest.append(raw)
                    except Exception:
                        # tiny backoff
                        time.sleep(0.002)

            cap_t = threading.Thread(target=capture_loop, daemon=True)
            cap_t.start()

            next_t = perf_counter()

            try:
                while True:
                    if t_end and time.time() >= t_end:
                        break

                    # If following window, update bbox (capture thread will use new bbox next loop)
                    if args.window and args.follow_window:
                        wb = _get_window_bbox_windows(args.window, trim=int(args.window_trim))
                        if wb is not None:
                            bbox = wb

                    now = perf_counter()
                    if now < next_t:
                        sleep(min(0.002, next_t - now))
                        continue

                    if not latest:
                        # no frame yet
                        sleep(0.001)
                        continue

                    raw = latest.pop()  # newest frame
                    # IMPORTANT: mss raw is BGRA; use that to avoid extra conversions
                    img = Image.frombuffer("RGB", (raw.width, raw.height), raw.bgra, "raw", "BGRX", 0, 1)

                    t0 = perf_counter()
                    out = process_frame(img, lock_to_base=bool(args.lock_base_size or writer is not None))
                    t1 = perf_counter()

                    if show:
                        cv2.imshow(preview_name, _pil_to_bgr(out))
                        cv2.waitKey(1)

                    if writer is not None:
                        if out.size != base_size:
                            out = out.resize(base_size, Image.Resampling.LANCZOS)
                        writer.write(_pil_to_bgr(out))

                    next_t += frame_dt
                    if t1 - next_t > 2 * frame_dt:
                        next_t = t1

            finally:
                stop_ev.set()
                try:
                    cap_t.join(timeout=0.5)
                except Exception:
                    pass


        except KeyboardInterrupt:
            pass
        except Exception as e:
            log.exception("Streaming failed: %s", e)
            return 1
        finally:
            try:
                if writer is not None:
                    if writer_is_ffmpeg:
                        writer.close()
                    else:
                        writer.release()
            except Exception:
                pass
            writer = None
            if show:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

    return 0



# =============== Entry ===============
def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
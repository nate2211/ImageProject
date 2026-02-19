"""
content.py — photo-safe ContentGenerator (no cell-division / no mosaic)
Rewritten with introspection for GUI parameter discovery.

What this does
--------------
• Strict allow-list: only `photo_enhance`, `learn_profile`, `profile_context` (overlay-only).
• Auto-sanitize any `profile_context` use to an overlay that preserves luminance/edges.
• Auto-wire: if you request `profile_context` but no profile path, we insert `learn_profile`
  before it and route a safe default `learn_profile.out`.
• Potent, guarded presets: crisper enhancements and stronger (but safe) profile tints.
• Extra safety clamps: caps for overlay strength, quantization off, non-cell motifs only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# We assume main.log and palettes.REGISTRY exist in your project structure
# If running standalone, you might need to adjust imports.
try:
    from main import log
except ImportError:
    import logging
    log = logging.getLogger("content")

from palettes import REGISTRY  # type: ignore

# ---------------------------- Safety configuration ---------------------------- #

SAFE_STAGES = {
    "photo_enhance",    # pixel-preserving enhancer
    "learn_profile",    # analysis only, passthrough
    "profile_context",  # allowed ONLY in overlay mode with detail-preservation
}

SAFE_OVERLAY_MOTIFS = {"clouds", "waves", "stripes"}
DEFAULT_OVERLAY_MOTIF = "clouds"

# Upper bounds to keep overlays subtle and protect detail
MAX_OVERLAY_ALPHA = 0.45
MIN_EDGE_PROTECT  = 0.60
FORCED_Q_LEVELS   = 0  # disable posterization

DEFAULT_PROFILE_PATH = "content.profile.npz"  # used when auto-wiring learn_profile


def _coerce(v: str) -> Any:
    try:
        if v.isdigit():
            return int(v)
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        low = str(v).lower()
        if low in ("true", "false"):
            return low == "true"
    return v


def _parse_pipeline(spec: Optional[str]) -> List[str]:
    if not spec:
        return []
    stages = [s.strip().lower() for s in spec.split("|") if s.strip()]
    if not stages:
        raise ValueError("Empty pipeline. Example: photo_enhance|photo_enhance")
    return stages


def _split_stage_extras(stages: List[str], raw_extras: Dict[str, Any]) -> List[Dict[str, Any]]:
    stage_extras = [dict() for _ in stages]
    global_extras: Dict[str, Any] = {}
    name_targets: Dict[str, Dict[str, Any]] = {}
    index_targets: Dict[int, Dict[str, Any]] = {}

    for k, v in raw_extras.items():
        if "." not in k:
            global_extras[k] = v
            continue
        prefix, key = k.split(".", 1)
        prefix = prefix.strip().lower(); key = key.strip()
        if prefix == "all":
            global_extras[key] = v
        elif prefix.isdigit():
            idx = int(prefix)
            if 0 <= idx < len(stages):
                index_targets.setdefault(idx, {})[key] = v
        else:
            name_targets.setdefault(prefix, {})[key] = v

    for i, name in enumerate(stages):
        merged: Dict[str, Any] = {}
        merged.update(global_extras)
        if name in name_targets:
            merged.update(name_targets[name])
        if i in index_targets:
            merged.update(index_targets[i])
        stage_extras[i] = merged
    return stage_extras


def _sanitize_profile_context(extras: Dict[str, Any]) -> Dict[str, Any]:
    """
    Harden profile_context extras so it cannot introduce cell visuals or muddy detail.
    """
    safe = dict(extras)

    # Absolute locks
    safe["compose"] = "overlay"
    safe["preserve_details"] = True

    # Motif
    motif = str(safe.get("motif", "")).strip().lower()
    if motif not in SAFE_OVERLAY_MOTIFS:
        safe["motif"] = DEFAULT_OVERLAY_MOTIF

    # Strength caps / guards
    alpha = float(safe.get("overlay_alpha", 0.32))
    safe["overlay_alpha"] = max(0.00, min(float(alpha), MAX_OVERLAY_ALPHA))
    edge_protect = float(safe.get("edge_protect", 0.78))
    safe["edge_protect"] = max(MIN_EDGE_PROTECT, min(edge_protect, 0.95))

    # Quality: avoid posterization
    safe["q_levels"] = FORCED_Q_LEVELS

    # Style knobs (defaults favor subtlety)
    defaults = {
        "contrast": 0.12,
        "substance": 0.0,
        "edge_glow": 0.0,
        "vignette": 0.0,
        "smooth": 0.12,
        "smoothness": 0.16,
        "clarity": 0.10,
        "saturation": 1.02,
        # geometry guidance fully off
        "photo_density": 0.0,
        "photo_orient": 0.0,
        # anti-cast & detail guards (if learning.py supports them)
        "preserve_blacks": 0.40,
        "protect_highlights": 0.20,
    }
    for k, v in defaults.items():
        safe.setdefault(k, v)

    return safe


def _ensure_profile_wiring(stages: List[str], stage_extras: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    If there's a `profile_context` stage without a profile path, insert `learn_profile`.
    """
    if "profile_context" not in stages:
        return stages, stage_extras

    # Find first profile_context index
    idx_pc = stages.index("profile_context")
    pc_ex = stage_extras[idx_pc] = _sanitize_profile_context(stage_extras[idx_pc])

    need_insert = False
    prof_path = str(pc_ex.get("profile", "")).strip()
    lp_out_key = "out"
    if not prof_path:
        # See if a prior learn_profile sets out; if not, we will insert one.
        for i in range(idx_pc):
            if stages[i] == "learn_profile":
                maybe = stage_extras[i].get(lp_out_key)
                if maybe:
                    pc_ex["profile"] = maybe
                    break
        else:
            need_insert = True

    if need_insert:
        stages = stages[:idx_pc] + ["learn_profile"] + stages[idx_pc:]
        stage_extras = stage_extras[:idx_pc] + [{}] + stage_extras[idx_pc:]
        # Fill defaults
        stage_extras[idx_pc]["out"] = stage_extras[idx_pc].get("out", DEFAULT_PROFILE_PATH)
        stage_extras[idx_pc + 1]["profile"] = stage_extras[idx_pc]["profile"] = stage_extras[idx_pc]["out"]
        # (Re-sanitize, though values won’t change)
        stage_extras[idx_pc + 1] = _sanitize_profile_context(stage_extras[idx_pc + 1])
    else:
        # If profile provided, still sanitize
        stage_extras[idx_pc] = _sanitize_profile_context(stage_extras[idx_pc])

    return stages, stage_extras


def _run_pipeline(src_img, stages: List[str], stage_extras: List[Dict[str, Any]], seed: Optional[int]):
    out = src_img
    for i, name in enumerate(stages):
        gen = REGISTRY.create(name, seed=seed)
        extras = stage_extras[i]
        if name == "profile_context":
            extras = _sanitize_profile_context(extras)
        out = gen.generate(out, **extras)
    return out


# ------------------------------- Presets -------------------------------- #

@dataclass(frozen=True)
class Preset:
    pipeline: str
    extras: Dict[str, Any]
    description: str = ""

PRESETS: Dict[str, Preset] = {
    "enhance_soft": Preset(
        pipeline="photo_enhance",
        extras=dict(
            denoise=0.10, clarity=0.28, local_contrast=0.18, saturation=1.04,
            protect_highlights=0.12, protect_shadows=0.08,
            auto_wb=True, tone="soft", tone_strength=0.55,
            dehaze=0.0, chroma_denoise=0.10, clarity_halo_guard=0.35, protect_skin=0.15,
            vibrance=0.10, gamut_guard=0.50, noise_floor=0.12, lc_mid_bias=0.55,
        ),
        description="Gentle, natural photo polish.",
    ),
    "enhance_crisp": Preset(
        pipeline="photo_enhance",
        extras=dict(
            denoise=0.12, clarity=0.38, local_contrast=0.28, saturation=1.05,
            protect_highlights=0.14, protect_shadows=0.08,
            auto_wb=True, wb_mode="bright", wb_strength=0.9,
            tone="soft", tone_strength=0.62,
            dehaze=0.10, dehaze_radius=7, chroma_denoise=0.12,
            clarity_halo_guard=0.45, sharpen_threshold=0.10,
            vibrance=0.12, gamut_guard=0.65, noise_floor=0.14, lc_mid_bias=0.58,
        ),
        description="Crisp detail and punchy mids.",
    ),
    "enhance_lowlight": Preset(
        pipeline="photo_enhance",
        extras=dict(
            denoise=0.20, clarity=0.24, local_contrast=0.30, saturation=1.06,
            protect_highlights=0.16, protect_shadows=0.02,
            auto_wb=True, wb_mode="median", tone="soft", tone_strength=0.65,
            shadow_lift=0.10, highlight_recovery=0.08,
            dehaze=0.22, dehaze_radius=9, chroma_denoise=0.20,
            clarity_halo_guard=0.48, vibrance=0.12, gamut_guard=0.70, noise_floor=0.16,
        ),
        description="Low-light clean-up with guarded contrast.",
    ),
    "profile_tint_soft": Preset(
        pipeline="learn_profile|profile_context|photo_enhance",
        extras={
            "profile_context.motif": "clouds",
            "profile_context.overlay_alpha": 0.28,
            "profile_context.edge_protect": 0.80,
            "profile_context.q_levels": 0,
            "profile_context.smooth": 0.12,
            "profile_context.smoothness": 0.16,
            "profile_context.clarity": 0.10,
            "photo_enhance.denoise": 0.10,
            "photo_enhance.tone": "soft",
            "photo_enhance.tone_strength": 0.55,
        },
        description="Learn global color cues, apply a gentle overlay.",
    ),
    "profile_tint_plus": Preset(
        pipeline="learn_profile|profile_context|photo_enhance",
        extras={
            "profile_context.motif": "waves",
            "profile_context.overlay_alpha": 0.34,
            "profile_context.edge_protect": 0.82,
            "profile_context.q_levels": 0,
            "profile_context.smooth": 0.14,
            "profile_context.smoothness": 0.18,
            "profile_context.clarity": 0.12,
            "profile_context.saturation": 1.03,
            "photo_enhance.denoise": 0.12,
            "photo_enhance.local_contrast": 0.24,
            "photo_enhance.clarity": 0.32,
            "photo_enhance.tone": "soft",
            "photo_enhance.tone_strength": 0.60,
            "photo_enhance.vibrance": 0.12,
            "photo_enhance.gamut_guard": 0.70,
        },
        description="A bit more color presence from the profile.",
    ),
}


# -------------------------- ContentGenerator --------------------------- #

@dataclass
class ContentGenerator:
    """
    High-level meta-generator with a strict safety allow-list.
    """
    seed: Optional[int] = None
    strict: bool = True
    allow_unknown: bool = False
    default_plan: str = "enhance_crisp"

    # -------- introspection for GUI --------
    @staticmethod
    def get_params() -> List[Dict[str, Any]]:
        """
        Return the definition of parameters for GUI generation.
        Includes meta-parameters (plan, strict) and common override knobs.
        """
        presets = sorted(PRESETS.keys())
        return [
            # Meta controls
            {
                "name": "plan",
                "type": str,
                "default": "enhance_crisp",
                "choices": presets,
                "help": "Pre-configured pipeline strategy."
            },
            {
                "name": "strict",
                "type": bool,
                "default": True,
                "help": "If True, errors on unknown stages. If False, ignores them."
            },
            {
                "name": "dry_run",
                "type": bool,
                "default": False,
                "help": "Log the pipeline plan without processing pixels."
            },

            # Common overrides (Photo Enhance)
            # These allow tweaking the underlying stages without writing manual extras
            {
                "name": "photo_enhance.denoise",
                "type": float,
                "default": 0.12,
                "min": 0.0, "max": 1.0,
                "help": "Override denoise strength for enhance plans."
            },
            {
                "name": "photo_enhance.clarity",
                "type": float,
                "default": 0.38,
                "min": 0.0, "max": 1.0,
                "help": "Override clarity/sharpness strength."
            },
            {
                "name": "photo_enhance.saturation",
                "type": float,
                "default": 1.05,
                "min": 0.0, "max": 2.0,
                "help": "Override global saturation."
            },

            # Common overrides (Profile Context)
            {
                "name": "profile_context.overlay_alpha",
                "type": float,
                "default": 0.28,
                "min": 0.0, "max": 0.45, # Clamped by safety rules anyway
                "help": "Opacity of the color overlay (tint plans only)."
            },
            {
                "name": "profile_context.motif",
                "type": str,
                "default": "clouds",
                "choices": ["clouds", "waves", "stripes"],
                "help": "Pattern style for the overlay (tint plans only)."
            },
        ]

    # -------- public helpers --------
    @staticmethod
    def list_presets() -> List[str]:
        return sorted(PRESETS.keys())

    @staticmethod
    def describe_preset(name: str) -> str:
        key = (name or "").strip().lower()
        if key not in PRESETS:
            return f"(unknown preset: {name})"
        p = PRESETS[key]
        return f"{key}: {p.description or '(no description)'} — pipeline={p.pipeline}"

    # -------- generate() entrypoint --------
    def generate(self, input_image, **kwargs):
        # meta toggles (kwargs win over ctor defaults)
        dry_run = self._to_bool(kwargs.get("dry_run", False))
        strict = self._to_bool(kwargs.get("strict", self.strict))
        allow_unknown = self._to_bool(kwargs.get("allow_unknown", self.allow_unknown))
        seed_override = kwargs.get("seed", None)
        seed = self.seed if seed_override is None else seed_override

        # accept both plan and preset
        plan = self._norm_str(kwargs.pop("plan", None) or kwargs.pop("preset", None))
        pipeline = kwargs.pop("pipeline", None)

        # default if nothing specified
        if not plan and not pipeline:
            plan = self.default_plan

        # Resolve pipeline + extras
        if plan:
            if plan not in PRESETS:
                raise KeyError(f"Unknown plan '{plan}'. Available: {', '.join(self.list_presets())}")
            spec = PRESETS[plan]
            stages = _parse_pipeline(pipeline if pipeline else spec.pipeline)
            # user extras override preset extras
            extras = {**spec.extras, **{k: self._norm_val(v) for k, v in kwargs.items()}}
        else:
            stages = _parse_pipeline(str(pipeline))
            extras = {k: self._norm_val(v) for k, v in kwargs.items()}

        # Validate against allow-list
        stages, disallowed = self._validate_stages(stages)
        if disallowed:
            msg = f"Disallowed or unknown stage(s) in pipeline: {', '.join(disallowed)}"
            if strict:
                raise SystemExit(msg)
            if not allow_unknown:
                log.warning(msg + " (ignored)")
            stages = [s for s in stages if s not in disallowed]

        # Route extras
        stage_extras, unused = self._split_stage_extras_verbose(stages, extras)
        if unused:
            log.info("Unused extras (no matching stage/index): %s", ", ".join(sorted(set(unused))))

        # Auto-wire learn_profile if needed and sanitize profile_context
        stages, stage_extras = _ensure_profile_wiring(stages, stage_extras)

        if dry_run:
            log.info("[dry_run] stages=%s", " | ".join(stages))
            for i, (name, ex) in enumerate(zip(stages, stage_extras)):
                ex_show = _sanitize_profile_context(ex) if name == "profile_context" else ex
                log.info("[dry_run] %d:%s extras=%s", i, name, {k: ex_show[k] for k in sorted(ex_show)})
            return input_image

        return _run_pipeline(input_image, stages, stage_extras, seed=seed)

    # -------- internals --------
    @staticmethod
    def _norm_str(v: Optional[str]) -> Optional[str]:
        return None if v is None else str(v).strip().lower()

    @staticmethod
    def _to_bool(v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if v is None:
            return False
        s = str(v).strip().lower()
        if s in ("true", "1", "yes", "y", "on"):
            return True
        if s in ("false", "0", "no", "n", "off"):
            return False
        return bool(v)

    def _norm_val(self, v: Any) -> Any:
        if isinstance(v, str):
            s = v.strip()
            if s.lower() in ("true", "false", "1", "0", "yes", "no", "on", "off"):
                return self._to_bool(s)
        return v

    @staticmethod
    def _validate_stages(stages: List[str]) -> Tuple[List[str], List[str]]:
        allowed = SAFE_STAGES
        disallowed = [s for s in stages if s not in allowed]
        return stages, disallowed

    @staticmethod
    def _split_stage_extras_verbose(stages: List[str], raw: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Route extras to stages; also return list of unused keys."""
        stage_extras = [dict() for _ in stages]
        unused: List[str] = []
        global_extras: Dict[str, Any] = {}
        name_targets: Dict[str, Dict[str, Any]] = {}
        index_targets: Dict[int, Dict[str, Any]] = {}

        for k, v in raw.items():
            if k in ("plan", "preset", "pipeline", "dry_run", "strict", "allow_unknown", "seed"):
                continue  # meta keys not routed
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
                    unused.append(k)
            else:
                if prefix in stages:
                    name_targets.setdefault(prefix, {})[key] = v
                else:
                    unused.append(k)

        for i, name in enumerate(stages):
            merged: Dict[str, Any] = {}
            merged.update(global_extras)
            if name in name_targets:
                merged.update(name_targets[name])
            if i in index_targets:
                merged.update(index_targets[i])
            # Sanitize in routing so dry_run/logs show final truth
            if name == "profile_context":
                merged = _sanitize_profile_context(merged)
            stage_extras[i] = merged

        return stage_extras, unused


# Register with the shared registry so `--generator content` works
REGISTRY.register("content", ContentGenerator)
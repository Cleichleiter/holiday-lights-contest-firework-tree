# animation.py
from __future__ import annotations

import math
import numpy as np
from typing import Optional, Dict, Any

from lib.base_animation import BaseAnimation
from lib.constants import NUM_PIXELS
from utils.geometry import POINTS_3D


# ----------------------------
# Small helpers (vectorized)
# ----------------------------
def _clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    # Classic smoothstep, safe for edge0==edge1
    t = (x - edge0) / (edge1 - edge0 + 1e-9)
    t = _clamp01(t)
    return t * t * (3.0 - 2.0 * t)


def _pulse01(x: np.ndarray) -> np.ndarray:
    """
    Smooth 0..1 pulse from a phase value (in cycles).
    Uses a cosine lobe: 0 at edges, 1 at center. Very musical-looking.
    """
    phase = x % 1.0
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * phase)


def _rgb_u8(rgb01: np.ndarray, gamma: float = 2.0) -> np.ndarray:
    """
    Clamp, gamma-shape, convert to uint8.
    Slight gamma (<2.2) tends to look brighter on LED-like displays.
    """
    x = _clamp01(rgb01)
    if gamma and gamma != 1.0:
        x = np.power(x, 1.0 / gamma)
    return (x * 255.0 + 0.5).astype(np.uint8)


def _hash01(i: np.ndarray, salt: float) -> np.ndarray:
    """
    Deterministic pseudo-random 0..1 per pixel using a cheap sine hash.
    No RNG state, no allocations.
    """
    # i is int array
    return np.modf(np.sin(i * 12.9898 + salt * 78.233) * 43758.5453)[0] % 1.0


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a * (1.0 - t) + b * t


# ----------------------------
# Animation
# ----------------------------
class Animation(BaseAnimation):
    """
    Beat-structured show inspired by "Last Christmas" at ~108 BPM.
    Sections (in beats):
      0-24  : intro (gold ring + shimmering rain)
      24-56 : chorus (red/white spiral, beat pulsing)
      56-80 : twinkle break (white sparkles with gentle drift)
      80+   : finale (fast multicolor chase + sparkle, ends strong)
    """

    def __init__(
        self,
        frameBuf,
        *,
        fps: Optional[int] = 30,
        bpm: float = 108.0,
        turns: float = 3.1,
        brightness: float = 1.15,
        gamma: float = 2.0,
        twinkle_density: float = 0.055,
        crossfade_beats: float = 2.0,
    ):
        super().__init__(frameBuf, fps=fps)
        self.t = 0.0

        self.bpm = float(bpm)
        self.beat_sec = 60.0 / max(1e-6, self.bpm)

        self.turns = float(turns)
        self.brightness = float(brightness)
        self.gamma = float(gamma)
        self.twinkle_density = float(twinkle_density)
        self.crossfade_beats = float(crossfade_beats)

        pts = np.asarray(POINTS_3D, dtype=np.float32)
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]

        # Normalize height 0..1
        zmin = float(z.min())
        zmax = float(z.max())
        self.z01 = (z - zmin) / (zmax - zmin + 1e-9)

        # Angle around trunk 0..1
        ang = np.arctan2(y, x).astype(np.float32)
        self.a01 = (ang + math.pi) / (2.0 * math.pi)

        # Precompute pixel indices for hashing
        self.i = np.arange(NUM_PIXELS, dtype=np.int32)

        # Colors (0..1)
        self.gold = np.array([1.00, 0.78, 0.22], dtype=np.float32)
        self.soft_gold = np.array([1.00, 0.88, 0.55], dtype=np.float32)
        self.red = np.array([1.00, 0.08, 0.10], dtype=np.float32)
        self.white = np.array([1.00, 1.00, 1.00], dtype=np.float32)

        # Finale palette
        self.palette = np.array(
            [
                [1.00, 0.10, 0.10],  # red
                [0.15, 1.00, 0.25],  # green
                [0.20, 0.55, 1.00],  # blue
                [1.00, 1.00, 0.15],  # yellow
                [1.00, 0.20, 1.00],  # magenta
                [0.15, 1.00, 1.00],  # cyan
            ],
            dtype=np.float32,
        )

    @classmethod
    def get_default_parameters(cls) -> Dict[str, Any]:
        return {
            "fps": 30,
            "bpm": 108.0,
            "turns": 3.1,
            "brightness": 1.15,
            "gamma": 2.0,
            "twinkle_density": 0.055,
            "crossfade_beats": 2.0,
        }

    def renderNextFrame(self):
        dt = 1.0 / max(1, int(self.fps or 30))
        self.t += dt

        beat = self.t / self.beat_sec  # beat counter (float)
        phase = beat % 1.0             # 0..1

        # Base frame buffer in linear 0..1 float
        rgb = np.zeros((NUM_PIXELS, 3), dtype=np.float32)

        # Helper: crossfade weight near boundaries
        def xfade(b: float, start: float, end: float) -> float:
            """
            Returns 0..1 for being inside [start,end], with smooth fade at edges.
            """
            if b < start - self.crossfade_beats or b > end + self.crossfade_beats:
                return 0.0
            # fade-in
            if b < start:
                return float(_smoothstep(0.0, 1.0, np.array([(b - (start - self.crossfade_beats)) / self.crossfade_beats], dtype=np.float32))[0])
            # fade-out
            if b > end:
                return float(1.0 - _smoothstep(0.0, 1.0, np.array([(b - end) / self.crossfade_beats], dtype=np.float32))[0])
            return 1.0

        # Section weights
        w_intro = xfade(beat, 0.0, 24.0)
        w_chor  = xfade(beat, 24.0, 56.0)
        w_twin  = xfade(beat, 56.0, 80.0)
        w_fin   = 1.0 if beat >= 80.0 else xfade(beat, 80.0, 9999.0)

        # ----------------------------
        # INTRO: gold ring + shimmering rain
        # ----------------------------
        if w_intro > 0.0:
            layer = np.zeros_like(rgb)

            # Ring: stable top band with slight shimmer
            ring_mask = self.z01 > 0.90
            shimmer = 0.75 + 0.25 * _pulse01(beat * 0.5 + self.a01 * 0.7)
            layer[ring_mask] += self.gold * (1.25 * shimmer[ring_mask])[:, None]

            # Multi-streak "rain": 3 descending highlights with slight angular offset
            # This reads more like intentional choreography than a single moving band.
            speed = 0.055  # beats -> height speed
            base = 1.05 - (beat * speed) % 1.35
            widths = [0.08, 0.10, 0.12]
            offsets = [0.00, 0.18, 0.41]

            rain = np.zeros(NUM_PIXELS, dtype=np.float32)
            for w, off in zip(widths, offsets):
                pos = (base + off) % 1.35
                d = np.abs(self.z01 - pos)
                # A smooth band that fades out
                band = _smoothstep(w, 0.0, d)
                # Add a mild twist shimmer so it isn't a flat cylinder band
                band *= 0.75 + 0.25 * _pulse01(beat * 0.8 + self.a01 * 1.3 + off)
                rain = np.maximum(rain, band)

            layer += rain[:, None] * self.soft_gold * 1.15

            rgb += layer * w_intro

        # ----------------------------
        # CHORUS: red/white spiral with beat pulse
        # ----------------------------
        if w_chor > 0.0:
            layer = np.zeros_like(rgb)

            # Spiral coordinate: angle + turns * height + time drift
            drift = beat * 0.12  # slower, smoother movement
            s = (self.a01 + self.turns * self.z01 + drift) % 1.0

            # Centered distance to spiral ridge at 0.5
            d = np.abs(s - 0.5)

            # Define a crisp-ish band, but with soft shoulders
            core = _smoothstep(0.14, 0.0, d)
            glow = _smoothstep(0.28, 0.0, d) * 0.35
            band = core + glow

            # Beat pulse: smooth lobe per beat
            p = _pulse01(beat)  # 0..1
            # Slight emphasis near the middle/top so it reads in 3D
            height_boost = 0.75 + 0.35 * self.z01

            # Red/white interchange on half-beats for that classic candy-cane feel
            swap = _pulse01(beat * 0.5)  # slower swap
            c = _lerp(self.white, self.red, swap)

            # Pulse brightness rides on the beat
            amp = 0.65 + 0.55 * p
            layer += band[:, None] * c[None, :] * (amp * height_boost)[:, None]

            rgb += layer * w_chor

        # ----------------------------
        # TWINKLE BREAK: deterministic sparkles + gentle drift
        # ----------------------------
        if w_twin > 0.0:
            layer = np.zeros_like(rgb)

            # A drifting field value per pixel; changes smoothly over time
            field = _hash01(self.i, salt=beat * 0.35)
            # Twinkles appear where field is small; density controlled
            tw = field < self.twinkle_density

            # Twinkle intensity: fade in/out based on another phase
            tw_phase = _hash01(self.i, salt=beat * 0.90 + 10.0)
            tw_int = 0.35 + 0.65 * _pulse01(tw_phase * 2.0 + phase)

            # Gentle vertical drift glow so it doesn’t look static
            drift = (self.z01 + beat * 0.02) % 1.0
            haze = 0.10 * _pulse01(drift * 1.5 + self.a01 * 0.5)

            layer += haze[:, None] * self.soft_gold[None, :] * 0.65
            layer[tw] += self.white * tw_int[tw][:, None] * 1.05

            rgb += layer * w_twin

        # ----------------------------
        # FINALE: multicolor chase + sparkle (fast + “finish strong”)
        # ----------------------------
        if w_fin > 0.0:
            layer = np.zeros_like(rgb)

            # Beat-synced global strobe (not fully off, keeps color visible)
            strobe = 0.55 + 0.45 * _pulse01(beat * 2.0)

            # Color index chase that wraps around height + angle for 3D movement
            # This reads like the tree is “spinning” colors, not random assignment.
            chase = (self.a01 * 2.0 + self.z01 * 1.3 + beat * 0.85) % 1.0
            idx = (chase * len(self.palette)).astype(np.int32) % len(self.palette)
            cols = self.palette[idx]

            # Add sparkles that get denser over time
            # Density ramps after beat 80
            ramp = _clamp01(np.array([(beat - 80.0) / 16.0], dtype=np.float32))[0]
            density = 0.03 + 0.08 * float(ramp)
            sp = _hash01(self.i, salt=beat * 1.7 + 33.0) < density
            sp_int = 0.50 + 0.50 * _pulse01(beat * 4.0 + self.a01 * 1.7)

            layer += cols * strobe
            layer[sp] += (self.white * 0.85) * sp_int[sp][:, None]

            # Keep a gold “star” ring at the top as a finishing signature
            star = self.z01 > 0.92
            star_glow = 0.75 + 0.25 * _pulse01(beat * 1.0 + self.a01 * 0.2)
            layer[star] += self.gold * (0.55 * star_glow[star])[:, None]

            rgb += layer * w_fin

        # Overall brightness shaping
        rgb *= self.brightness

        # Write frame
        self.frameBuf[:] = _rgb_u8(rgb, gamma=self.gamma)

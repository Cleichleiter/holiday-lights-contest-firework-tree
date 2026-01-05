# animation.py
from __future__ import annotations

from typing import Optional, Dict, Any, Tuple
import numpy as np

from lib.base_animation import BaseAnimation
from utils.geometry import POINTS_3D

try:
    # Provided by the contest repo
    from utils.colors import hsv_to_rgb
except Exception:
    # Fallback (should not be needed in the contest repo, but safe)
    def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
        h = float(h) % 1.0
        s = float(np.clip(s, 0.0, 1.0))
        v = float(np.clip(v, 0.0, 1.0))
        i = int(h * 6.0)
        f = h * 6.0 - i
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        i %= 6
        if i == 0:
            r, g, b = v, t, p
        elif i == 1:
            r, g, b = q, v, p
        elif i == 2:
            r, g, b = p, v, t
        elif i == 3:
            r, g, b = p, q, v
        elif i == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q
        return int(r * 255), int(g * 255), int(b * 255)


def _clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a * (1.0 - t) + b * t


class TreeBreathingColor(BaseAnimation):
    """
    Tree Breathing Color
    - Whole-tree "breathing" brightness envelope (smooth sine/cosine).
    - One dominant color per breath group (Blue -> Teal -> Green -> Gold).
    - Color transitions occur only at the bottom of the exhale (phase wrap),
      then crossfade smoothly.
    - Finale: faster multicolor breathing/shimmer (high visibility on a real tree).
    """

    def __init__(
        self,
        frameBuf,
        *,
        fps: Optional[int] = 30,
        breath_seconds: float = 3.6,
        breaths_per_color: int = 3,
        transition_seconds: float = 1.0,
        base_brightness: float = 0.10,
        peak_brightness: float = 1.00,
        outer_emphasis: float = 0.28,
        # Finale controls
        finale_seconds: float = 4.0,
        finale_speed_multiplier: float = 2.2,
        finale_hue_speed: float = 0.16,
        sparkle_probability: float = 0.06,
        sparkle_boost: float = 0.35,
    ):
        super().__init__(frameBuf, fps=fps)

        # ---- Parameters
        self.fps = fps if fps is not None else 30
        self.breath_seconds = float(breath_seconds)
        self.breaths_per_color = int(breaths_per_color)
        self.transition_seconds = float(transition_seconds)

        self.base_brightness = float(base_brightness)
        self.peak_brightness = float(peak_brightness)
        self.outer_emphasis = float(outer_emphasis)

        self.finale_seconds = float(finale_seconds)
        self.finale_speed_multiplier = float(finale_speed_multiplier)
        self.finale_hue_speed = float(finale_hue_speed)
        self.sparkle_probability = float(sparkle_probability)
        self.sparkle_boost = float(sparkle_boost)

        # ---- Precompute point normalization (robust to real-tree imperfections)
        pts = POINTS_3D.astype(np.float32)
        min_pt = pts.min(axis=0)
        max_pt = pts.max(axis=0)
        mid = (min_pt + max_pt) * 0.5
        centered = pts - mid

        x = centered[:, 0]
        y = centered[:, 1]
        z = centered[:, 2]

        # "Radial" distance from the trunk axis (x,z), normalized
        r = np.sqrt(x * x + z * z)
        r_max = float(np.max(r)) if float(np.max(r)) > 1e-6 else 1.0
        self.rn = (r / r_max).astype(np.float32)

        # Height normalized 0..1 (for gentle spatial variation in finale)
        y_min, y_max = float(np.min(y)), float(np.max(y))
        denom = (y_max - y_min) if (y_max - y_min) > 1e-6 else 1.0
        self.yn = ((y - y_min) / denom).astype(np.float32)

        # Outer emphasis multiplier (kept subtle; doesn’t depend on exact geometry)
        self.outer_weight = (1.0 + self.outer_emphasis * self.rn).astype(np.float32)

        # ---- Palette (dominant per stage)
        # Tuned for camera: saturated, distinct, not pastel.
        self.palette = [
            np.array([25, 55, 255], dtype=np.float32),   # Deep Blue
            np.array([0, 210, 190], dtype=np.float32),   # Teal
            np.array([0, 255, 90], dtype=np.float32),    # Green
            np.array([255, 195, 25], dtype=np.float32),  # Gold
        ]

        # State machine
        self.frame = 0
        self.prev_phase = 0.0

        self.breath_count = 0
        self.color_index = 0
        self.current_color = self.palette[self.color_index].copy()

        self.is_transitioning = False
        self.transition_t = 0.0
        self.transition_from = self.current_color.copy()
        self.transition_to = self.current_color.copy()

        self.is_finale = False
        self.finale_start_frame = None

        # When to enter finale:
        self.total_breaths_before_finale = self.breaths_per_color * len(self.palette)

    @classmethod
    def validate_parameters(cls, parameters: Dict[str, Any]):
        super().validate_parameters(parameters)

        def _req_float(name: str, lo: float, hi: float):
            if name in parameters:
                v = float(parameters[name])
                if not (lo <= v <= hi):
                    raise TypeError(f"{name} must be between {lo} and {hi}")

        def _req_int(name: str, lo: int, hi: int):
            if name in parameters:
                v = int(parameters[name])
                if not (lo <= v <= hi):
                    raise TypeError(f"{name} must be between {lo} and {hi}")

        _req_float("breath_seconds", 1.2, 12.0)
        _req_int("breaths_per_color", 1, 12)
        _req_float("transition_seconds", 0.0, 4.0)

        _req_float("base_brightness", 0.0, 0.6)
        _req_float("peak_brightness", 0.3, 1.5)
        _req_float("outer_emphasis", 0.0, 1.0)

        _req_float("finale_seconds", 0.0, 12.0)
        _req_float("finale_speed_multiplier", 1.0, 6.0)
        _req_float("finale_hue_speed", 0.02, 0.8)
        _req_float("sparkle_probability", 0.0, 0.25)
        _req_float("sparkle_boost", 0.0, 1.0)

    def _breath_envelope(self, phase: float) -> float:
        """
        Smooth 0..1 envelope:
        - phase 0: minimum (bottom of exhale)
        - phase 0.5: maximum (peak inhale)
        - phase 1: wraps back to minimum
        """
        # 0.5*(1 - cos(2πp)) yields 0 at p=0, 1 at p=0.5, 0 at p=1
        return 0.5 * (1.0 - np.cos(2.0 * np.pi * phase))

    def _tick_breath_boundary(self, phase: float):
        """
        Detect phase wrap (near 1 -> 0), which is the bottom of exhale.
        We use wrap detection rather than exact thresholds for stability.
        """
        wrapped = phase < self.prev_phase
        self.prev_phase = phase
        if not wrapped:
            return

        # Breath boundary (bottom of exhale)
        self.breath_count += 1

        if self.is_finale:
            return

        # Enter finale after completing full palette cycle
        if self.breath_count >= self.total_breaths_before_finale:
            self.is_finale = True
            self.finale_start_frame = self.frame
            self.is_transitioning = False
            return

        # Advance color only on boundary and only every N breaths
        if (self.breath_count % self.breaths_per_color) == 0:
            next_index = min(self.color_index + 1, len(self.palette) - 1)
            if next_index != self.color_index:
                self.color_index = next_index
                self.transition_from = self.current_color.copy()
                self.transition_to = self.palette[self.color_index].copy()
                self.transition_t = 0.0
                self.is_transitioning = self.transition_seconds > 1e-6

                if not self.is_transitioning:
                    self.current_color = self.transition_to.copy()

    def _apply_transition(self, dt: float):
        if not self.is_transitioning:
            return

        self.transition_t += dt
        t = float(self.transition_t / max(self.transition_seconds, 1e-6))
        if t >= 1.0:
            self.current_color = self.transition_to.copy()
            self.is_transitioning = False
        else:
            self.current_color = _lerp(self.transition_from, self.transition_to, t)

    def _sparkle_mask(self, frame_idx: int) -> np.ndarray:
        """
        Deterministic sparkle mask (no RNG state), stable across runs.
        """
        if self.sparkle_probability <= 0.0:
            return np.zeros(len(self.frameBuf), dtype=bool)

        i = np.arange(len(self.frameBuf), dtype=np.uint32)
        f = np.uint32(frame_idx)
        # Simple hash mix
        h = (i * np.uint32(73856093)) ^ (f * np.uint32(19349663)) ^ (i << np.uint32(16))
        # Convert to [0,1)
        u = (h.astype(np.uint32) / np.float32(2**32))
        return u < np.float32(self.sparkle_probability)

    def renderNextFrame(self):
        # Time
        t = self.frame / float(self.fps)

        # Breath phase
        if not self.is_finale:
            breath_period = self.breath_seconds
        else:
            breath_period = max(self.breath_seconds / max(self.finale_speed_multiplier, 1.0), 0.6)

        phase = (t / breath_period) % 1.0
        self._tick_breath_boundary(phase)

        # Update color transition timing
        dt = 1.0 / float(self.fps)
        if not self.is_finale:
            self._apply_transition(dt)

        # Breath envelope and brightness
        env = float(self._breath_envelope(phase))
        brightness = self.base_brightness + (self.peak_brightness - self.base_brightness) * env
        brightness = float(np.clip(brightness, 0.0, 1.5))

        # Clear to black background
        self.frameBuf[:] = 0

        if not self.is_finale:
            # Solid dominant color, whole tree, with a subtle outer emphasis.
            color = self.current_color  # float32 RGB
            rgb = (color[None, :] * (brightness * self.outer_weight)[:, None]).astype(np.float32)

            # Clip and write
            self.frameBuf[:] = np.clip(rgb, 0.0, 255.0).astype(np.uint8)

        else:
            # Finale duration (optional stop growing; stays in finale mode and cycles)
            # If someone wants it to "end", they can Ctrl+C; in real tree it loops anyway.
            # We keep the finale behavior stable after its initial window.
            if self.finale_start_frame is None:
                self.finale_start_frame = self.frame

            # Multicolor breathing: hue field varies gently with height and radius
            # so it reads as "alive" without requiring perfect geometry.
            hue = (t * self.finale_hue_speed) + (0.18 * self.yn) + (0.12 * self.rn)
            hue = np.mod(hue, 1.0)

            # Brightness still uses breathing envelope (fast), plus outer emphasis
            v = _clamp01((brightness * self.outer_weight).astype(np.float32))

            # Sparkle: brief bright hits to add camera-visible life
            sparkle = self._sparkle_mask(self.frame)
            if np.any(sparkle) and self.sparkle_boost > 0.0:
                v[sparkle] = _clamp01(v[sparkle] + np.float32(self.sparkle_boost))

            # Convert HSV -> RGB per pixel
            # (Vectorizing with custom hsv_to_rgb isn’t guaranteed; do a tight loop over 500 pixels)
            out = self.frameBuf
            for i in range(len(out)):
                r, g, b = hsv_to_rgb(float(hue[i]), 1.0, float(v[i]))
                out[i, 0] = r
                out[i, 1] = g
                out[i, 2] = b

        self.frame += 1

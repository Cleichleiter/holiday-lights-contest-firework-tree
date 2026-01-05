# animation.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np

from lib.base_animation import BaseAnimation
from lib.constants import NUM_PIXELS
from utils.geometry import POINTS_3D


def _clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _wrap01(d: np.ndarray) -> np.ndarray:
    """Distance on a unit circle [0,1) with wrap-around."""
    d = np.abs(d)
    return np.minimum(d, 1.0 - d)


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = _clamp01((x - edge0) / (edge1 - edge0 + 1e-9))
    return t * t * (3.0 - 2.0 * t)


def _gamma_encode(rgb01: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    return np.power(_clamp01(rgb01), 1.0 / gamma)


def _rgb_u8(rgb01: np.ndarray) -> np.ndarray:
    return (np.clip(rgb01 * 255.0, 0.0, 255.0)).astype(np.uint8)


def _color_u8(r: int, g: int, b: int) -> np.ndarray:
    return np.array([r, g, b], dtype=np.uint8)


def _color01_from_u8(c: np.ndarray) -> np.ndarray:
    return c.astype(np.float32) / 255.0


@dataclass(frozen=True)
class ShowBeats:
    intro: int
    verse1: int
    prechorus1: int
    chorus1: int
    verse2: int
    chorus2: int
    bridge: int
    build: int
    final_chorus: int
    outro: int

    @property
    def total(self) -> int:
        return (
            self.intro
            + self.verse1
            + self.prechorus1
            + self.chorus1
            + self.verse2
            + self.chorus2
            + self.bridge
            + self.build
            + self.final_chorus
            + self.outro
        )


class LastChristmasShow(BaseAnimation):
    """
    Last Christmas (Wham!) inspired light show, designed for a real tree.

    Narrative (as requested):
      - Intro/Verse1/Pre-chorus: gold ring + gold rain
      - Chorus 1 starts: gold ring dissolves AND gold rain stops permanently
      - Chorus sections: red/white spiral bands (white pulses on beats)
      - Verse 2: dim bands + twinkling white
      - Bridge: cool white sparse twinkle (no red)
      - Build: red returns gradually with soft beat pulses
      - Final chorus: full red/white spiral
      - Outro: fast multicolor blinking finish

    Notes:
      - Timing is beat-driven (BPM) using beat counts per section.
      - You can tune bpm and section beat counts via args.
    """

    def __init__(
        self,
        frameBuf: np.ndarray,
        *,
        fps: Optional[int] = 30,
        bpm: float = 108.0,
        # Section beat counts (tunable)
        intro_beats: int = 8,
        verse1_beats: int = 32,
        prechorus1_beats: int = 16,
        chorus1_beats: int = 32,
        verse2_beats: int = 48,
        chorus2_beats: int = 32,
        bridge_beats: int = 32,
        build_beats: int = 16,
        final_chorus_beats: int = 32,
        outro_beats: int = 16,
        # Geometry/visual tuning
        spiral_turns: float = 3.0,
        band_width: float = 0.10,  # width in spiral phase units (0..1), thicker = more real-tree friendly
        band_softness: float = 0.12,  # soft edge feather
        spiral_speed: float = 0.20,  # cycles per second-ish (phase increment)
        # Gold phase
        ring_top_percent: float = 0.08,  # top band thickness as percent of tree height (0..1)
        ring_brightness: float = 1.00,
        rain_speed: float = 0.95,  # how fast "rain energy" advects downward (rank shift per second)
        rain_inject_strength: float = 0.85,
        rain_decay: float = 0.86,  # per-second-ish decay
        # Twinkle
        twinkle_rate: float = 0.07,  # probability per pixel per second (scaled)
        twinkle_decay: float = 3.0,  # larger = faster fade
        twinkle_max_active: int = 65,
        # Beat accents
        white_pulse_strength: float = 0.90,
        white_pulse_width_beats: float = 0.22,  # pulse width as fraction of a beat
        # Outro multicolor
        outro_blink_subdivision: int = 2,  # 2 => half-beat toggles, 4 => quarter-beat
        outro_brightness: float = 1.15,
        # Global brightness trim
        master_brightness: float = 1.00,
        # Background
        background_u8: tuple = (0, 0, 0),
    ):
        super().__init__(frameBuf, fps=fps)

        self.t = 0.0  # seconds
        self.fps_eff = float(fps if fps is not None and fps > 0 else 60.0)

        self.bpm = float(bpm)
        self.beat_sec = 60.0 / max(1e-6, self.bpm)

        self.beats = ShowBeats(
            intro=int(intro_beats),
            verse1=int(verse1_beats),
            prechorus1=int(prechorus1_beats),
            chorus1=int(chorus1_beats),
            verse2=int(verse2_beats),
            chorus2=int(chorus2_beats),
            bridge=int(bridge_beats),
            build=int(build_beats),
            final_chorus=int(final_chorus_beats),
            outro=int(outro_beats),
        )

        # Visual params
        self.spiral_turns = float(spiral_turns)
        self.band_width = float(band_width)
        self.band_softness = float(band_softness)
        self.spiral_speed = float(spiral_speed)

        self.ring_top_percent = float(ring_top_percent)
        self.ring_brightness = float(ring_brightness)

        self.rain_speed = float(rain_speed)
        self.rain_inject_strength = float(rain_inject_strength)
        self.rain_decay = float(rain_decay)

        self.twinkle_rate = float(twinkle_rate)
        self.twinkle_decay = float(twinkle_decay)
        self.twinkle_max_active = int(twinkle_max_active)

        self.white_pulse_strength = float(white_pulse_strength)
        self.white_pulse_width_beats = float(white_pulse_width_beats)

        self.outro_blink_subdivision = int(outro_blink_subdivision)
        self.outro_brightness = float(outro_brightness)

        self.master_brightness = float(master_brightness)
        self.bg01 = _color01_from_u8(np.array(background_u8, dtype=np.uint8))

        # Colors (u8 -> 0..1)
        self.gold01 = _color01_from_u8(_color_u8(255, 185, 60))     # warm gold
        self.red01 = _color01_from_u8(_color_u8(255, 0, 0))
        self.white01 = _color01_from_u8(_color_u8(255, 255, 255))
        self.cool_white01 = _color01_from_u8(_color_u8(210, 225, 255))  # slightly cool

        # Precompute centered geometry & derived coords
        pts = np.asarray(POINTS_3D, dtype=np.float32)
        min_pt = np.min(pts, axis=0)
        max_pt = np.max(pts, axis=0)
        mid = (min_pt + max_pt) / 2.0
        self.P = pts - mid

        # Normalize Z to [0,1]
        z = self.P[:, 2]
        zmin, zmax = float(np.min(z)), float(np.max(z))
        self.z01 = (z - zmin) / (zmax - zmin + 1e-9)

        # Angle around vertical axis in [0,1)
        ang = np.arctan2(self.P[:, 1], self.P[:, 0])  # [-pi, pi]
        self.a01 = (ang + math.pi) / (2.0 * math.pi)  # [0, 1)

        # For "rain" advection: sort indices by height (z01 ascending)
        self.z_order = np.argsort(self.z01)
        self.z_rank = np.empty(NUM_PIXELS, dtype=np.int32)
        self.z_rank[self.z_order] = np.arange(NUM_PIXELS, dtype=np.int32)

        # Determine top band for ring/rain injection
        self.ring_mask = self.z01 >= (1.0 - self.ring_top_percent)

        # Stateful buffers (float 0..1)
        self.rain_energy = np.zeros(NUM_PIXELS, dtype=np.float32)      # per-pixel gold energy
        self.twinkle_energy = np.zeros(NUM_PIXELS, dtype=np.float32)   # per-pixel twinkle energy

        # Outro palette (stable per pixel for a nicer finish)
        palette = np.array(
            [
                [255, 0, 0],
                [0, 255, 0],
                [0, 120, 255],
                [255, 180, 50],
                [255, 255, 255],
                [255, 60, 180],
            ],
            dtype=np.uint8,
        )
        rng = np.random.default_rng(12345)
        self.outro_palette01 = _color01_from_u8(palette[rng.integers(0, len(palette), size=NUM_PIXELS)])

    # ---- Parameter plumbing expected by the contest runner ----

    @classmethod
    def get_default_parameters(cls) -> Dict[str, Any]:
        return {
            "fps": 30,
            "bpm": 108.0,
            "intro_beats": 8,
            "verse1_beats": 32,
            "prechorus1_beats": 16,
            "chorus1_beats": 32,
            "verse2_beats": 48,
            "chorus2_beats": 32,
            "bridge_beats": 32,
            "build_beats": 16,
            "final_chorus_beats": 32,
            "outro_beats": 16,
            "spiral_turns": 3.0,
            "band_width": 0.10,
            "band_softness": 0.12,
            "spiral_speed": 0.20,
            "ring_top_percent": 0.08,
            "ring_brightness": 1.00,
            "rain_speed": 0.95,
            "rain_inject_strength": 0.85,
            "rain_decay": 0.86,
            "twinkle_rate": 0.07,
            "twinkle_decay": 3.0,
            "twinkle_max_active": 65,
            "white_pulse_strength": 0.90,
            "white_pulse_width_beats": 0.22,
            "outro_blink_subdivision": 2,
            "outro_brightness": 1.15,
            "master_brightness": 1.00,
            "background_u8": (0, 0, 0),
        }

    @classmethod
    def validate_parameters(cls, parameters):
        super().validate_parameters(parameters)
        p = {**cls.get_default_parameters(), **(parameters or {})}

        if p["fps"] is not None:
            if not isinstance(p["fps"], int) or p["fps"] <= 0 or p["fps"] > 240:
                raise TypeError("fps must be an int in 1..240 or None")

        if not (40.0 <= float(p["bpm"]) <= 200.0):
            raise TypeError("bpm must be between 40 and 200")

        for k in [
            "intro_beats",
            "verse1_beats",
            "prechorus1_beats",
            "chorus1_beats",
            "verse2_beats",
            "chorus2_beats",
            "bridge_beats",
            "build_beats",
            "final_chorus_beats",
            "outro_beats",
        ]:
            if int(p[k]) <= 0:
                raise TypeError(f"{k} must be a positive integer")

        if not (0.5 <= float(p["spiral_turns"]) <= 8.0):
            raise TypeError("spiral_turns must be between 0.5 and 8.0")
        if not (0.03 <= float(p["band_width"]) <= 0.35):
            raise TypeError("band_width must be between 0.03 and 0.35")
        if not (0.02 <= float(p["band_softness"]) <= 0.35):
            raise TypeError("band_softness must be between 0.02 and 0.35")
        if not (0.01 <= float(p["ring_top_percent"]) <= 0.25):
            raise TypeError("ring_top_percent must be between 0.01 and 0.25")
        if not (0.0 <= float(p["master_brightness"]) <= 3.0):
            raise TypeError("master_brightness must be between 0 and 3")

        bg = p.get("background_u8")
        if (
            not isinstance(bg, (tuple, list))
            or len(bg) != 3
            or any((not isinstance(v, int) or v < 0 or v > 255) for v in bg)
        ):
            raise TypeError("background_u8 must be a 3-tuple/list of ints in 0..255")

    # ---- Timing helpers ----

    def _beat_pos(self) -> float:
        """Current beat position (can be fractional)."""
        return self.t / self.beat_sec

    def _section_and_local(self, beat_pos: float):
        """Return (section_name, local_beat, section_index)."""
        b = beat_pos
        edges = []
        cur = 0
        for name, n in [
            ("intro", self.beats.intro),
            ("verse1", self.beats.verse1),
            ("prechorus1", self.beats.prechorus1),
            ("chorus1", self.beats.chorus1),
            ("verse2", self.beats.verse2),
            ("chorus2", self.beats.chorus2),
            ("bridge", self.beats.bridge),
            ("build", self.beats.build),
            ("final_chorus", self.beats.final_chorus),
            ("outro", self.beats.outro),
        ]:
            edges.append((name, cur, cur + n))
            cur += n

        for idx, (name, s, e) in enumerate(edges):
            if b < e:
                return name, (b - s), idx

        return "outro", (b - edges[-1][1]), len(edges) - 1

    def _pulse(self, local_beat: float) -> float:
        """
        Beat pulse: returns 0..1 with a tight bump at each beat.
        local_beat is beat position within the current section.
        """
        frac = local_beat - math.floor(local_beat)
        width = max(0.01, float(self.white_pulse_width_beats))
        # center pulse at start of beat (frac=0)
        d = min(frac, 1.0 - frac)
        # Convert desired width (fraction of beat) into a smooth bump
        x = 1.0 - (d / (width + 1e-6))
        return float(_smoothstep(0.0, 1.0, np.array([x], dtype=np.float32))[0])

    # ---- Visual primitives ----

    def _spiral_band_mask(self, phase: float) -> np.ndarray:
        """
        Creates a soft mask for a spiral band.
        phase is in [0,1) and moves the band around the tree.
        """
        # spiral coordinate (0..1)
        s = (self.a01 + self.spiral_turns * self.z01 + phase) % 1.0
        # distance to band center at 0.0 (wrap)
        d = _wrap01(s)
        # band core and soft edge
        core = self.band_width * 0.5
        feather = self.band_softness
        # mask: 1 inside core, fades to 0 by core+feather
        return 1.0 - _smoothstep(core, core + feather, d)

    def _update_rain(self, dt: float, strength: float):
        """
        Gold rain implemented as energy advection down the tree using z-rank shifting.
        This is robust for uneven physical layouts.
        """
        if strength <= 0.0:
            # hard stop: fully zero
            self.rain_energy[:] = 0.0
            return

        # Decay
        decay = math.exp(-max(0.0, self.rain_decay) * dt)
        self.rain_energy *= decay

        # Advect by shifting along z order
        shift = int(max(0.0, self.rain_speed) * dt * 60.0)  # scaled to feel good at 30–60fps
        if shift > 0:
            ordered = self.rain_energy[self.z_order]
            ordered = np.roll(ordered, shift)
            ordered[:shift] = 0.0
            self.rain_energy[self.z_order] = ordered

        # Inject at top band
        inject_mask = self.ring_mask
        # Random injection with cap (avoid noisy saturation)
        rng = np.random.default_rng(int(self.t * 1000) ^ 0xA5A5)
        p = 0.20 * dt * 60.0  # base spawn probability, scaled with dt
        spawn = rng.random(NUM_PIXELS) < p
        spawn &= inject_mask
        self.rain_energy[spawn] = np.maximum(
            self.rain_energy[spawn],
            strength * self.rain_inject_strength * (0.6 + 0.4 * rng.random(np.count_nonzero(spawn))),
        )

    def _update_twinkle(self, dt: float, enabled: bool, intensity: float, cool: bool = False):
        """
        White twinkle: sparse random sparks that decay smoothly.
        """
        # Decay always
        self.twinkle_energy *= math.exp(-max(0.1, self.twinkle_decay) * dt)

        if not enabled or intensity <= 0.0:
            return

        # Cap active twinkles
        active = int(np.count_nonzero(self.twinkle_energy > 0.12))
        room = max(0, self.twinkle_max_active - active)
        if room <= 0:
            return

        rng = np.random.default_rng((int(self.t * 1000) * 1103515245 + 12345) & 0xFFFFFFFF)
        # Spawn probability scaled
        p = float(self.twinkle_rate) * dt
        candidates = rng.random(NUM_PIXELS) < p
        idx = np.where(candidates)[0]
        if idx.size == 0:
            return

        # Limit to room
        if idx.size > room:
            idx = rng.choice(idx, size=room, replace=False)

        self.twinkle_energy[idx] = np.maximum(
            self.twinkle_energy[idx],
            intensity * (0.55 + 0.45 * rng.random(idx.size)),
        )

    # ---- Main render ----

    def renderNextFrame(self):
        dt = 1.0 / self.fps_eff
        self.t += dt

        beat_pos = self._beat_pos()
        section, local_beat, _ = self._section_and_local(beat_pos)

        # Base canvas
        rgb = np.tile(self.bg01, (NUM_PIXELS, 1)).astype(np.float32)

        # Phase driver for spirals
        phase = (self.t * self.spiral_speed) % 1.0

        # Helper: chorus spiral base
        def apply_red_white_spiral(intensity: float, pulse_boost: float):
            nonlocal rgb
            mask = self._spiral_band_mask(phase)

            # Red base (steady)
            red_layer = (mask[:, None] * self.red01[None, :]) * intensity
            rgb += red_layer

            # White beat pulses on strong beats
            pulse = self._pulse(local_beat)
            white_amt = mask * (pulse * pulse_boost)
            rgb += (white_amt[:, None] * self.white01[None, :])

        # ---- Section behaviors ----

        if section in ("intro", "verse1", "prechorus1"):
            # Gold ring + gold rain
            # Ring pulses slightly more as we approach chorus
            if section == "intro":
                ring_alpha = 0.25 + 0.75 * _smoothstep(0.0, 2.0, np.array([local_beat], np.float32))[0]
                rain_strength = 0.35
            elif section == "verse1":
                ring_alpha = 0.75
                rain_strength = 0.55
            else:  # prechorus1
                # build intensity
                ramp = float(_smoothstep(0.0, self.beats.prechorus1, np.array([local_beat], np.float32))[0])
                ring_alpha = 0.85 + 0.15 * ramp
                rain_strength = 0.70 + 0.20 * ramp

            # Ring
            rgb[self.ring_mask] += self.gold01 * (self.ring_brightness * ring_alpha)

            # Rain
            self._update_rain(dt, strength=rain_strength)
            rgb += (self.rain_energy[:, None] * self.gold01[None, :])

            # No twinkle
            self._update_twinkle(dt, enabled=False, intensity=0.0)

        elif section == "chorus1":
            # Chorus 1: gold dissolves + rain hard stops, red/white spiral begins
            # First beats: dissolve gold quickly
            dissolve = float(_smoothstep(0.0, 2.0, np.array([local_beat], np.float32))[0])
            gold_keep = 1.0 - dissolve

            # Ensure rain is permanently stopped from here forward
            self._update_rain(dt, strength=0.0)

            # Optional: tiny lingering gold in the first 1-2 beats (dissolve)
            if gold_keep > 0.0:
                rgb[self.ring_mask] += self.gold01 * (0.55 * gold_keep)
                # no falling rain, only ring residue

            # Spiral bands
            apply_red_white_spiral(intensity=0.95, pulse_boost=self.white_pulse_strength)

            # No twinkle during chorus
            self._update_twinkle(dt, enabled=False, intensity=0.0)

        elif section == "verse2":
            # Verse 2: dim/slow bands + soft white twinkle
            # Keep spiral but lower intensity
            apply_red_white_spiral(intensity=0.50, pulse_boost=0.35)

            # Twinkle white (pretty, sparse)
            self._update_twinkle(dt, enabled=True, intensity=0.75, cool=False)
            rgb += (self.twinkle_energy[:, None] * self.white01[None, :]) * 0.55

            # Ensure rain stays off forever
            self._update_rain(dt, strength=0.0)

        elif section == "chorus2":
            # Chorus 2: stronger than chorus 1, no twinkle
            apply_red_white_spiral(intensity=1.05, pulse_boost=self.white_pulse_strength * 1.05)
            self._update_twinkle(dt, enabled=False, intensity=0.0)
            self._update_rain(dt, strength=0.0)

        elif section == "bridge":
            # Bridge: cool white sparse twinkle only; red fades out
            self._update_rain(dt, strength=0.0)

            # Very sparse cool twinkle
            self._update_twinkle(dt, enabled=True, intensity=0.70, cool=True)
            rgb += (self.twinkle_energy[:, None] * self.cool_white01[None, :]) * 0.45

        elif section == "build":
            # Build: red returns gradually + soft pulses
            ramp = float(_smoothstep(0.0, self.beats.build, np.array([local_beat], np.float32))[0])
            # Increase intensity over build
            apply_red_white_spiral(intensity=0.45 + 0.55 * ramp, pulse_boost=0.40 + 0.35 * ramp)
            # No twinkle
            self._update_twinkle(dt, enabled=False, intensity=0.0)
            self._update_rain(dt, strength=0.0)

        elif section == "final_chorus":
            # Final chorus: full intensity
            apply_red_white_spiral(intensity=1.15, pulse_boost=self.white_pulse_strength * 1.10)
            self._update_twinkle(dt, enabled=False, intensity=0.0)
            self._update_rain(dt, strength=0.0)

        else:
            # Outro: fast multicolor blinking finish (no gold, no rain)
            self._update_rain(dt, strength=0.0)
            self._update_twinkle(dt, enabled=False, intensity=0.0)

            subdiv = max(1, int(self.outro_blink_subdivision))
            # Determine blink state by beat subdivisions
            beat = beat_pos
            step = math.floor(beat * subdiv)
            on = (step % 2) == 0

            if on:
                rgb += self.outro_palette01 * self.outro_brightness
            else:
                # Off state still slightly visible to avoid harsh video clipping
                rgb += self.outro_palette01 * (0.18 * self.outro_brightness)

        # Master brightness and output
        rgb *= self.master_brightness

        # Keep within bounds, gamma encode for nicer perceived brightness
        rgb = _gamma_encode(rgb, gamma=2.0)
        self.frameBuf[:] = _rgb_u8(rgb)

Last Christmas – Beat-Structured Christmas Tree Animation

This animation was designed for the Tarabyte Holiday Lights Contest as a visually musical light show inspired by Wham!’s “Last Christmas”.
Rather than attempting real-time audio processing, the animation is beat-structured, allowing it to stay tightly synchronized to the song when started together.

The result is a smooth, expressive, and performance-safe animation that reads clearly on a 3D tree while remaining compatible with the contest framework.

Design Philosophy

This animation is built around three core goals:

Musical Readability
Movements, pulses, and transitions are aligned to musical beats rather than raw frame timing. This produces motion that feels intentional, lyrical, and synchronized—even without live audio input.

3D Awareness
All effects leverage the tree’s actual geometry:

vertical height (z)

angular position around the trunk

smooth interpolation across spatial bands

This avoids flat “ring” or “stacked” looks and instead creates motion that wraps, falls, and spins through the tree volume.

Visual Progression
The show evolves across distinct musical sections, mirroring the emotional arc of Last Christmas:

warm and restrained → bold and festive → airy and delicate → celebratory finale

Musical Structure (108 BPM)

The animation assumes 108 BPM, which closely matches Last Christmas.

Beats	Section	Description
0–24	Intro / Verse	Gold star ring with shimmering golden rain
24–56	Chorus	Red & white spiral with smooth beat pulsing
56–80	Break	White twinkles with gentle golden haze
80+	Finale	Fast multicolor chase, sparkles, and gold crown

Soft crossfades are applied between sections to avoid abrupt visual jumps.

Visual Effects Overview
Intro – Gold Ring & Shimmering Rain

Stable gold “star” ring at the top of the tree

Multiple descending golden shimmer bands

Subtle angular variation so rain feels dimensional, not cylindrical

Chorus – Red & White Spiral

Continuous spiral using angle + height

Beat-synced brightness pulse

Smooth red/white color swapping for a candy-cane feel

Gentle glow to preserve readability at distance

Twinkle Break

Deterministic (non-random) twinkles for visual stability

White sparkles fade in and out smoothly

Soft golden haze drifting upward to prevent emptiness

Finale – Multicolor Celebration

Rotating multicolor chase through the tree

Beat-synced strobe that never fully blacks out

Increasing sparkle density as the song closes

Persistent gold crown at the top as a visual signature

Performance & Optimization Notes

This animation is optimized for smooth playback on constrained hardware:

No per-frame random allocations

Deterministic sparkle generation (hash-based)

Precomputed geometry attributes

Vectorized NumPy operations throughout

Gamma-corrected output for brighter LED appearance

The animation is safe to run at 30–60 FPS.

Optional Audio Sync (Local WAV File)

While the contest framework does not currently support direct audio playback or FFT analysis, you can locally sync this animation to the song by starting both together.

Recommended Setup

Obtain a local WAV file of Wham! – Last Christmas

Open the WAV in a media player

Start playback

Immediately run the animation

Because the animation is beat-based rather than time-coded, minor start offsets are not noticeable once the chorus begins.

BPM Reference

Song BPM: ~108

Animation default BPM: 108

If your version of the song differs slightly, you can fine-tune synchronization using the bpm parameter.

Running the Animation
python run_animation.py --show-tree --background black --args "{\"fps\": 30, \"bpm\": 108}"

Useful Parameters
Parameter	Description
fps	Frames per second (30 recommended)
bpm	Beats per minute (adjust for sync)
turns	Number of spiral wraps
brightness	Overall brightness multiplier
gamma	Gamma correction for LED output
twinkle_density	Sparkle density during twinkle & finale
Notes for Judges & Reviewers

This animation is deterministic and repeatable

Designed for real tree geometry, not a flat strip

Emphasizes clarity, musicality, and progression over raw brightness

Safe for long runtimes without visual fatigue

Final Thought

This piece was intentionally designed to feel like a coordinated Christmas light show, not a visualization experiment.
It favors rhythm, pacing, and spatial motion—qualities that translate well both on a simulator and on a physical tree.
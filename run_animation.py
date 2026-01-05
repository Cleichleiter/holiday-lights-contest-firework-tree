# run_animation.py
from __future__ import annotations

import argparse
import importlib
import inspect
import json
from typing import Any, Dict, Optional

import numpy as np

from lib.constants import NUM_PIXELS
from lib.matplotlib_controller import MatplotlibController


def _parse_args_json(raw: Optional[str]) -> Dict[str, Any]:
    if raw is None:
        return {}
    raw = raw.strip()
    if raw == "":
        return {}
    val = json.loads(raw)
    if val is None:
        return {}
    if not isinstance(val, dict):
        raise TypeError("--args JSON must decode to an object/dict, e.g. '{\"fps\": 30}'")
    return val


def _load_animation_class(sample: Optional[str]):
    if sample:
        mod = importlib.import_module(f"samples.{sample}")
    else:
        mod = importlib.import_module("animation")

    anim_cls = getattr(mod, "Animation", None)
    if anim_cls is None:
        raise RuntimeError(
            f"Could not find class named 'Animation' in {'samples.' + sample if sample else 'animation.py'}"
        )
    return anim_cls


def _controller_kwargs(
    *,
    controller_sig: inspect.Signature,
    animation_kwargs: Dict[str, Any],
    n_pixels: int,
    background: str,
    show_tree: bool,
):
    params = set(controller_sig.parameters.keys())
    params.discard("self")

    kwargs: Dict[str, Any] = {}

    # Common names across repo revisions
    if "animation_kwargs" in params:
        kwargs["animation_kwargs"] = animation_kwargs
    if "n_pixels" in params:
        kwargs["n_pixels"] = n_pixels
    if "background" in params:
        kwargs["background"] = background
    if "bg" in params and "background" not in kwargs:
        kwargs["bg"] = background
    if "show_tree" in params:
        kwargs["show_tree"] = show_tree

    return kwargs


def _run_controller(controller):
    # Run method name varies across revisions
    if hasattr(controller, "run") and callable(controller.run):
        controller.run()
        return
    if hasattr(controller, "start") and callable(controller.start):
        controller.start()
        return
    if hasattr(controller, "start_animation") and callable(controller.start_animation):
        controller.start_animation()
        return
    raise RuntimeError("Controller has no run()/start()/start_animation() method.")


def main():
    parser = argparse.ArgumentParser(description="Script for running animations")
    parser.add_argument("--args", dest="args_json", default=None, help="Animation args as JSON")
    parser.add_argument("--no_validation", action="store_true", help="Skip validate_parameters")
    parser.add_argument("--sample", type=str, default=None, help="Run a sample from samples/<name>.py")
    parser.add_argument("--list-samples", action="store_true", help="List available samples")
    parser.add_argument("--background", type=str, default="black", help="Background color for visualization")
    parser.add_argument("--show-tree", action="store_true", help="Show the tree mesh in the plot")
    args = parser.parse_args()

    if args.list_samples:
        try:
            pkg = importlib.import_module("samples")
            pkg_path = pkg.__path__[0]
            import os

            files = sorted(
                f[:-3]
                for f in os.listdir(pkg_path)
                if f.endswith(".py") and f not in ("__init__.py",)
            )
            print("Available samples:")
            for f in files:
                print(" -", f)
        except Exception as e:
            print(f"Could not list samples: {e}")
        return

    try:
        animation_kwargs = _parse_args_json(args.args_json)
    except Exception as e:
        print(f"Error loading animation: {e}")
        return

    try:
        AnimClass = _load_animation_class(args.sample)
    except Exception as e:
        print(f"Error loading animation: {e}")
        return

    # Validate parameters if present and not disabled
    if not args.no_validation:
        validate = getattr(AnimClass, "validate_parameters", None)
        if callable(validate):
            try:
                validate(animation_kwargs)
            except Exception as e:
                print(f"Error loading animation: {e}")
                return

    # Build controller kwargs based on signature
    sig = inspect.signature(MatplotlibController.__init__)
    kwargs = _controller_kwargs(
        controller_sig=sig,
        animation_kwargs=animation_kwargs,
        n_pixels=NUM_PIXELS,
        background=args.background,
        show_tree=args.show_tree,
    )

    # Two possible controller styles exist:
    #   A) MatplotlibController(animation_instance, ...)
    #   B) MatplotlibController(AnimationClass, ...)  # controller calls the class internally
    #
    # We attempt A first; if we see "'Animation' object is not callable", we retry using B.
    frame_buf = np.zeros((NUM_PIXELS, 3), dtype=np.uint8)

    # Attempt A: pass instantiated animation
    try:
        anim = AnimClass(frame_buf, **animation_kwargs)
        controller = MatplotlibController(anim, **kwargs)
        _run_controller(controller)
        return
    except Exception as e:
        msg = str(e)
        if "object is not callable" not in msg:
            print(f"Error loading animation: {e}")
            return

    # Attempt B: pass animation class (callable)
    try:
        controller = MatplotlibController(AnimClass, **kwargs)
        _run_controller(controller)
        return
    except Exception as e:
        print(f"Error loading animation: {e}")
        return


if __name__ == "__main__":
    main()

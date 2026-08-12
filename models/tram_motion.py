"""modallabs.models.tram_motion -- registration shim for the TRAM motion-capture project.

The trainer itself lives in its own repo (~/tram-motion/lane) because it owns job specs,
weight staging, the TRAM command line and a solve accept/reject pass that have no business
in the harness. This module exists only so `import modallabs.models` fires the
@register("tram_motion") decorator, the same way every built-in trainer does.

Install the project into the same environment to activate the lane:

    uv pip install -e ~/tram-motion/lane

Without it, models/__init__._safe_import downgrades this to a clean "skipped" and the
rest of the registry is unaffected.

ASCII only. No emojis.
"""
from __future__ import annotations

from trammotion.trainer import TramMotionTrainer  # noqa: F401 -- fires @register

__all__ = ["TramMotionTrainer"]

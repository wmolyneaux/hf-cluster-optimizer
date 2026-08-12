"""modallabs.models.longcat_avatar -- registration shim for the LongCatAvatar project.

The trainer itself lives in its own repo (~/LongCatAvatar) because it owns job specs,
asset staging, weight-set definitions and a render accept/reject pass that have no
business in the harness. This module exists only so `import modallabs.models` fires the
@register("longcat_avatar") decorator, the same way every built-in trainer does.

Install the project into the same environment to activate the lane:

    uv pip install -e ~/LongCatAvatar

Without it, models/__init__._safe_import downgrades this to a clean "skipped" and the
rest of the registry is unaffected.

ASCII only. No emojis.
"""
from __future__ import annotations

from longcatavatar.trainer import LongCatAvatarTrainer  # noqa: F401 -- fires @register

__all__ = ["LongCatAvatarTrainer"]

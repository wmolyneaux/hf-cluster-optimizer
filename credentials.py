"""modallabs — credential injection points. Names and shapes only, never values.

One place that knows what each Modal Secret is called and which keys it carries, so the
launcher (`modal_app.py`) and the lanes that consume the keys cannot drift apart. Mirrors
the convention `tram-motion/lane/trammotion/config.py` established for `smpl-credentials`:
a named constant per secret, the exact `modal secret create` line to recreate it, and a
consumption-side check that fails loud with that line rather than proceeding
unauthenticated.

Deliberately a stdlib-only leaf, like `hostmem`: it imports nothing from modallabs and
nothing from modal, so a lane can consult it inside a container without dragging the
launcher in.

Named `credentials.py`, not `secrets.py`, on purpose. This package uses a flat layout --
the repo root IS the `modallabs` package dir -- so a module called `secrets.py` here can
shadow the stdlib `secrets` module for anything that lands the root on sys.path.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

__all__ = [
    "HF_SECRET_NAME", "HF_SECRET_KEYS", "hf_token", "require_hf_token",
    "create_hint",
]

# THE HuggingFace credential injection point. Used for gated model pulls and for
# `push_to_hub` from a Trainer.
#
# These are NOT invented names. The Secret `huggingface-secret` already exists in the
# wmolyneauxv workspace (created 2026-07-30 18:45), and its key names were read back on
# 2026-08-11 from a short CPU container that printed KEY NAMES ONLY, never values -- the
# same technique used to pin SMPL_SECRET_KEYS. It carries exactly one key:
#
#   modal secret create huggingface-secret HF_TOKEN=hf_...
#
# Token scope matters and is not visible from the key name. As read on 2026-08-11 the
# stored token belongs to HF user `oznaru` (orgs: Berkeley) with **role "read"**, so it
# can pull gated weights but CANNOT create or write a repo -- any `push_to_hub` will 403
# no matter which namespace is requested. Replacing it needs a write-scoped token from
# https://huggingface.co/settings/tokens.
HF_SECRET_NAME = "huggingface-secret"
HF_SECRET_KEYS: Sequence[str] = ("HF_TOKEN",)

# The hub client's own env var, accepted on read so a locally-exported token works too.
# Not part of HF_SECRET_KEYS: the Secret carries HF_TOKEN, and that is what gets created.
_HF_FALLBACK_ENV = "HUGGING_FACE_HUB_TOKEN"


def create_hint(name: str, keys: Sequence[str]) -> str:
    """The exact command that repairs a missing/incomplete secret."""
    return f"  modal secret create {name} " + " ".join(f"{k}=<value>" for k in keys)


def hf_token() -> Optional[str]:
    """The HF token if present, else None. For paths where anonymous access is fine --
    pulling an ungated model should not require a credential."""
    return os.environ.get(HF_SECRET_KEYS[0]) or os.environ.get(_HF_FALLBACK_ENV) or None


def require_hf_token(what: str) -> str:
    """The HF token, or a loud failure naming the secret and how to create it.

    Use on any path that cannot succeed anonymously (a gated pull, any push). Failing
    here beats proceeding unauthenticated and discovering it in a 401/403 after the GPU
    is hot.
    """
    tok = hf_token()
    if not tok:
        raise RuntimeError(
            f"{what} needs a HuggingFace token, but Secret {HF_SECRET_NAME} provided no "
            f"{HF_SECRET_KEYS[0]}. Create or repair it with:\n"
            f"{create_hint(HF_SECRET_NAME, HF_SECRET_KEYS)}\n"
            f"      The token must be WRITE-scoped for any push_to_hub; a read token 403s."
        )
    return tok

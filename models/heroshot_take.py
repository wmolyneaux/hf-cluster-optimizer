"""modallabs.models.heroshot_take -- Cycles take chunks for the berkeley-usd heroshot.

One run = one FRAME WINDOW of one take. The camera path depends on the TOTAL
frame count (place_rig.py: t = f/(n-1)), so every worker is told the full
--frames and only restricts what it WRITES via --frame-start/--frame-end --
both of which are place_rig digest inputs as of manifest v2. Fan N windows out
as N config runs and the harness's .spawn() concurrency, --dry-run pricing,
lane routing and the four BILL_SAFETY termination layers all apply unchanged.

MEASURED basis (berkeley-usd/docs/FAST-TAKE-PLAN.md):
  - local M5 Metal take: 753 f x 4.463 s/f = 56.5 min (walk270_v3, complete run)
  - chunk-context / standalone renders sit INSIDE the identical-command
    run-to-run band (60/72 px at +/-1 LSB vs control 74 of 589,824); a wrong
    --frames moves 99.48% of pixels, mean |diff| 45.87 -- hence _BAND below.
  - the landed --frame-start/--frame-end script reproduced walk270_v3 f_0400
    at 66 px, +/-1 LSB (regression run, 2026-08-12).

The trainer shells Blender; it does not import bpy. Blender's own stdout is
streamed to <output_dir>/blender_<tag>.log, so the L4 dead-man switch sees
progress at every frame without trainer cooperation.

PILOT MODE (config.pilot: true): one container renders the splice trio
(chunk [398,410), standalone [400,401), NEGATIVE --frames 401) plus nothing
else, band-compares them in-container, and reports steady-state s/f on H100 --
the one number the fleet sizing table is waiting on. The negative arm MUST
fail the band or the comparator is broken; both outcomes are asserted.

ASCII only. No emojis.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from modallabs.base import (
    Trainer,
    TrainerEpochResult,
    TrainerSetup,
    TrainerStepResult,
)
from modallabs.registry import register


class HeroshotTakeError(RuntimeError):
    pass


_BLENDER = os.environ.get("HEROSHOT_BLENDER", "/opt/blender/blender")
_BUSD = Path(os.environ.get("BUSD_MOUNT", "/busd"))
# MEASURED on pilot r2 (2026-08-12): reading the campus tree straight off the
# volume FUSE mount cost ~430 s of scene setup per COLD invocation (chunk arm:
# 460.4 s wall for ~26 s of rendering) vs ~40 s warm (attempt 2: 43.8 s total)
# -- thousands of small texture files at per-file FUSE latency. The fix is one
# sequential read: stage_berkeley_take.sh puts busd_take.tar on the volume,
# and each container untars it ONCE to local disk. A brief-lane worker must
# fit L3's 480 s; 430 s of avoidable I/O does not.
_BUSD_TAR_NAME = "busd_take.tar"
_LOCAL_ROOT = Path("/tmp/busd")
_REQUIRED = ("retarget", "frames")
# Acceptance band, MEASURED on the identical-command control (74 px of 589,824
# at +/-1 LSB). The trap signature is 4 orders of magnitude away (99.48% px).
_BAND_FRAC = 0.0005          # <= 0.05% of pixels may differ ...
_BAND_MAXDIFF = 1            # ... and only by one 8-bit step.
_TIME_RE = re.compile(r"^Time: (\d{2}):(\d{2}\.\d{2}) \(Sav")
# Crash-retry bound. An L3/L4 watchdog kill is os._exit, which Modal treats as
# a container CRASH and reschedules -- fleet r1 (2026-08-12) looped
# kill -> fresh container -> recompile -> kill, billing every lap, until the
# app was stopped BY HAND ($1.4 actual vs $1.02 modelled). One retry is
# legitimate (preemption, transient infra, or a retry that CACHE-HITs prior
# work -- pilot r1's retry did exactly that); a THIRD container entering the
# same run dir means the setup systematically cannot finish, and burning
# another H100 proves nothing new. Raising here is an app-level exception:
# runner.train_one catches it, the function RETURNS phase=failed, and Modal
# does not reschedule -- the loop is cut.
_MAX_ATTEMPTS = 2


def _band_compare(a_png: Path, b_png: Path) -> Dict[str, float]:
    """Pixel band comparison; prints what it selected, never a bare verdict."""
    import numpy as np
    from PIL import Image

    a = np.asarray(Image.open(a_png).convert("RGB"), np.int16)
    b = np.asarray(Image.open(b_png).convert("RGB"), np.int16)
    if a.shape != b.shape:
        raise HeroshotTakeError(f"shape mismatch {a.shape} vs {b.shape}: "
                                f"{a_png} vs {b_png}")
    d = np.abs(a - b)
    n_px = int(d.shape[0] * d.shape[1])
    n_diff = int((d.sum(2) > 0).sum())
    return {"n_px": n_px, "n_diff": n_diff, "frac": n_diff / n_px,
            "maxdiff": int(d.max()), "meandiff": float(d.mean())}


def _band_ok(r: Dict[str, float]) -> bool:
    return r["frac"] <= _BAND_FRAC and r["maxdiff"] <= _BAND_MAXDIFF


@register("heroshot_take")
class HeroshotTakeTrainer(Trainer):

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = dict(config)
        self._setup_obj: Optional[TrainerSetup] = None
        self._results: List[Dict[str, Any]] = []
        self._root: Path = _BUSD

    def _resolve_root(self, log_fn) -> Path:
        """Untar the input bundle to local disk once per container, else fall
        back to the raw volume mount (correct but ~430 s slower when cold)."""
        tar_p = _BUSD / _BUSD_TAR_NAME
        if not tar_p.exists():
            log_fn(f"no {_BUSD_TAR_NAME} on the volume; using FUSE mount directly "
                   f"(MEASURED ~430 s cold setup penalty)")
            return _BUSD
        marker = _LOCAL_ROOT / ".untarred"
        if not marker.exists():
            t0 = time.time()
            _LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
            subprocess.run(["tar", "-xf", str(tar_p), "-C", str(_LOCAL_ROOT)],
                           check=True)
            marker.write_text(f"{time.time():.0f}\n")
            log_fn(f"untarred {tar_p.stat().st_size >> 20} MiB of inputs to "
                   f"{_LOCAL_ROOT} in {time.time() - t0:.1f}s")
        return _LOCAL_ROOT

    # ------------------------------------------------------------------ config
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HeroshotTakeTrainer":
        missing = [k for k in _REQUIRED if k not in config]
        if missing:
            raise HeroshotTakeError(f"heroshot_take config missing keys: {missing}")
        cfg = dict(config)
        n = int(cfg["frames"])
        cfg.setdefault("frame_start", 0)
        cfg.setdefault("frame_end", n)
        a, b = int(cfg["frame_start"]), int(cfg["frame_end"])
        if not (0 <= a < b <= n):
            raise HeroshotTakeError(
                f"window [{a},{b}) invalid for frames={n}: need 0 <= start < end <= frames")
        cfg.setdefault("res", "1024x576")
        cfg.setdefault("samples", 96)
        cfg.setdefault("look", "plain")
        cfg.setdefault("stage", "usd/shots/shotHeroGlade.usda")
        cfg.setdefault("glb", "rig/rigged.glb")
        cfg.setdefault("pilot", False)
        if int(cfg.get("epochs", 1)) != 1:
            raise HeroshotTakeError("one epoch is one window: set epochs to 1")
        return cls(cfg)

    # ------------------------------------------------------------------- setup
    def setup(self, setup: TrainerSetup) -> None:
        self._setup_obj = setup
        cfg = self.config
        if cfg.get("stub"):
            setup.log_fn("heroshot_take: STUB mode -- no Blender, no volume")
            return
        # Bound the L3/L4-kill -> Modal crash-retry loop (see _MAX_ATTEMPTS).
        # The run dir persists on the runs volume across attempts, so the
        # counter survives os._exit; a fresh run_id starts the count fresh.
        attempts_p = setup.output_dir / ".heroshot_attempts"
        try:
            n_prev = int(attempts_p.read_text().strip() or "0") \
                if attempts_p.exists() else 0
        except (OSError, ValueError):
            n_prev = 0
        if n_prev >= _MAX_ATTEMPTS:
            raise HeroshotTakeError(
                f"attempt {n_prev + 1} on this run dir: {n_prev} earlier "
                "container(s) entered and never finished (watchdog kill -> "
                "Modal crash-retry). A third container would fail the same "
                "way at the same price. Fix the cause (cold kernel cache? "
                "window too big for the lane?), then relaunch under a FRESH "
                f"run_id -- or delete {attempts_p} to deliberately re-arm.")
        attempts_p.write_text(f"{n_prev + 1}\n", encoding="utf-8")
        setup.log_fn(f"attempt {n_prev + 1}/{_MAX_ATTEMPTS} for this run dir")
        self._root = self._resolve_root(setup.log_fn)
        root = self._root
        # Fail at minute 0, before any GPU sampling: every input the render
        # will read must already be staged, and the binary must run.
        needed = [
            Path(_BLENDER),
            root / "tools/heroshot/place_rig.py",
            root / "tools/render/lut_repair.py",
            root / "tools/render/material_lut.json",
            root / cfg["retarget"],
            root / cfg["stage"],
            root / cfg["glb"],
            root / "textures/_sky/campusSky_2026-09-22T0910PDT.exr",
        ]
        for p in needed:
            if not p.exists() or (p.is_file() and p.stat().st_size == 0):
                raise HeroshotTakeError(f"input not staged: {p}")
            setup.log_fn(f"input ok: {p}")
        ver = subprocess.run([_BLENDER, "--version"], capture_output=True,
                             text=True, timeout=120)
        if ver.returncode != 0:
            raise HeroshotTakeError(f"blender --version rc={ver.returncode}: "
                                    f"{(ver.stdout + ver.stderr)[-500:]}")
        setup.log_fn(f"blender: {ver.stdout.splitlines()[0]}")

    # ------------------------------------------------------------------ render
    def _run_blender(self, tag: str, outdir: Path, frames: int,
                     fstart: int, fend: int) -> Dict[str, Any]:
        assert self._setup_obj is not None
        cfg = self.config
        root = self._root
        log_path = self._setup_obj.output_dir / f"blender_{tag}.log"
        argv = [
            _BLENDER, "-b", "--factory-startup",
            "-P", str(root / "tools/heroshot/place_rig.py"), "--",
            "--retarget", str(root / cfg["retarget"]),
            "--frames", str(frames),
            "--frame-start", str(fstart), "--frame-end", str(fend),
            "--res", str(cfg["res"]), "--samples", str(cfg["samples"]),
            "--look", str(cfg["look"]),
            "--stage", str(root / cfg["stage"]),
            "--glb", str(root / cfg["glb"]),
            "--resume",
            "--outdir", str(outdir),
        ]
        env = dict(os.environ)
        env["BUSD_ROOT"] = str(root)
        env["PYTHONUNBUFFERED"] = "1"
        self._setup_obj.log_fn(f"[{tag}] {' '.join(argv)}")
        t0 = time.time()
        # Blender's stdout goes STRAIGHT to a file under the run dir: that is
        # what keeps the L4 dead-man switch fed (a write per frame), and the
        # full log survives for the s/f parse and the device assert below.
        with log_path.open("wb") as fh:
            proc = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT,
                                    env=env)
            rc = proc.wait()
        wall = time.time() - t0
        text = log_path.read_text(errors="replace")
        if rc != 0:
            raise HeroshotTakeError(
                f"[{tag}] blender rc={rc} after {wall:.1f}s; log tail:\n{text[-3000:]}")
        m = re.search(r"device probe: (GPU-\w+)", text)
        if not m:
            # place_rig refuses CPU itself (SystemExit -> rc!=0); no probe line
            # on rc==0 means the script changed under us -- refuse loudly.
            raise HeroshotTakeError(f"[{tag}] no 'device probe: GPU-*' line in log")
        device = m.group(1)
        times = [int(a) * 60 + float(b)
                 for a, b in (mm.groups() for mm in map(_TIME_RE.match,
                                                        text.splitlines()) if mm)]
        # first Time line is the silmask render, second is the first frame of
        # the process (carries kernel warmup); steady state is the rest.
        steady = times[2:] if len(times) > 2 else []
        # verify the window's frames actually exist -- glob is the truth, not rc
        missing = [k for k in range(fstart, fend)
                   if not (outdir / ("f_%04d.png" % k)).exists()]
        if missing:
            raise HeroshotTakeError(f"[{tag}] {len(missing)} window frames missing, "
                                    f"first {missing[:5]}")
        if not (outdir / "f_0000.png").exists():
            raise HeroshotTakeError(f"[{tag}] gate frame f_0000.png missing")
        man_p = outdir / "manifest.json"
        if not man_p.exists():
            raise HeroshotTakeError(f"[{tag}] manifest.json missing")
        man = json.loads(man_p.read_text())
        di = man.get("digest_inputs", {})
        if (di.get("frame_start"), di.get("frame_end")) != (fstart, fend):
            raise HeroshotTakeError(
                f"[{tag}] manifest window {di.get('frame_start')},{di.get('frame_end')} "
                f"!= requested {fstart},{fend}")
        r = {
            "tag": tag, "rc": rc, "wall_sec": round(wall, 1), "device": device,
            "digest": man.get("digest", "")[:16],
            "frames_written": fend - fstart,
            "s_f_steady": round(sum(steady) / len(steady), 3) if steady else None,
            "s_f_min": round(min(steady), 3) if steady else None,
            "s_f_max": round(max(steady), 3) if steady else None,
        }
        self._setup_obj.log_fn(f"[{tag}] done: {r}")
        return r

    # ---------------------------------------------------------------- lifecycle
    def train_iter(self) -> Iterable[Any]:
        return iter([dict(self.config)])

    def eval_iter(self) -> Iterable[Any]:
        return iter(())

    def train_step(self, batch: Any) -> TrainerStepResult:
        assert self._setup_obj is not None
        cfg = dict(batch)
        out_root = self._setup_obj.output_dir
        if cfg.get("stub"):
            (out_root / "take").mkdir(parents=True, exist_ok=True)
            (out_root / "take" / "stub.txt").write_text("stub")
            m = {"frames": 0.0, "wall_sec": 0.0}
            self._results.append({"tag": "stub", **m})
            return TrainerStepResult(metrics=m, n_examples=1)

        n = int(cfg["frames"])
        if cfg.get("pilot"):
            # -- splice trio + the s_f the fleet table is waiting on ----------
            chunk = self._run_blender("chunk", out_root / "take_chunk", n, 398, 410)
            alone = self._run_blender("alone", out_root / "take_alone", n, 400, 401)
            neg = self._run_blender("negative", out_root / "take_neg", 401, 400, 401)
            band_ac = _band_compare(out_root / "take_alone/f_0400.png",
                                    out_root / "take_chunk/f_0400.png")
            band_neg = _band_compare(out_root / "take_neg/f_0400.png",
                                     out_root / "take_alone/f_0400.png")
            band_gate = _band_compare(out_root / "take_alone/f_0000.png",
                                      out_root / "take_chunk/f_0000.png")
            self._setup_obj.log_fn(f"BAND alone-vs-chunk: {band_ac}")
            self._setup_obj.log_fn(f"BAND gate f_0000 cross-invocation: {band_gate}")
            self._setup_obj.log_fn(f"BAND NEGATIVE (wrong n): {band_neg}")
            if not _band_ok(band_ac):
                raise HeroshotTakeError(f"SPLICE BAND FAILED on this device: {band_ac}")
            if not _band_ok(band_gate):
                raise HeroshotTakeError(f"gate-frame band FAILED: {band_gate}")
            if _band_ok(band_neg) or band_neg["frac"] < 0.10:
                raise HeroshotTakeError(
                    f"NEGATIVE control passed the band -- comparator broken: {band_neg}")
            m = {
                "s_f_steady": float(chunk["s_f_steady"] or 0.0),
                "band_ac_frac": band_ac["frac"], "band_ac_max": band_ac["maxdiff"],
                "band_neg_frac": band_neg["frac"],
                "wall_sec": chunk["wall_sec"] + alone["wall_sec"] + neg["wall_sec"],
            }
            self._results += [chunk, alone, neg,
                              {"tag": "band_alone_vs_chunk", **band_ac},
                              {"tag": "band_gate_f0000", **band_gate},
                              {"tag": "band_negative", **band_neg}]
            return TrainerStepResult(metrics=m, n_examples=3)

        a, b = int(cfg["frame_start"]), int(cfg["frame_end"])
        r = self._run_blender("take", out_root / "take", n, a, b)
        self._results.append(r)
        m = {"frames": float(r["frames_written"]), "wall_sec": r["wall_sec"],
             "s_f_steady": float(r["s_f_steady"] or 0.0)}
        return TrainerStepResult(metrics=m, n_examples=r["frames_written"])

    def eval_step(self, batch: Any) -> TrainerStepResult:  # pragma: no cover
        raise HeroshotTakeError("eval_step unreachable: eval_iter is empty")

    def epoch_summary(self, epoch: int) -> TrainerEpochResult:
        last = self._results[-1] if self._results else {}
        wall = last.get("wall_sec", 0.0)
        return TrainerEpochResult(train_metrics={"wall_sec": float(wall or 0.0)},
                                  val_metrics={}, is_best=False,
                                  monitor_value=float(wall or 0.0))

    def save_checkpoint(self, path: Path) -> None:
        p = Path(path).with_suffix(".json")
        i = 1
        while p.exists():
            p = Path(path).with_suffix(f".{i}.json")
            i += 1
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "trainer": "heroshot_take",
            "config": self.config,
            "results": self._results,
        }, indent=2), encoding="utf-8")

    def load_checkpoint(self, path: Path) -> None:
        p = Path(path).with_suffix(".json")
        if p.exists():
            self._results = list(json.loads(p.read_text()).get("results", []))

    def num_epochs(self) -> int:
        return 1

    def teardown(self) -> None:
        pass

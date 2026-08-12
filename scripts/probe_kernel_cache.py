#!/usr/bin/env python
"""Locate, prime, and PROVE the Cycles/OptiX kernel cache for the heroshot lane.

WHY THIS EXISTS
---------------
MEASURED on fleet r1 (2026-08-12): the FIRST Cycles render in a fresh container
spends ~420-430 s in "Loading render kernels (may take a few minutes the first
time)" before a single frame is written. That is longer than the brief lane's
480 s L3 window, so the watchdog killed the worker and Modal's crash-retry
recompiled from scratch in ANOTHER fresh container. Retry churn, not render
work, is what overran the pilot.

The landed fix mounts a Modal Volume at /root/.cache and asserts nothing. This
file is the assertion. It refuses to take the mount point on faith:

  locate   (cheap GPU) -- snapshot the whole filesystem, compile the kernels for
                          real, snapshot again, and print WHAT WAS WRITTEN. Then
                          re-run in the SAME container (fast?) and then again
                          with the candidate deleted (slow again?). Correlation
                          plus causation, not documentation.
  prime    (H100)      -- compile sm_90 kernels with the volume mounted at the
                          LOCATED path; report bytes that actually landed.
  coldtest (H100)      -- a DIFFERENT @app.function, so Modal cannot hand back
                          the primed container. Asserts a different
                          MODAL_TASK_ID and an empty container-local scratch,
                          then measures kernel-load time again.

  modal run scripts/probe_kernel_cache.py --mode locate
  HEROSHOT_CACHE_MOUNT=<path> modal run scripts/probe_kernel_cache.py --mode prime
  HEROSHOT_CACHE_MOUNT=<path> modal run scripts/probe_kernel_cache.py --mode coldtest

HEROSHOT_CACHE_MOUNT / HEROSHOT_OPTIX_CACHE_PATH are read LOCALLY at import
time (the decorators run on the launcher), so one file covers "test the
hypothesis" and "test the fix" without an edit between the two.

ASCII only. No emojis.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO.parent) not in sys.path:
    sys.path.insert(0, str(_REPO.parent))

# Import, do not re-declare: a copy-pasted image definition is a second thing
# that can drift from the one the lane actually runs, and the whole point here
# is to measure THE LANE'S container.
from modallabs.modal_app import heroshot_image  # noqa: E402

app = modal.App("heroshot-kernel-probe")

# Where to mount the cache volume. Default is the LANDED guess; the locate run
# exists to confirm or refute it.
_CACHE_MOUNT = os.environ.get("HEROSHOT_CACHE_MOUNT", "/kcache")

# MEASURED on the T4 locate run (2026-08-12), which refuted /root/.cache:
#   * the ONLY thing a Cycles OptiX render created anywhere on the filesystem
#     was /var/tmp/OptixCache_root/optix7cache.db (1,114,112 B). Nothing under
#     /root/.cache. The landed mount was decorative.
#   * Blender 4.5.12's tarball ships cubins for sm_30/35/37/50/52/60/61/70/75/
#     86/89/120 and NO sm_90 and NO sm_80. T4 is sm_75, finds its cubin, and
#     loads render kernels in under a second (whole render 3.37 s) -- which is
#     why the T4 could never have reproduced the H100 stall.
#   * with no sm_90 cubin, Cycles falls back to lib/kernel_compute_75.ptx and
#     the CUDA DRIVER JITs PTX->SASS for sm_90. That is the 420-430 s. The
#     driver's cache is $HOME/.nv/ComputeCache, not $HOME/.cache.
#
# So there are TWO caches to catch, in two places, neither of them the landed
# one. Both are redirected by env var AND backstopped by a symlink from the
# default location, deliberately pointing at DIFFERENT subdirectories so the
# filesystem diff says which mechanism actually caught it instead of leaving
# two plausible explanations for one result.
_OPTIX_ENV_DIR = f"{_CACHE_MOUNT}/optix_env"
_OPTIX_LINK_DIR = f"{_CACHE_MOUNT}/optix_link"
_CUDA_ENV_DIR = f"{_CACHE_MOUNT}/nv_env"
_CUDA_LINK_DIR = f"{_CACHE_MOUNT}/nv_link"
_LINKS = {"/var/tmp/OptixCache_root": _OPTIX_LINK_DIR, "/root/.nv": _CUDA_LINK_DIR}
_ENV = {
    "PYTHONUNBUFFERED": "1",
    "OPTIX_CACHE_PATH": _OPTIX_ENV_DIR,
    "CUDA_CACHE_PATH": _CUDA_ENV_DIR,
    "CUDA_CACHE_DISABLE": "0",
    # Default compute-cache cap is small enough to evict a megakernel; the
    # whole point is that nothing gets evicted between the prime and the fleet.
    "CUDA_CACHE_MAXSIZE": str(4 * 1024 ** 3),
}
_secret = modal.Secret.from_dict(_ENV)


# "direct" mounts the volume AT the cache path, so the JIT writes straight
# through the Modal FUSE mount. "copy" uses the volume as TRANSPORT only: the
# cache is copied to local disk before the render and back afterwards, so the
# compiler only ever touches local disk. copy also removes write contention --
# eight fleet workers would otherwise all be writing one volume.
_CACHE_MODE = os.environ.get("HEROSHOT_CACHE_MODE", "direct")
# Local working locations = the defaults the two compilers pick on their own,
# MEASURED for OptiX (/var/tmp/OptixCache_root) and documented for the CUDA
# driver ($HOME/.nv/ComputeCache).
_LOCAL_OPTIX = "/var/tmp/OptixCache_root"
_LOCAL_CUDA = "/root/.nv/ComputeCache"


def _copy_tree(src: str, dst: str) -> dict:
    t0 = time.time()
    if not Path(src).is_dir():
        return {"src": src, "dst": dst, "copied": False, "reason": "src absent"}
    Path(dst).mkdir(parents=True, exist_ok=True)
    rc = subprocess.run(f"cp -a {src}/. {dst}/", shell=True,
                        capture_output=True, text=True)
    n = sum(len(f) for _, _, f in os.walk(dst))
    return {"src": src, "dst": dst, "copied": rc.returncode == 0, "n_files": n,
            "sec": round(time.time() - t0, 2), "err": rc.stderr[-200:]}


def _wire_cache() -> dict:
    """Point both compilers at a cache that will persist, and say how."""
    made: dict = {"mode": _CACHE_MODE}
    if _CACHE_MODE == "copy":
        # Env vars point at LOCAL disk; the volume is only read in / written out.
        os.environ["OPTIX_CACHE_PATH"] = _LOCAL_OPTIX
        os.environ["CUDA_CACHE_PATH"] = _LOCAL_CUDA
        made["copy_in"] = [_copy_tree(f"{_CACHE_MOUNT}/optix", _LOCAL_OPTIX),
                           _copy_tree(f"{_CACHE_MOUNT}/nv", _LOCAL_CUDA)]
        return made
    for d in (_OPTIX_ENV_DIR, _OPTIX_LINK_DIR, _CUDA_ENV_DIR, _CUDA_LINK_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)
    for src, dst in _LINKS.items():
        p = Path(src)
        try:
            if p.is_symlink() or p.exists():
                made[src] = f"already exists -> {os.path.realpath(src)}"
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            p.symlink_to(dst)
            made[src] = f"symlink -> {dst}"
        except OSError as exc:
            made[src] = f"FAILED: {exc}"
    return made


def _copy_out() -> list:
    """prime only: publish the local cache onto the volume."""
    if _CACHE_MODE != "copy":
        return []
    return [_copy_tree(_LOCAL_OPTIX, f"{_CACHE_MOUNT}/optix"),
            _copy_tree(_LOCAL_CUDA, f"{_CACHE_MOUNT}/nv")]

kernel_cache = modal.Volume.from_name("heroshot-kernel-cache", create_if_missing=True)
busd = modal.Volume.from_name("berkeley-usd-take", create_if_missing=False)

_BLENDER = "/opt/blender/blender"
_SKIP_ROOTS = {"/proc", "/sys", "/dev", "/busd", "/runs"}

# Minimal Cycles/OptiX render. Mirrors place_rig.py's backend order and its
# use_hardware_raytracing choice, because those select WHICH OptiX modules get
# compiled -- a cache primed with different settings is a cache for a different
# kernel set.
_MINI_PY = r'''
import bpy, os, sys, time
print("ENV HOME=%r XDG_CACHE_HOME=%r OPTIX_CACHE_PATH=%r USER=%r" % (
    os.environ.get("HOME"), os.environ.get("XDG_CACHE_HOME"),
    os.environ.get("OPTIX_CACHE_PATH"), os.environ.get("USER")))
pr = bpy.context.preferences.addons["cycles"].preferences
resolved = "CPU"
for want in ("OPTIX", "CUDA"):
    try:
        pr.compute_device_type = want
        pr.get_devices()
        if any(getattr(d, "type", "") == want for d in pr.devices):
            resolved = "GPU-" + want
            break
    except Exception as exc:
        print("backend %s unavailable: %s" % (want, exc))
if not resolved.startswith("GPU-"):
    raise SystemExit("GPU REQUIRED: no Cycles GPU backend")
print("device probe: " + resolved)
for d in pr.devices:
    print("  device: %s type=%s" % (d.name, d.type))
    d.use = True
try:
    pr.use_hardware_raytracing = True
except Exception as exc:
    print("no use_hardware_raytracing: %s" % exc)
sc = bpy.context.scene
sc.render.engine = "CYCLES"
sc.cycles.device = "GPU"
sc.cycles.samples = 1
sc.render.resolution_x = 32
sc.render.resolution_y = 32
sc.render.filepath = sys.argv[-1]
t0 = time.time()
bpy.ops.render.render(write_still=True)
print("RENDER_WALL_SEC %.2f" % (time.time() - t0))
'''


# --------------------------------------------------------------- measurement
class _LogWatch(threading.Thread):
    """Wall-clock stamp the FIRST appearance of each marker in a growing log.

    Kernel-load time is the gap between "Loading render kernels" and the first
    "Path Tracing Sample": parsing it beats timing the whole process, which
    also contains scene build, BVH and I/O.
    """

    # "| Sample " and not "Path Tracing Sample": MEASURED 2026-08-12, Blender
    # 4.5.12 -b prints "... | Scene, ViewLayer | Sample 0/1". The first probe
    # watched for a string this build never emits and reported
    # kernel_load_sec: null on a run that was otherwise fine.
    MARKERS = ("device probe:", "Loading render kernels", "| Sample ",
               "Loading denoising kernels", "Saved:", "Updating Scene")

    def __init__(self, path: Path, t0: float) -> None:
        super().__init__(daemon=True)
        self.path, self.t0, self.at = path, t0, {}
        # NOT self._stop: threading.Thread._stop is an INTERNAL METHOD, and
        # shadowing it with an Event makes Thread.join() raise
        # "TypeError: 'Event' object is not callable" -- which is exactly what
        # it did, after a full T4 kernel compile, discarding the result.
        self._halt = threading.Event()

    def run(self) -> None:
        seen = 0
        while not self._halt.is_set():
            try:
                data = self.path.read_text(errors="replace")
            except OSError:
                time.sleep(0.25)
                continue
            if len(data) > seen:
                now = time.time() - self.t0
                for mk in self.MARKERS:
                    if mk not in self.at and mk in data:
                        self.at[mk] = round(now, 2)
                seen = len(data)
            time.sleep(0.1)

    def stop(self, t_exit: float) -> tuple:
        """Backfill markers flushed in the process's dying moments. They get
        t_exit -- an UPPER BOUND, not a measurement -- and are named in
        `backfilled` so a number that came from here is never mistaken for one
        the poller actually timed."""
        self._halt.set()
        self.join(timeout=3)
        backfilled = []
        try:
            data = self.path.read_text(errors="replace")
            for mk in self.MARKERS:
                if mk not in self.at and mk in data:
                    self.at[mk] = round(t_exit, 2)
                    backfilled.append(mk)
        except OSError:
            pass
        return dict(self.at), backfilled


def _timed_blender(argv: list, log: Path, env: dict) -> dict:
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("")
    t0 = time.time()
    watch = _LogWatch(log, t0)
    watch.start()
    with log.open("wb") as fh:
        rc = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT, env=env).wait()
    wall = time.time() - t0
    at, backfilled = watch.stop(wall)
    kl = None
    if "Loading render kernels" in at and "| Sample " in at:
        kl = round(at["| Sample "] - at["Loading render kernels"], 2)
    return {"rc": rc, "wall_sec": round(wall, 2), "markers_at_sec": at,
            "markers_backfilled_at_exit": backfilled,
            "kernel_load_sec": kl,
            "log_tail": log.read_text(errors="replace")[-1500:]}


def _snapshot(root: str = "/") -> dict:
    out = {}
    for dirpath, dirnames, filenames in os.walk(root, topdown=True,
                                                onerror=lambda e: None):
        dirnames[:] = [d for d in dirnames
                       if os.path.join(dirpath, d) not in _SKIP_ROOTS]
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            out[p] = (round(st.st_mtime, 2), st.st_size)
    return out


def _diff(before: dict, after: dict) -> dict:
    new = {p: v for p, v in after.items() if p not in before}
    chg = {p: (before[p], v) for p, v in after.items()
           if p in before and before[p] != v}
    return {"new": new, "changed": chg}


def _dirsizes(paths: dict) -> list:
    """Roll a file list up to directories -- the cache is a DIRECTORY, and its
    parent is the thing you can mount."""
    agg = {}
    for p, (_, size) in paths.items():
        d = os.path.dirname(p)
        n, b = agg.get(d, (0, 0))
        agg[d] = (n + 1, b + size)
    return sorted(({"dir": d, "n_files": n, "bytes": b} for d, (n, b) in agg.items()),
                  key=lambda r: -r["bytes"])


def _shipped_kernels() -> dict:
    """What GPU kernel artifacts does the TARBALL actually contain? The first
    pass looked only for the classic extensions and found NOTHING, which is
    itself the finding -- so also list anything named like a kernel and the
    biggest files, rather than concluding from an empty list."""
    by_ext, named, big = [], [], []
    for dp, _dn, fns in os.walk("/opt/blender"):
        for fn in fns:
            p = os.path.join(dp, fn)
            try:
                sz = os.path.getsize(p)
            except OSError:
                continue
            low = fn.lower()
            if low.endswith((".ptx", ".cubin", ".fatbin", ".optixir", ".hipfb",
                             ".zst", ".so", ".so.1")):
                by_ext.append({"path": p, "bytes": sz})
            if any(k in low for k in ("kernel", "optix", "cuda", "cycles", "nvrtc")):
                named.append({"path": p, "bytes": sz})
            if sz > 20_000_000:
                big.append({"path": p, "bytes": sz})
    key = lambda r: -r["bytes"]  # noqa: E731
    return {"by_extension": sorted(by_ext, key=key)[:30],
            "named_like_a_kernel": sorted(named, key=key)[:30],
            "over_20MB": sorted(big, key=key)[:15]}


# ------------------------------------------------------------------- locate
def _say(label: str, obj) -> None:
    """Print findings AS THEY LAND. A timeout kill discards the return value
    but not the logs, so the expensive evidence must not wait for the return."""
    print(f"\n===== {label} =====\n{json.dumps(obj, indent=2, default=str)}",
          flush=True)


@app.function(image=heroshot_image, gpu="T4", timeout=2400, secrets=[_secret],
              scaledown_window=2)
def locate() -> dict:
    """Cheapest GPU that can run OptiX at all. The cache PATH does not depend
    on the arch (only the filenames inside it do), so sm_75 answers "where" for
    T4 money instead of H100 money."""
    res: dict = {"task_id": os.environ.get("MODAL_TASK_ID"),
                 "env": {k: os.environ.get(k) for k in
                         ("HOME", "XDG_CACHE_HOME", "OPTIX_CACHE_PATH", "USER", "TMPDIR")},
                 "mounts": subprocess.run(["df", "-hT"], capture_output=True,
                                          text=True).stdout,
                 "shipped_kernels": _shipped_kernels()}
    # Corroboration only -- the filesystem diff below is the evidence.
    st = subprocess.run(
        "for f in /opt/blender/blender /opt/blender/lib/*.so* "
        "/usr/lib/x86_64-linux-gnu/libnvoptix.so.1; do "
        "  [ -f \"$f\" ] && strings -a \"$f\" | grep -iE "
        "'optixcache|OPTIX_CACHE|cycles/kernels|/\\.cache|optix7cache' "
        "  | sed \"s|^|$(basename $f): |\"; done | sort -u | head -60",
        shell=True, capture_output=True, text=True)
    res["binary_strings"] = st.stdout.splitlines()
    _say("environment + shipped kernels", res)

    Path("/tmp/mini.py").write_text(_MINI_PY)
    argv = [_BLENDER, "-b", "--factory-startup", "-noaudio",
            "-P", "/tmp/mini.py", "--", "/tmp/probe.png"]
    env = dict(os.environ)

    before = _snapshot()
    res["n_files_before"] = len(before)
    res["run1_cold"] = _timed_blender(argv, Path("/tmp/b1.log"), env)
    after = _snapshot()
    d = _diff(before, after)
    # /tmp noise from our own probe is not the cache.
    interesting = {p: v for p, v in d["new"].items()
                   if not p.startswith(("/tmp/b1.log", "/tmp/probe.png", "/tmp/mini.py"))}
    res["created_dirs"] = _dirsizes(interesting)[:25]
    res["created_files_sample"] = sorted(interesting)[:40]
    res["n_changed"] = len(d["changed"])
    _say("RUN 1 (cold) + WHAT IT WROTE", {"run1": res["run1_cold"],
                                          "created_dirs": res["created_dirs"],
                                          "created_files_sample":
                                              res["created_files_sample"]})

    # The controls are wrapped: a bug in a LATER phase must not discard the
    # location finding, which is the expensive part. (Learned the paid way --
    # a shadowed Thread._stop threw away a whole T4 compile.)
    try:
        # CONTROL 1: same container, new process. Fast here is necessary but
        # not sufficient -- a warm GPU/driver could explain it too, which is
        # what CONTROL 2 separates.
        res["run2_same_container"] = _timed_blender(argv, Path("/tmp/b2.log"), env)
        _say("RUN 2 (same container, cache present)", res["run2_same_container"])

        # CONTROL 2: delete the candidate, run again. Slow again == causation.
        cand = [r["dir"] for r in res["created_dirs"][:3] if r["bytes"] > 100_000]
        res["deleted"] = cand
        for c in cand:
            subprocess.run(["rm", "-rf", c], check=False)
        res["run3_cache_deleted"] = _timed_blender(argv, Path("/tmp/b3.log"), env)
        _say("RUN 3 (same container, cache DELETED)",
             {"deleted": cand, "run3": res["run3_cache_deleted"]})
    except Exception as exc:            # noqa: BLE001 -- report, never discard
        res["controls_error"] = f"{type(exc).__name__}: {exc}"
        _say("CONTROLS FAILED (location finding above still stands)",
             res["controls_error"])
    return res


# ------------------------------------------------------------- place_rig runs
def _place_rig_once(tag: str, out_root: Path, extra: Optional[list] = None) -> dict:
    """The REAL render script, one frame. Kernel variants compiled by Cycles
    depend on the scene's feature set, so priming with the default cube would
    prime a cache the campus scene then misses."""
    t_boot = time.time()
    local = Path("/tmp/busd")
    tar = Path("/busd/busd_take.tar")
    if not tar.exists():
        raise RuntimeError("busd_take.tar not staged on berkeley-usd-take")
    fresh_fs = not (local / ".untarred").exists()
    if fresh_fs:
        local.mkdir(parents=True, exist_ok=True)
        subprocess.run(["tar", "-xf", str(tar), "-C", str(local)], check=True)
        (local / ".untarred").write_text(str(time.time()))
    untar_sec = round(time.time() - t_boot, 2)

    outdir = out_root / tag
    outdir.mkdir(parents=True, exist_ok=True)
    argv = [_BLENDER, "-b", "--factory-startup",
            "-P", str(local / "tools/heroshot/place_rig.py"), "--",
            "--retarget", str(local / "shots/retarget_walk270_fix.json"),
            "--frames", "753", "--frame-start", "0", "--frame-end", "1",
            "--res", "1024x576", "--samples", "96", "--look", "plain",
            "--stage", str(local / "usd/shots/shotHeroGlade.usda"),
            "--glb", str(local / "rig/rigged.glb"),
            "--resume", "--outdir", str(outdir)] + list(extra or [])
    env = dict(os.environ)
    env["BUSD_ROOT"] = str(local)
    r = _timed_blender(argv, out_root / f"{tag}.log", env)
    # The CUDA JIT cache is keyed on (PTX, arch, DRIVER VERSION). If two
    # containers land on hosts with different drivers the cache misses and the
    # JIT returns, so a pass that did not record the driver cannot tell
    # "the cache carries" from "both containers happened to match".
    drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,name",
                          "--format=csv,noheader"], capture_output=True, text=True)
    r.update({"tag": tag, "untar_sec": untar_sec,
              "container_fs_was_fresh": fresh_fs,
              "gpu_driver": drv.stdout.strip() or drv.stderr.strip()[:200],
              "task_id": os.environ.get("MODAL_TASK_ID"),
              "frames_written": sorted(p.name for p in outdir.glob("*.png"))})
    return r


def _cache_state(label: str) -> dict:
    root = Path(_CACHE_MOUNT)
    files, total = [], 0
    for dp, _dn, fns in os.walk(root):
        for fn in fns:
            p = os.path.join(dp, fn)
            try:
                sz = os.path.getsize(p)
            except OSError:
                continue
            total += sz
            files.append({"path": p, "bytes": sz})
    return {"label": label, "mount": _CACHE_MOUNT, "n_files": len(files),
            "total_bytes": total,
            "files": sorted(files, key=lambda r: -r["bytes"])[:20]}


_H100 = dict(image=heroshot_image, gpu="H100", secrets=[_secret],
             volumes={"/busd": busd, _CACHE_MOUNT: kernel_cache})


@app.function(timeout=1200, scaledown_window=2, **_H100)
def prime() -> dict:
    """Pay the sm_90 JIT ONCE and leave it on the volume. Doubles as the
    sm_90 LOCATE run: it snapshots the whole filesystem around the render, so
    a cache that ignores both the env var and the symlink is reported by path
    instead of being silently missed. timeout=1200 is the worst-case bill (L2),
    ~2.2x the ~545 s this is expected to take."""
    kernel_cache.reload()
    res = {"task_id": os.environ.get("MODAL_TASK_ID"),
           "wired": _wire_cache(), "before": _cache_state("before")}
    _say("wiring + cache BEFORE prime", res)
    before_fs = _snapshot()
    res["render"] = _place_rig_once("prime", Path("/tmp/out"))
    _say("prime render", res["render"])
    d = _diff(before_fs, _snapshot())
    noise = ("/tmp/out", "/tmp/busd", "/runs")
    res["created_dirs_anywhere"] = [r for r in _dirsizes(
        {p: v for p, v in d["new"].items() if not p.startswith(noise)})[:25]]
    _say("WHAT THE sm_90 COMPILE WROTE, filesystem-wide",
         res["created_dirs_anywhere"])
    kernel_cache.commit()
    res["after"] = _cache_state("after")
    _say("cache AFTER prime (post-commit)", res["after"])
    return res


@app.function(timeout=600, scaledown_window=2, **_H100)
def coldtest() -> dict:
    """THE ACTUAL TEST. A DIFFERENT @app.function: Modal never hands one
    function's warm container to another, so this cannot silently be the
    primed container. Not taken on faith either -- it reports MODAL_TASK_ID to
    compare against prime's, and asserts the container-local /tmp/busd untar
    marker is ABSENT (a reused container would still have it, since /tmp is
    ephemeral container state and not on any volume)."""
    kernel_cache.reload()
    res = {"task_id": os.environ.get("MODAL_TASK_ID"),
           "tmp_busd_marker_present_at_start": Path("/tmp/busd/.untarred").exists(),
           "wired": _wire_cache(), "before": _cache_state("before")}
    _say("COLD container: identity + cache seen on the volume", res)
    if res["tmp_busd_marker_present_at_start"]:
        res["COLD_START_NOT_PROVEN"] = ("/tmp/busd/.untarred already existed: this "
                                        "container had already run a heroshot render, "
                                        "so any speedup below is NOT evidence.")
        _say("WARNING", res["COLD_START_NOT_PROVEN"])
    res["render"] = _place_rig_once("cold", Path("/tmp/out"))
    _say("COLD render", res["render"])
    res["after"] = _cache_state("after")
    return res


# Cycles selects OptiX MODULE VARIANTS by kernel feature set, so a cache primed
# against one look is not automatically a cache for another. The v2.3.2 NPR
# merge turns on twelve extra passes (light AOVs: diffuse/glossy/transmission
# direct+indirect+colour, emit, environment, normal) on top of the Z,
# cryptomatte and object-index the plain look already writes.
#
# Reasoning says the dominant cost cannot move: the CUDA side of the tarball
# ships exactly ONE artifact, lib/kernel_compute_75.ptx.zst, and the 548.8 s is
# the DRIVER JIT-ing that single fixed module to sm_90. Render passes are film
# configuration, not a different kernel binary, and Cycles' feature-ADAPTIVE
# compilation is a compile-from-source path that needs nvcc and kernel.cu --
# neither of which is in the tarball.
#
# But that is an argument, and arguments are what this project keeps having to
# refute with measurements. So this mode runs the REAL post-NPR place_rig with
# the maximal feature set against the ALREADY-PRIMED cache and reports
# kernel_load_sec. Small => the cache covers the look that will actually ship.
@app.function(timeout=600, scaledown_window=2, **_H100)
def npr() -> dict:
    kernel_cache.reload()
    res = {"task_id": os.environ.get("MODAL_TASK_ID"),
           "tmp_busd_marker_present_at_start": Path("/tmp/busd/.untarred").exists(),
           "wired": _wire_cache(), "before": _cache_state("before")}
    _say("NPR feature-set test: cold container, cache primed against PLAIN", res)
    # --allow-fused: the post-NPR place_rig added an ASSET GATE that refuses
    # the fused rigged.glb without a stated structural reason (MEASURED: rc=1
    # in 1.48 s on the first attempt). The gate is right and the reason is
    # real -- this frame is a kernel-cache probe and is thrown away.
    res["render"] = _place_rig_once(
        "npr", Path("/tmp/out"),
        ["--npr", "cel", "--outline", "both", "--exr-aov",
         "--allow-fused",
         "kernel-cache feature-set probe; frame is discarded, never delivered"])
    _say("NPR render", res["render"])
    res["after"] = _cache_state("after")
    _say("cache AFTER the NPR render (grew? by how much?)", res["after"])
    return res


@app.local_entrypoint()
def main(mode: str = "locate") -> None:
    fns = {"locate": locate, "prime": prime, "coldtest": coldtest, "npr": npr}
    if mode not in fns:
        raise SystemExit(f"mode must be one of {sorted(fns)}")
    print(f"[probe] mode={mode} cache_mount={_CACHE_MOUNT!r} env={_ENV} "
          f"links={_LINKS}")
    out = fns[mode].remote()
    Path(f"/tmp/probe_{mode}.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str)[:12000])
    print(f"\n[probe] full result: /tmp/probe_{mode}.json")

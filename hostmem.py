"""modallabs — host memory reader.

Deliberately a leaf module: stdlib only, no imports from the rest of the
package and no third-party deps. That is what lets a project which is not a
modallabs *training* dependent still install the harness `--no-deps` and share
this one implementation, instead of copying the arithmetic and letting the two
copies drift.

    from modallabs.hostmem import available_ram_gb, total_ram_gb

Why this exists: without a GPU, a concurrent worker is a whole model resident
in its own process, so the safe worker count is set by RAM, not by cores.
Sizing a local pool by cores is how a 24 GB box ends up holding 12 x 3.4 GB
model workers -- 29 GB resident, the compressor at 19.8 GB, and a jetsam
cascade in which the kernel kills 100+ system daemons rather than the Python
that caused the pressure. (Measured on a 24 GB M5, 2026-08-11: six JetsamEvent
reports in 11 minutes, per-worker lifetimeMax 3.37 GB.)
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

__all__ = ["available_ram_gb", "total_ram_gb"]


def available_ram_gb() -> Optional[float]:
    """Host RAM reclaimable without swapping, in GB. None if it cannot be read.

    Darwin: free + inactive pages, which is psutil's own definition of
    `available` on that platform. Compressed pages are deliberately not counted
    as available -- by the time the compressor is holding gigabytes the machine
    is already in the failure this guard exists to prevent.

    Linux: MemAvailable from /proc/meminfo, the kernel's own estimate.

    Returns None rather than a guess on any other platform or on any parse
    failure, so callers can say "not RAM-bounded" out loud instead of implying
    a guarantee that was never computed.
    """
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10)
            if out.returncode != 0:
                return None
            page_m = re.search(r"page size of (\d+) bytes", out.stdout)
            page = int(page_m.group(1)) if page_m else 4096
            want = ("Pages free", "Pages inactive")
            total = 0
            found = 0
            for line in out.stdout.splitlines():
                for key in want:
                    if line.startswith(key + ":"):
                        total += int(line.split(":")[1].strip().rstrip("."))
                        found += 1
            return (total * page) / 2**30 if found == len(want) else None
        if sys.platform.startswith("linux"):
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2**20  # kB -> GB
            return None
    except Exception:
        return None
    return None


def total_ram_gb() -> Optional[float]:
    """Physical RAM in GB, or None if it cannot be read."""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                 capture_output=True, text=True, timeout=10)
            return int(out.stdout.strip()) / 2**30 if out.returncode == 0 else None
        if sys.platform.startswith("linux"):
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 2**20
            return None
    except Exception:
        return None
    return None

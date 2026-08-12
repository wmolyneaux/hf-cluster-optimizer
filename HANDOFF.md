# hf-gpu-cluster-optimizer (modallabs) — HANDOFF

**Session 2026-08-11.** Added two Orpheus lanes, a host-RAM bound, a credential convention,
and a 600 s lane. **Everything below is UNCOMMITTED.** Will declined a push (the repo is
public); nothing was committed either, so this tree is the only copy.

**GPU spend attributable to this repo: $0.00.** The $4.44 spent this session was voicecraft's
13 runs (`~/voicecraft/HANDOFF.md`), which exercised these lanes.

---

## 0. FIRST: two sessions' work is mixed in this tree

`git status` shows work from **two** sessions. Committing it as one change would conflate them.

| Path | Whose | Note |
|---|---|---|
| `hostmem.py` | 2026-08-11 (new) | 84 lines |
| `credentials.py` | 2026-08-11 (new) | 77 lines |
| `models/orpheus_voice.py` | 2026-08-11 (new) | 368 lines |
| `models/orpheus_tts.py` | 2026-08-11 (new) | 303 lines |
| `concurrent_train.py` | 2026-08-11 | **all** +103 lines. File was clean at HEAD (mtime Jul 30) when the session began. |
| `modal_app.py` | **BOTH** | 375 added lines; only **13** are 2026-08-11 (grep `brief\|_creds\|HF_SECRET\|peft\|snac`). ~86 mention `tram`/`longcat` — that is Aug 4 work, uncommitted since. |
| `models/__init__.py` | **BOTH** | 4 added `_safe_import` lines: `longcat_avatar` + `tram_motion` (Aug 4), `orpheus_voice` + `orpheus_tts` (Aug 11) |
| `tests/smoke.py` | **BOTH** | 4 added cases, same split |
| `models/longcat_avatar.py`, `models/tram_motion.py` | **Aug 4, untracked** | never committed; `models/__init__.py` imports them |

**Suggested split:** commit the Aug 4 tram/longcat lane wiring first (it is the older, larger
change and was already working), then tonight's four new files + the RAM bound + the 13
modal_app lines as a second commit.

---

## 1. State — DONE and verified

### `hostmem.py` (new) — host RAM, stdlib only
`available_ram_gb()` / `total_ram_gb()`. Darwin uses `vm_stat` free+inactive (psutil's own
definition of `available` on that platform); Linux uses `MemAvailable`. **Compressed pages are
deliberately not counted as available** — by the time the compressor holds gigabytes the
machine is already in the failure this prevents. Returns `None` rather than a guess on any
other platform or parse failure, so callers can say "not RAM-bounded" out loud.

A stdlib-only **leaf** on purpose: imports nothing from modallabs, nothing from modal. That is
what lets a non-training project `uv pip install -e . --no-deps` and share it. voicecraft does
exactly that (`prep/transcribe.py`).

### `concurrent_train.py` — the local pool is now RAM-bounded
`_default_max_workers` was `cpu_count() // 2`, i.e. **RAM-blind**. On a 24 GB box that means 5
concurrent whole-model workers. Added `LOCAL_RESERVE_GB = 4.0` (a **policy** choice, labelled
as such, not a measurement), `_declared_mem_gb()`, and `_ram_bounded_workers()`, wired into
`run()` with the arithmetic logged and persisted to `summary.json`
(`max_workers_requested`, `host_available_gb`, `ram_bound`).

Rules: only ever clamps **down**; bounds on the **heaviest k** footprints, not the average,
because the pool can schedule the heaviest runs together; and if **any** run lacks `mem_gb:`
it prints "**NOT RAM-bounded**" rather than implying a guarantee it cannot make.

Verified — 7 assertions, all passing:
- incident replay: 12 requested @ 3.37 GB on a 20 GB box → **4** (the real event ran 12)
- undeclared runs → unchanged count + "NOT RAM-bounded"
- mixed 9/9/0.5 GB, 16 GB budget → **1** (heaviest-k, not average)
- single oversized run → floors at 1 **and warns with the real footprint** (an early version
  reported a bogus `0.0 GB` here)
- unreadable RAM → degrades honestly, never to a fake bound
- end-to-end: 3 runs declaring 9 GB each on a 7.7 GB box → clamped **3 → 1**, all 3 still
  succeeded, arithmetic in `summary.json`

### `credentials.py` (new) — the secret convention, copied from smpl
Mirrors `tram-motion/lane/trammotion/config.py` + `scripts/stage_weights.py`, at Will's
instruction (*"implement how we do all the other hf harness with secrets, similar to what we
did with smpl"*): a named constant per secret **with provenance**, the exact
`modal secret create` line, and a consumption check that fails loud quoting it.

`HF_SECRET_NAME = "huggingface-secret"`, `HF_SECRET_KEYS = ("HF_TOKEN",)`, `hf_token()`
(optional — anonymous access is fine for ungated pulls), `require_hf_token()` (loud),
`create_hint()`.

**The key name was read back, not guessed** — the smpl technique: a few seconds of CPU in a
container printing **KEY NAMES ONLY, never values**. Probe script:
`/private/tmp/.../scratchpad/probe_hf_secret.py` (session scratch — recreate from §2 if gone).

Named `credentials.py` and **not `secrets.py`**: this package is flat-layout (the repo root
*is* the `modallabs` package), so a `secrets.py` here can shadow the stdlib module.

### `modal_app.py` — 13 lines
1. `from modallabs import credentials as _creds`
2. `secrets=[modal.Secret.from_name(_creds.HF_SECRET_NAME)]` on `_COMMON`, attached
   **unconditionally** (an earlier best-effort `try/except` degraded silently, which is what
   the smpl convention exists to prevent)
3. `peft` added to the generic image — **the README has always documented a `peft:` block on
   any `hf_*` run and the image never carried it**, so every documented PEFT run failed on
   import after the GPU was hot
4. `snac` added — for `orpheus_tts`, rather than giving it a whole typed lane and image
5. `brief` lane at **600 s** + `_remote_brief` + `_LANE_FNS` entry

The `brief` lane is off **measured** runtimes (217 s to train, 187 s to generate), not a guess.
Worst case per run **$2.75 → $0.92**; a 7-run fan-out gates at $6.42 instead of $19.25. Verified
boundary: 600 → `brief`, 601 → `short`.

### `models/orpheus_voice.py` (new) — Orpheus 3B LoRA
Ported from voicecraft's bespoke `modal_app/train.py`. Exists as its own lane because generic
`hf_causal_lm` tokenizes a **text column** and Orpheus trains on a 7-tokens-per-frame SNAC
interleave — the part that fails *silently* when wrong, so the ids are built once in
voicecraft and consumed verbatim here.

Checkpoints are PEFT adapters (96 MB, not the 3B base) written as ordinary `save_pretrained`
dirs plus **`base_model.txt`**, because an adapter without its base is not reloadable. Hub push
is deferred to `teardown()` — pushing per `is_best` epoch bills uploads at GPU rates.

**`_preflight_hub_push()` is the piece worth keeping.** A token's key name says nothing about
its scope. `whoami` at `setup()` refuses a read-only token or an unwritable namespace **before
anything allocates**, per PORTING.md's burn rule (*failing at minute 0 is free*). It exists
because the first run trained 216 s and then 403'd. Tested: read-only refused, wrong namespace
refused, own namespace passes, org namespace passes.

### `models/orpheus_tts.py` (new) — generation
An **inference** lane, following the `tram_motion` / `longcat_avatar` precedent: the `Trainer`
contract as a job contract, one "step" per prompt, checkpoint = a manifest.

Runs on the **generic** image (hence `snac` above). Writes WAV with the stdlib `wave` module
specifically so `libsndfile` never has to be apt-installed. `keep_valid_frames()` drops frames
with any token outside its 4096-wide codebook band and **reports the count** — a sampled model
can emit one, and subtracting the offset would otherwise yield noise or a crash. Zero valid
frames **raises**, because a silent empty wav is the "trained on garbage" signature.

`reference_dataset` / `reference_indices` prepend real turns as prior context (few-shot cloning
on top of the LoRA). **Measured as no help** for similarity and it tripled WER — kept anyway,
because the mechanism is sound and the measurement is the valuable part.

### Verification, whole-suite
`python -m modallabs.tests.smoke` → **ran=5 skipped=13 failed=1**.

**The 1 failure is PRE-EXISTING and unrelated.** "crash isolation: subsequent run failed"
needs `torch`, which is not installed in `.venv` (13 skips are the same cause). Confirmed it
cannot be mine: `_run_case` calls `train_one` directly, never `concurrent_train.run()`, which
is the only function I touched. It failed identically before my first edit.

Real-world: **13 Modal runs, 0 failures**, across both new lanes.

---

## 2. Recreation recipes for anything in session scratch

**Re-read the HF secret's key names** (the smpl technique — key names only, never values):

```python
# probe.py ; then: modal run probe.py
import modal
app = modal.App("hf-secret-probe")
image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub")

@app.function(image=image, secrets=[modal.Secret.from_name("huggingface-secret")],
              timeout=120, cpu=0.25)
def probe():
    import os
    for k, v in sorted(os.environ.items()):          # SHAPE ONLY, never the value
        if any(s in k.upper() for s in ("HF", "HUGGING", "TOKEN")):
            print(f"{k} = <{len(v)} chars, starts {v[:3]!r}>")
    from huggingface_hub import whoami
    w = whoami(token=os.environ["HF_TOKEN"])
    print({"name": w.get("name"), "orgs": [o.get("name") for o in w.get("orgs") or []],
           "role": ((w.get("auth") or {}).get("accessToken") or {}).get("role")})

@app.local_entrypoint()
def main(): probe.remote()
```

Result on 2026-08-11: one key `HF_TOKEN` (37 chars, `hf_`); user **`oznaru`**, orgs
`["Berkeley"]`, **role `read`**. So `push_to_hub` 403s under any namespace until a
write-scoped token replaces it. `wmolyneaux` is the **GitHub** handle, not the HF one.

**Re-verify the RAM bound / credentials** — both test scripts are in session scratch and are
short enough to rewrite from §1's bullet lists; the assertions are stated there.

---

## 3. Open items — ranked, each with its next action

| # | Item | Next action |
|---|---|---|
| 1 | **Everything is uncommitted, in one copy** | `git add` + commit as **two** commits per §0. Will declined a push; repo is public, so ask before pushing. |
| 2 | Other configs still gate at `short` | Adopt `max_runtime_sec: 600` wherever a runtime has actually been measured. `wan_vace_shot` is ~1250 s measured, so it stays `short`. |
| 3 | Generic image gained `peft` + `snac` | **First run after this pays a rebuild** (observed: several minutes before any output on round 2). Not a bug; do not debug it as a hang. |
| 4 | Smoke suite is not green | `VIRTUAL_ENV=.venv uv pip install torch` clears the 1 failure + 13 skips. These venvs are **uv**-made and have no `pip`. |
| 5 | No inference lane for whisper | voicecraft wanted prep on Modal too; never built. The generic image already has `transformers`, so whisper large-v3 could run there without a typed lane. |
| 6 | `hostmem` / `credentials` must stay stdlib-only | Adding a third-party import to either breaks the `--no-deps` sharing that voicecraft relies on. |

---

## 4. Budget

- **This repo's own development: $0.00.** No GPU time is needed to build or test a lane —
  `stub: True` smoke cases run on CPU in ~0.1 s each, which is why PORTING.md requires one.
- **$4.44** was spent this session, all voicecraft, across 13 H100 runs ($18.61 → $23.05
  billed; **credits are exhausted, the card is paying**). Actual ran ~30% over estimate on
  every run — container startup billed beyond measured runtime. Carry that into estimates.
- **The `brief` lane is the standing saving:** any measured-short job now gates at **$0.92**
  instead of $2.75. Compounding on fan-out — 7 runs, $6.42 instead of $19.25.
- H100 is priced at **$5.50/hr** in `_GPU_HOURLY_USD`. That table's own comment says ALWAYS
  verify against modal.com/pricing, and it is worth heeding: it disagreed with voicecraft's
  table on A100-40G by 48% ($3.10 vs $2.10). Gate on the higher number.

---

## 5. Landmines — do not reintroduce

- **Every lane is pinned to `_REMOTE_GPU = "H100"`,** and `main()` **REFUSES** a run requesting
  other silicon rather than redirecting it (Modal fixes `(gpu, timeout)` at decoration time).
  A config asking for `A100-40G` will not run, whatever the price table lists. This was a
  launch blocker discovered the hard way.
- **Never size a local pool by cores.** See `hostmem.py`'s docstring for the measured
  consequence: 12 × 3.4 GB workers, 29 GB resident on 24 GB, and a jetsam cascade in which the
  kernel killed 100+ system daemons rather than the Python that caused it.
- **Do not name a module `secrets.py`** in this repo. Flat layout → it shadows the stdlib.
- **Do not wrap `Secret.from_name` in `try/except`.** Silent degradation turns a missing
  secret into a 401 after the GPU is hot. Attach unconditionally; check at consumption.
- **`modal volume get <dir> <explicit-dest>` DESTROYS DATA** (modal 1.5.3). Given a directory
  source and a non-existent dest it writes ONE member as a single file at that path — a valid
  file, exit code 0, remaining members silently gone. Verified: 2 wavs in, 1 file out. **Safe
  form is `cwd=<dir>` with no dest argument.**
- **A token's key name says nothing about its scope.** Preflight with `whoami` before training,
  not after. See `orpheus_voice._preflight_hub_push`.
- **Check the image carries what a lane imports.** `peft` was documented in the README for
  months and absent from the image the whole time.
- Adding a lane bucket to `_LANES` requires **its own module-level `@app.function`** plus a
  `_LANE_FNS` entry; `_lane_for` sorts by cap, so a new bucket slots in by value.

### Parked, with the reason

- **`reference_dataset` conditioning in `orpheus_tts`** — implemented, measured, **no
  similarity gain and WER tripled** (0.096 → 0.288). Left in place as a mechanism; do not
  re-pitch it as a quality lever without new evidence.
- **A typed lane + own image for `orpheus_tts`** — unnecessary. `snac` is pure-python over
  torch and WAV writing needs only the stdlib, so the generic image suffices. A typed lane
  costs an image build and another timeout bucket.

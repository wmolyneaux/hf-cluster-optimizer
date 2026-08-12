"""Orpheus TTS generation — speak text in a LoRA-tuned voice, and prove it isn't noise.

An inference lane, following the precedent of `tram_motion` and `longcat_avatar`: the
Trainer contract is a job contract here, not a gradient one. `epochs: 1`, one "step" per
prompt, and the checkpoint is a manifest of what was produced.

    runs:
      - name: will_says
        type: orpheus_tts
        seed: 42
        config:
          adapter_path: /runs/voicecraft_will_v1/will_voice_lora_r16/best_checkpoint
          speaker: will
          prompts:
            - "The quick brown fox jumps over the lazy dog."
          out_dir: /runs/_tts/will_v1
        modal:
          gpu: H100
          max_runtime_sec: 1800

Runs on the generic training image: it already carries torch/transformers/peft, and `snac`
was added there rather than giving this lane its own image. WAV is written with the stdlib
`wave` module on purpose, so libsndfile never has to be apt-installed.

The de-interleave below is the exact inverse of the forward layout in
voicecraft/orpheus_tokens.py. It is duplicated here because the harness image cannot import
the voicecraft repo; `assert_bands` is what keeps the copy honest, and
voicecraft/infer/generate.py holds the same inverse for local use.
"""
from __future__ import annotations

import json
import os
import wave
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from modallabs.base import (
    Trainer, TrainerEpochResult, TrainerSetup, TrainerStepResult,
)
from modallabs.credentials import hf_token as _hf_token
from modallabs.registry import register

TOKENISER_LENGTH = 128256
END_OF_TEXT = 128009
START_OF_SPEECH = TOKENISER_LENGTH + 1
END_OF_SPEECH = TOKENISER_LENGTH + 2
START_OF_HUMAN = TOKENISER_LENGTH + 3
END_OF_HUMAN = TOKENISER_LENGTH + 4
START_OF_AI = TOKENISER_LENGTH + 5
END_OF_AI = TOKENISER_LENGTH + 6
AUDIO_TOKENS_START = TOKENISER_LENGTH + 10
CODEBOOK = 4096
FRAME = 7
SNAC_REPO = "hubertsiuzdak/snac_24khz"
SR = 24000


def deinterleave(tokens: List[int]):
    """7-per-frame Orpheus interleave -> SNAC's three codebooks. Inverse of the forward map."""
    import torch
    n = len(tokens) // FRAME
    c0 = [0] * n
    c1 = [0] * (2 * n)
    c2 = [0] * (4 * n)
    b = AUDIO_TOKENS_START
    for f in range(n):
        t = tokens[f * FRAME:(f + 1) * FRAME]
        c0[f] = t[0] - b
        c1[2 * f] = t[1] - (b + CODEBOOK)
        c2[4 * f] = t[2] - (b + 2 * CODEBOOK)
        c2[4 * f + 1] = t[3] - (b + 3 * CODEBOOK)
        c1[2 * f + 1] = t[4] - (b + 4 * CODEBOOK)
        c2[4 * f + 2] = t[5] - (b + 5 * CODEBOOK)
        c2[4 * f + 3] = t[6] - (b + 6 * CODEBOOK)
    return [torch.tensor([c0], dtype=torch.long),
            torch.tensor([c1], dtype=torch.long),
            torch.tensor([c2], dtype=torch.long)]


def keep_valid_frames(tokens: List[int]) -> tuple[List[int], int]:
    """Drop frames with any token outside its own 4096-wide band.

    A sampled model can emit a token in the wrong band; subtracting the offset would
    then yield a negative or out-of-range SNAC index and either crash the decoder or
    silently produce noise. Dropping the frame is the honest response, and the count
    of dropped frames is reported rather than swallowed.
    """
    good: List[int] = []
    dropped = 0
    for f in range(len(tokens) // FRAME):
        fr = tokens[f * FRAME:(f + 1) * FRAME]
        if all(AUDIO_TOKENS_START + k * CODEBOOK <= fr[k] < AUDIO_TOKENS_START + (k + 1) * CODEBOOK
               for k in range(FRAME)):
            good.extend(fr)
        else:
            dropped += 1
    return good, dropped


def write_wav(path: Path, samples, sr: int = SR) -> None:
    """16-bit mono PCM via the stdlib, so the image needs no libsndfile."""
    import numpy as np
    a = np.clip(np.asarray(samples, dtype="float32"), -1.0, 1.0)
    pcm = (a * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


@register("orpheus_tts")
class OrpheusTTSTrainer(Trainer):
    """Generate speech from text with a LoRA-tuned Orpheus adapter."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = dict(config)
        self.stub = bool(self.config.get("stub", False))
        self.base_model = str(self.config.get(
            "base_model", "unsloth/orpheus-3b-0.1-pretrained"))
        self.adapter_path = self.config.get("adapter_path")
        self.speaker = str(self.config.get("speaker", "will"))
        self.prompts: List[str] = list(self.config.get("prompts") or [])
        self.out_dir = Path(str(self.config.get("out_dir", "/runs/_tts")))
        self.temperature = float(self.config.get("temperature", 0.6))
        self.top_p = float(self.config.get("top_p", 0.95))
        self.repetition_penalty = float(self.config.get("repetition_penalty", 1.1))
        self.max_new_tokens = int(self.config.get("max_new_tokens", 1200))
        # Optional few-shot conditioning: prepend REAL turns of the target speaker before
        # the prompt. A training example is already a complete turn
        # ([SOH] text [EOT][EOH][SOAI][SOS] codes [EOS][EOAI]), so prepending one gives the
        # model actual audio of the voice to continue from, on top of what LoRA baked in.
        # Costs no training -- it is prompt construction -- and targets timbre, which is
        # what the LoRA was measured to be weakest at.
        self.reference_dataset = self.config.get("reference_dataset")
        self.reference_indices = list(self.config.get("reference_indices") or [])
        self._reference_ids: List[int] = []
        self.model = None
        self.tokenizer = None
        self.snac = None
        self.device = "cpu"
        self._results: List[Dict[str, Any]] = []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "OrpheusTTSTrainer":
        if config.get("stub"):
            return cls(config)
        if not config.get("prompts"):
            raise ValueError("orpheus_tts: `prompts` is required (a list of strings to speak)")
        if not config.get("adapter_path"):
            raise ValueError(
                "orpheus_tts: `adapter_path` is required -- the best_checkpoint directory "
                "written by an orpheus_voice run. Omitting it would silently generate in "
                "the BASE voice, which is the one failure that looks like success."
            )
        return cls(config)

    def setup(self, setup: TrainerSetup) -> None:
        self.device = setup.device
        if self.stub:
            return
        import torch
        from peft import PeftModel
        from snac import SNAC
        from transformers import AutoModelForCausalLM, AutoTokenizer

        adapter = Path(str(self.adapter_path))
        if not (adapter / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"orpheus_tts: no adapter_config.json under {adapter} -- that is not a "
                f"PEFT checkpoint directory"
            )
        base_txt = adapter / "base_model.txt"
        base_name = base_txt.read_text().strip() if base_txt.exists() else self.base_model

        self.tokenizer = AutoTokenizer.from_pretrained(base_name, token=_hf_token())
        base = AutoModelForCausalLM.from_pretrained(
            base_name,
            torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
            attn_implementation="sdpa", token=_hf_token(),
        )
        self.model = PeftModel.from_pretrained(base, str(adapter)).to(self.device).eval()
        self.snac = SNAC.from_pretrained(SNAC_REPO).eval().to(self.device)
        torch.manual_seed(setup.seed)
        print(f"    adapter: {adapter}", flush=True)
        print(f"    base:    {base_name}", flush=True)

        if self.reference_dataset:
            from datasets import load_from_disk
            ds_path = Path(str(self.reference_dataset))
            if not ds_path.exists():
                raise FileNotFoundError(
                    f"orpheus_tts: reference_dataset {ds_path} not found -- it should be "
                    f"the same tokenized dataset the adapter trained on"
                )
            ds = load_from_disk(str(ds_path))
            idxs = self.reference_indices or [0]
            bad = [i for i in idxs if not (0 <= int(i) < len(ds))]
            if bad:
                raise IndexError(
                    f"orpheus_tts: reference_indices {bad} out of range for a dataset of "
                    f"{len(ds)} examples"
                )
            for i in idxs:
                self._reference_ids.extend(list(ds[int(i)]["input_ids"]))
            # Every prepended token eats context AND generation budget, so say how much.
            print(f"    reference conditioning: {len(idxs)} real turn(s), "
                  f"{len(self._reference_ids)} tokens prepended", flush=True)

    def train_iter(self) -> Iterable[Any]:
        return iter(list(enumerate(self.prompts)) if not self.stub else [(0, "stub")])

    def eval_iter(self) -> Iterable[Any]:
        return iter([])

    def train_step(self, batch: Any) -> TrainerStepResult:
        idx, text = batch
        if self.stub:
            self._results.append({"index": idx, "text": text, "stub": True})
            return TrainerStepResult(metrics={"frames": 0.0}, n_examples=1)

        import torch
        prompt = self.tokenizer.encode(f"{self.speaker}: {text}", add_special_tokens=True)
        ids = (list(self._reference_ids)
               + [START_OF_HUMAN] + prompt + [END_OF_TEXT] + [END_OF_HUMAN]
               + [START_OF_AI] + [START_OF_SPEECH])
        inp = torch.tensor([ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            out = self.model.generate(
                input_ids=inp,
                attention_mask=torch.ones_like(inp),
                max_new_tokens=self.max_new_tokens,
                do_sample=True, temperature=self.temperature, top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                eos_token_id=END_OF_SPEECH,
                pad_token_id=TOKENISER_LENGTH + 7,
            )
        gen = out[0][inp.shape[1]:].tolist()
        # Cut at end-of-speech, then trim to a whole number of 7-token frames.
        if END_OF_SPEECH in gen:
            gen = gen[:gen.index(END_OF_SPEECH)]
        gen = gen[:len(gen) - (len(gen) % FRAME)]

        kept, dropped = keep_valid_frames(gen)
        n_frames = len(kept) // FRAME
        if n_frames == 0:
            # Loud, not a silent empty wav: this is the "trained on garbage" signature.
            raise RuntimeError(
                f"orpheus_tts: prompt {idx} produced 0 valid frames out of "
                f"{len(gen)//FRAME} generated -- every frame had a token outside its "
                f"codebook band. The adapter or the token layout is wrong."
            )
        with torch.no_grad():
            codes = [c.to(self.device) for c in deinterleave(kept)]
            audio = self.snac.decode(codes)
        wave_np = audio.squeeze().float().cpu().numpy()

        path = self.out_dir / f"{idx:02d}_{self.speaker}.wav"
        write_wav(path, wave_np)
        dur = len(wave_np) / SR
        peak = float(abs(wave_np).max()) if len(wave_np) else 0.0
        rms = float((wave_np.astype("float64") ** 2).mean() ** 0.5) if len(wave_np) else 0.0
        print(f"    [{idx}] {n_frames} frames -> {dur:.2f}s  peak {peak:.3f}  rms {rms:.4f}"
              f"{f'  ({dropped} frames dropped)' if dropped else ''}  -> {path}", flush=True)
        self._results.append({
            "index": idx, "text": text, "wav": str(path), "frames": n_frames,
            "frames_dropped": dropped, "duration_sec": round(dur, 3),
            "peak": round(peak, 4), "rms": round(rms, 5),
        })
        return TrainerStepResult(
            metrics={"frames": float(n_frames), "duration_sec": dur, "rms": rms},
            n_examples=1,
        )

    def eval_step(self, batch: Any) -> TrainerStepResult:
        return TrainerStepResult(metrics={}, n_examples=0)

    def epoch_summary(self, epoch: int) -> TrainerEpochResult:
        n = len(self._results)
        total = sum(float(r.get("duration_sec", 0.0)) for r in self._results)
        return TrainerEpochResult(
            train_metrics={"clips": float(n), "total_sec": total},
            val_metrics={}, is_best=True, monitor_value=total,
        )

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        target = path.with_suffix("") if path.suffix in (
            ".pt", ".pth", ".joblib", ".json", ".safetensors") else path
        target.mkdir(parents=True, exist_ok=True)
        (target / "generated.json").write_text(
            json.dumps({"adapter": str(self.adapter_path), "speaker": self.speaker,
                        "results": self._results}, indent=2), encoding="utf-8")

    def load_checkpoint(self, path: Path) -> None:
        raise NotImplementedError("orpheus_tts produces audio; there is nothing to resume")

    def teardown(self) -> None:
        self.model = None
        self.snac = None

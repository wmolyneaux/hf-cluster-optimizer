"""Orpheus 3B LoRA voice clone — trains on pre-tokenized SNAC+text ids for one speaker.

Ported from voicecraft/modal_app/train.py, which mirrored Canopy Labs' finetune/train.py.
This lane exists because the generic `hf_causal_lm` type tokenizes a text column, and
Orpheus does not train on text: it trains on a 7-tokens-per-frame interleave of SNAC
audio codes wrapped in Orpheus's own control tokens. That layout is the load-bearing
part -- it fails silently when wrong -- so the ids are built once by
voicecraft/prep/tokenize_orpheus.py, validated there against the codebook bands, and
consumed here verbatim. This trainer never re-derives them.

Dataset: a `datasets.save_to_disk` directory with `input_ids`, `labels`,
`attention_mask` columns (lists of ints, variable length).

Recommended cfg for one hour of one voice (~700 examples, ~450-token sequences):

    runs:
      - name: will_voice
        type: orpheus_voice
        seed: 42
        config:
          data_path: /vol/orpheus_tokenized
          lora_r: 16
          epochs: 3
          batch_size: 1
          lr: 5e-5
          push_to_hub: wmolyneaux/voicecraft-will-v1
          hub_private: true
        modal:
          gpu: A100-40G          # bf16 3B + LoRA; T4 has neither bf16 nor the VRAM
          max_runtime_sec: 5400

Deliberate deviations from Canopy's trainer, each forced by this environment:
  * LoRA via peft        -- a full fine-tune of 3B for one voice costs ~10x for no
                            named benefit; revisit only on measured evidence that
                            LoRA underfits timbre.
  * attn sdpa            -- theirs hardcodes flash_attention_2, a long CUDA compile
                            at image-build time with no benefit at ~450 tokens.
  * report_to none       -- theirs calls wandb.init(), which blocks on a login prompt.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from modallabs.base import (
    Trainer, TrainerEpochResult, TrainerSetup, TrainerStepResult,
)
from modallabs.credentials import (
    HF_SECRET_NAME, create_hint, hf_token as _hf_token, require_hf_token,
)
from modallabs.registry import register


# Orpheus pad token. Transcribed from Canopy's preprocessing notebook and asserted
# against the same constant in voicecraft/prep/tokenize_orpheus.py.
PAD_TOKEN = 128263

DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]


@register("orpheus_voice")
class OrpheusVoiceTrainer(Trainer):
    """LoRA fine-tune of Orpheus 3B on one speaker's pre-tokenized SNAC ids."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = dict(config)
        self.stub = bool(self.config.get("stub", False))
        self.base_model = str(self.config.get(
            "base_model", "unsloth/orpheus-3b-0.1-pretrained"))
        self.lora_r = int(self.config.get("lora_r", 16))
        self.batch_size = int(self.config.get("batch_size", 1))
        self.lr = float(self.config.get("lr", 5e-5))
        self.val_frac = float(self.config.get("val_frac", 0.05))
        self.pad_token = int(self.config.get("pad_token", PAD_TOKEN))
        self.push_to_hub = self.config.get("push_to_hub") or None
        self.hub_private = bool(self.config.get("hub_private", True))
        self.model = None
        self.tokenizer = None
        self.opt = None
        self.device = "cpu"
        self._best: Optional[float] = None
        self._train_buf: List[Dict[str, float]] = []
        self._eval_buf: List[Dict[str, float]] = []
        self._pushed_from: Optional[Path] = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "OrpheusVoiceTrainer":
        # Fail before setup allocates GPU: failing at minute 0 is free.
        if not config.get("stub") and not config.get("data_path"):
            raise ValueError(
                "orpheus_voice: `data_path` is required -- the datasets.save_to_disk "
                "directory written by voicecraft/prep/tokenize_orpheus.py"
            )
        r = int(config.get("lora_r", 16))
        if r <= 0:
            raise ValueError(f"orpheus_voice: lora_r must be positive, got {r}")
        if config.get("push_to_hub") and not str(config["push_to_hub"]).count("/") == 1:
            raise ValueError(
                f"orpheus_voice: push_to_hub must be 'owner/name', got "
                f"{config['push_to_hub']!r}"
            )
        return cls(config)

    # -- setup ---------------------------------------------------------------

    def _preflight_hub_push(self) -> None:
        """Prove the token can actually write `push_to_hub` BEFORE training starts.

        A token's key name says nothing about its scope. The token in
        `huggingface-secret` on 2026-08-11 was role="read", so the first run trained for
        216 s and then 403'd on the push -- the adapter survived only because teardown
        warns instead of raising. Checking here costs one HTTPS call at minute 0 and
        turns that into a free, actionable failure.
        """
        from huggingface_hub import whoami

        token = require_hf_token(f"push_to_hub={self.push_to_hub!r}")
        try:
            who = whoami(token=token)
        except Exception as exc:
            raise RuntimeError(
                f"orpheus_voice: could not verify the HuggingFace token from Secret "
                f"{HF_SECRET_NAME} ({type(exc).__name__}: {exc}). Repair it with:\n"
                f"{create_hint(HF_SECRET_NAME, ['HF_TOKEN'])}"
            ) from exc

        role = ((who.get("auth") or {}).get("accessToken") or {}).get("role")
        user = who.get("name")
        orgs = [o.get("name") for o in (who.get("orgs") or [])]
        owner = str(self.push_to_hub).split("/", 1)[0]

        if role == "read":
            raise RuntimeError(
                f"orpheus_voice: the token in Secret {HF_SECRET_NAME} is READ-ONLY "
                f"(user {user!r}, role {role!r}); push_to_hub={self.push_to_hub!r} would "
                f"403 after training. Mint a write-scoped token at "
                f"https://huggingface.co/settings/tokens, then:\n"
                f"{create_hint(HF_SECRET_NAME, ['HF_TOKEN'])}\n"
                f"      Or drop `push_to_hub` from the cfg -- the adapter is written to "
                f"the runs volume either way."
            )
        if owner != user and owner not in orgs:
            raise RuntimeError(
                f"orpheus_voice: push_to_hub={self.push_to_hub!r} targets namespace "
                f"{owner!r}, but this token belongs to {user!r} "
                f"(orgs: {orgs or 'none'}). Creating a repo there will 403. Use "
                f"{user}/<name> or one of the orgs."
            )
        print(f"    hub preflight OK: {user!r} role={role!r} may write {owner!r}",
              flush=True)

    def setup(self, setup: TrainerSetup) -> None:
        self.device = setup.device
        if self.push_to_hub and not self.stub:
            self._preflight_hub_push()
        if self.stub:
            # Smoke path: exercises the contract -- batching, steps, checkpoint,
            # done sentinel -- without weights, a GPU, or a network call.
            self._train_batches = [[{"input_ids": [1, 2, 3], "labels": [1, 2, 3],
                                     "attention_mask": [1, 1, 1]}]]
            self._val_batches = list(self._train_batches)
            return

        import torch
        from datasets import load_from_disk
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        data_path = Path(str(self.config["data_path"]))
        if not data_path.exists():
            raise FileNotFoundError(
                f"orpheus_voice: dataset not found at {data_path}. Upload the output "
                f"of prep/tokenize_orpheus.py to the volume first."
            )
        ds = load_from_disk(str(data_path))
        if len(ds) == 0:
            raise RuntimeError(f"orpheus_voice: {data_path} holds 0 examples")
        missing = {"input_ids", "labels", "attention_mask"} - set(ds.column_names)
        if missing:
            raise RuntimeError(
                f"orpheus_voice: {data_path} is missing column(s) {sorted(missing)}; "
                f"got {ds.column_names}. This is not a tokenize_orpheus.py dataset."
            )

        lens = sorted(len(x) for x in ds["input_ids"])
        print(f"    dataset: {len(ds)} examples, seq len median "
              f"{lens[len(lens)//2]}, max {lens[-1]}", flush=True)

        ds = ds.shuffle(seed=setup.seed)
        n_val = max(1, int(len(ds) * self.val_frac)) if len(ds) > 1 else 0
        val_rows = [ds[i] for i in range(n_val)]
        train_rows = [ds[i] for i in range(n_val, len(ds))]
        if not train_rows:
            raise RuntimeError(
                f"orpheus_voice: val_frac={self.val_frac} left 0 training examples "
                f"out of {len(ds)}"
            )
        self._train_batches = self._batch(train_rows)
        self._val_batches = self._batch(val_rows)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model, token=_hf_token())
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
            attn_implementation="sdpa",
            token=_hf_token(),
        )
        self.model = get_peft_model(model, LoraConfig(
            r=self.lora_r,
            lora_alpha=int(self.config.get("lora_alpha", self.lora_r * 2)),
            lora_dropout=float(self.config.get("lora_dropout", 0.05)),
            bias="none", task_type="CAUSAL_LM",
            target_modules=list(self.config.get("target_modules",
                                                DEFAULT_TARGET_MODULES)),
        )).to(self.device)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"    LoRA r={self.lora_r}: {trainable/1e6:.1f}M trainable of "
              f"{total/1e9:.2f}B ({trainable/total*100:.2f}%)", flush=True)

        self.opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=self.lr)

    def _batch(self, rows: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        B = max(1, self.batch_size)
        return [rows[i:i + B] for i in range(0, len(rows), B)]

    # -- the loop ------------------------------------------------------------

    def train_iter(self) -> Iterable[Any]:
        self._train_buf.clear()
        return iter(self._train_batches)

    def eval_iter(self) -> Iterable[Any]:
        self._eval_buf.clear()
        return iter(self._val_batches)

    def _collate(self, batch: List[Dict[str, Any]]):
        """Right-pad to the batch's longest sequence; padded labels are -100 (ignored)."""
        import torch
        n = max(len(b["input_ids"]) for b in batch)
        ids, labels, mask = [], [], []
        for b in batch:
            p = n - len(b["input_ids"])
            ids.append(list(b["input_ids"]) + [self.pad_token] * p)
            labels.append(list(b["labels"]) + [-100] * p)
            mask.append(list(b["attention_mask"]) + [0] * p)
        t = lambda x: torch.tensor(x, device=self.device)  # noqa: E731
        return {"input_ids": t(ids), "labels": t(labels), "attention_mask": t(mask)}

    def train_step(self, batch: Any) -> TrainerStepResult:
        if self.stub:
            m = {"loss": 0.0}
            self._train_buf.append(m)
            return TrainerStepResult(metrics=m, n_examples=len(batch))
        self.model.train()
        out = self.model(**self._collate(batch))
        out.loss.backward()
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        m = {"loss": float(out.loss.item())}
        self._train_buf.append(m)
        return TrainerStepResult(metrics=m, n_examples=len(batch))

    def eval_step(self, batch: Any) -> TrainerStepResult:
        if self.stub:
            m = {"loss": 0.0}
            self._eval_buf.append(m)
            return TrainerStepResult(metrics=m, n_examples=len(batch))
        import torch
        self.model.eval()
        with torch.no_grad():
            out = self.model(**self._collate(batch))
        m = {"loss": float(out.loss.item())}
        self._eval_buf.append(m)
        return TrainerStepResult(metrics=m, n_examples=len(batch))

    def epoch_summary(self, epoch: int) -> TrainerEpochResult:
        mean = lambda buf, k: (sum(d[k] for d in buf) / len(buf)) if buf else 0.0  # noqa: E731
        train_m = {"loss": mean(self._train_buf, "loss")}
        val_m = {"loss": mean(self._eval_buf, "loss")}
        # Lower val loss is better, so the monitor is negated: the framework's
        # is_best contract is "higher monitor_value wins".
        monitor = -val_m["loss"]
        is_best = self._best is None or monitor > self._best
        if is_best:
            self._best = monitor
        return TrainerEpochResult(
            train_metrics=train_m, val_metrics=val_m,
            is_best=is_best, monitor_value=monitor,
        )

    # -- checkpoints ---------------------------------------------------------

    def save_checkpoint(self, path: Path) -> None:
        """Write the LoRA adapter (tens of MB, not the 3B base) plus the tokenizer.

        Saved as an ordinary save_pretrained directory, so reloading for TTS is
        `PeftModel.from_pretrained(base, <this dir>)` with no bespoke unpacking.
        The framework mirrors it to the modallabs-runs volume; the Hub push is
        deferred to teardown so it costs no GPU seconds mid-training.
        """
        path = Path(path)
        target = path.with_suffix("") if path.suffix in (
            ".pt", ".pth", ".joblib", ".json", ".safetensors") else path
        target.mkdir(parents=True, exist_ok=True)
        if self.stub:
            (target / "adapter_config.json").write_text('{"stub": true}',
                                                        encoding="utf-8")
            return
        self.model.save_pretrained(target)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(target)
        # Record the base model alongside the adapter: an adapter without the
        # checkpoint it was trained against is not reloadable.
        (target / "base_model.txt").write_text(self.base_model + "\n", encoding="utf-8")
        self._pushed_from = target

    def load_checkpoint(self, path: Path) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        target = Path(path)
        if target.suffix in (".pt", ".pth", ".joblib", ".json", ".safetensors"):
            target = target.with_suffix("")
        base_txt = target / "base_model.txt"
        base_name = base_txt.read_text().strip() if base_txt.exists() else self.base_model
        base = AutoModelForCausalLM.from_pretrained(
            base_name,
            torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
            attn_implementation="sdpa", token=_hf_token(),
        )
        self.model = PeftModel.from_pretrained(base, str(target)).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(str(target))

    def teardown(self) -> None:
        """Push the adapter to the Hub, then drop refs so empty_cache can reclaim."""
        if self.push_to_hub and self._pushed_from and not self.stub:
            # Deliberately here and not in save_checkpoint: this uploads on every
            # `is_best` epoch otherwise, and every upload second is billed at GPU
            # rates. One push of the final adapter (tens of MB) is seconds.
            try:
                self.model.push_to_hub(str(self.push_to_hub), private=self.hub_private,
                                       token=_hf_token())
                if self.tokenizer is not None:
                    self.tokenizer.push_to_hub(str(self.push_to_hub),
                                               private=self.hub_private,
                                               token=_hf_token())
                print(f"    pushed adapter -> hf.co/{self.push_to_hub} "
                      f"(private={self.hub_private})", flush=True)
            except Exception as exc:
                # Never lose a finished training run to a Hub outage: the adapter is
                # already on the volume. Report loudly, do not raise. Scope and namespace
                # were checked in setup(), so reaching here means something transient --
                # or something the preflight cannot see.
                print(f"    WARNING: push_to_hub failed ({type(exc).__name__}: {exc}). "
                      f"Adapter is safe at {self._pushed_from}; push it by hand with "
                      f"`huggingface-cli upload {self.push_to_hub} {self._pushed_from}`. "
                      f"If this is a 401/403, the token in Secret {HF_SECRET_NAME} lacks "
                      f"write scope for that namespace:\n"
                      f"{create_hint(HF_SECRET_NAME, ['HF_TOKEN'])}",
                      flush=True)
        self.model = None
        self.opt = None

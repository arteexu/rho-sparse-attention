#!/usr/bin/env python3
"""Real-data RHO-1 style training run with dense and MoE transformers.

This script wires the RHO-1 mechanics from ``rho_moe_experiment.py`` into real
text datasets loaded with Hugging Face Datasets:

1. Train a reference LM on desired-domain text.
2. Build a real pretraining mixture from a general corpus plus optional
   desired-domain examples.
3. Train CLM and SLM variants.
4. Compare dense and MoE transformer variants.

Defaults are deliberately small: GSM8K is used as desired-domain text and
WikiText-2 as general pretraining/eval text. The model is small and randomly
initialized, so this is a capability/prototype run rather than a competitive
pretraining recipe.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from rho_moe_experiment import ExperimentConfig, MiniTransformerLM, resolve_device, set_seed, topk_mask, write_csv


GENERAL_SOURCE = 0
DESIRED_SOURCE = 1
SELECTION_STRATEGIES = {"clm", "slm", "random", "current_loss", "ref_low"}


@dataclass
class RealRunConfig:
    seed: int = 11
    tokenizer_name: str = "wordlevel"
    max_vocab_size: int = 4096
    seq_len: int = 64
    batch_size: int = 8
    ref_steps: int = 40
    steps: int = 80
    eval_every: int = 20
    lr: float = 3e-4
    weight_decay: float = 0.01
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 128
    dropout: float = 0.0
    n_experts: int = 4
    moe_top_k: int = 2
    moe_aux_weight: float = 0.05
    select_ratio: float = 0.6
    select_ratio_schedule: str = ""
    slm_warmup_steps: int = 10
    save_every: int = 100
    resume_dir: str = ""
    reference_dataset: str = "openai/gsm8k"
    reference_config: str = "main"
    reference_train_split: str = "train"
    reference_eval_split: str = "test"
    reference_text_fields: str = "question,answer"
    general_dataset: str = "Salesforce/wikitext"
    general_config: str = "wikitext-2-raw-v1"
    general_train_split: str = "train"
    general_eval_split: str = "validation"
    general_text_fields: str = "text"
    max_reference_rows: int = 512
    max_general_rows: int = 1024
    max_eval_rows: int = 256
    max_train_sequences: int = 512
    max_ref_sequences: int = 256
    max_eval_sequences: int = 128
    desired_mix_fraction: float = 0.35
    run: str = "all"
    device: str = "auto"
    out_dir: str = "runs"


PRESETS: dict[str, dict[str, object]] = {
    "smoke": {
        "seq_len": 48,
        "batch_size": 4,
        "ref_steps": 8,
        "steps": 12,
        "eval_every": 6,
        "d_model": 48,
        "n_layers": 1,
        "n_heads": 4,
        "d_ff": 96,
        "max_reference_rows": 96,
        "max_general_rows": 192,
        "max_eval_rows": 64,
        "max_train_sequences": 96,
        "max_ref_sequences": 48,
        "max_eval_sequences": 32,
    },
    "small": {},
    "medium": {
        "max_vocab_size": 8192,
        "seq_len": 128,
        "batch_size": 8,
        "ref_steps": 200,
        "steps": 800,
        "eval_every": 50,
        "save_every": 100,
        "d_model": 128,
        "n_layers": 4,
        "n_heads": 4,
        "d_ff": 512,
        "n_experts": 4,
        "moe_top_k": 2,
        "max_reference_rows": 2048,
        "max_general_rows": 4096,
        "max_eval_rows": 512,
        "max_train_sequences": 4096,
        "max_ref_sequences": 1024,
        "max_eval_sequences": 256,
    },
    "large_local": {
        "max_vocab_size": 12000,
        "seq_len": 128,
        "batch_size": 8,
        "ref_steps": 600,
        "steps": 2500,
        "eval_every": 100,
        "save_every": 100,
        "lr": 2e-4,
        "d_model": 256,
        "n_layers": 6,
        "n_heads": 8,
        "d_ff": 1024,
        "n_experts": 4,
        "moe_top_k": 2,
        "max_reference_rows": 7473,
        "max_general_rows": 20000,
        "max_eval_rows": 1024,
        "max_train_sequences": 12000,
        "max_ref_sequences": 4096,
        "max_eval_sequences": 512,
    },
}


class TokenizerLike(Protocol):
    eos_token_id: int | None
    pad_token_id: int | None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ...

    def __len__(self) -> int:
        ...


class WordLevelTokenizer:
    pad_token = "<pad>"
    unk_token = "<unk>"
    eos_token = "<eos>"

    def __init__(self, texts: list[str], max_vocab_size: int):
        self.token_pattern = re.compile(r"\d+|[A-Za-z]+|[^\sA-Za-z\d]", re.ASCII)
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(self._tokenize(text))
        vocab = [self.pad_token, self.unk_token, self.eos_token]
        for token, _count in counts.most_common(max(0, max_vocab_size - len(vocab))):
            if token not in vocab:
                vocab.append(token)
        self.itos = vocab
        self.stoi = {token: idx for idx, token in enumerate(vocab)}
        self.pad_token_id = self.stoi[self.pad_token]
        self.unk_token_id = self.stoi[self.unk_token]
        self.eos_token_id = self.stoi[self.eos_token]

    def _tokenize(self, text: str) -> list[str]:
        return [match.group(0).lower() for match in self.token_pattern.finditer(text)]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self.stoi.get(token, self.unk_token_id) for token in self._tokenize(text)]

    def __len__(self) -> int:
        return len(self.itos)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.itos, indent=2))


def text_from_row(row: dict, fields: list[str]) -> str:
    parts = []
    for field in fields:
        value = row.get(field)
        if value is not None:
            parts.append(str(value).strip())
    return "\n".join(part for part in parts if part)


def load_text_rows(dataset_name: str, config: str, split: str, fields: str, limit: int) -> list[str]:
    ds = load_dataset(dataset_name, config, split=split) if config else load_dataset(dataset_name, split=split)
    field_list = [field.strip() for field in fields.split(",") if field.strip()]
    texts: list[str] = []
    for row in ds:
        text = text_from_row(row, field_list)
        if text and not text.isspace():
            texts.append(text)
        if len(texts) >= limit:
            break
    if not texts:
        raise ValueError(f"No text rows loaded from {dataset_name}/{config}/{split} with fields={fields}")
    return texts


def mix_sources(
    general_texts: list[str],
    desired_texts: list[str],
    desired_fraction: float,
    rng: random.Random,
) -> list[tuple[str, int]]:
    desired_fraction = min(max(desired_fraction, 0.0), 1.0)
    general = [(text, GENERAL_SOURCE) for text in general_texts]
    desired = [(text, DESIRED_SOURCE) for text in desired_texts]
    target_desired = int(round((len(general) + len(desired)) * desired_fraction))
    target_desired = min(target_desired, len(desired))
    target_general = min(len(general), max(len(general), len(desired) - target_desired))
    mixed = general[:target_general] + desired[:target_desired]
    rng.shuffle(mixed)
    return mixed


def pack_texts(
    tokenizer: TokenizerLike,
    rows: list[tuple[str, int]],
    seq_len: int,
    max_sequences: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_ids: list[int] = []
    all_sources: list[int] = []
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Tokenizer must define eos_token_id")
    for text, source in rows:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        ids.append(eos)
        all_ids.extend(ids)
        all_sources.extend([source] * len(ids))
        if len(all_ids) >= seq_len * max_sequences:
            break

    usable = min(len(all_ids), len(all_sources), seq_len * max_sequences)
    usable = (usable // seq_len) * seq_len
    if usable < seq_len:
        raise ValueError("Not enough tokenized text to create one training sequence")

    ids_tensor = torch.tensor(all_ids[:usable], dtype=torch.long).view(-1, seq_len)
    source_tensor = torch.tensor(all_sources[:usable], dtype=torch.long).view(-1, seq_len)
    return ids_tensor, source_tensor


def sample_batch(
    ids: torch.Tensor,
    sources: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.randint(0, ids.size(0), (batch_size,), device=torch.device("cpu"))
    return ids[idx].to(device), sources[idx].to(device)


def random_valid_mask(valid: torch.Tensor, select_ratio: float) -> torch.Tensor:
    score = torch.rand(valid.shape, device=valid.device)
    score = score.masked_fill(~valid, -float("inf"))
    flat_valid = valid.flatten()
    n_valid = int(flat_valid.sum().item())
    if n_valid == 0:
        return torch.zeros_like(valid)
    k = max(1, int(math.ceil(select_ratio * n_valid)))
    flat_score = score.flatten()
    selected_flat = torch.zeros_like(flat_valid)
    selected_idx = torch.topk(flat_score, k=k).indices
    selected_flat[selected_idx] = True
    return selected_flat.view_as(valid) & valid


def parse_select_ratio_schedule(schedule: str) -> list[tuple[int, float]]:
    if not schedule.strip():
        return []
    points: list[tuple[int, float]] = []
    for raw_part in schedule.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError("select ratio schedule entries must look like step:ratio")
        raw_step, raw_ratio = part.split(":", 1)
        step = int(raw_step)
        ratio = float(raw_ratio)
        if step < 0 or not (0.0 < ratio <= 1.0):
            raise ValueError("select ratio schedule requires step >= 0 and ratio in (0, 1]")
        points.append((step, ratio))
    return sorted(points, key=lambda item: item[0])


def select_ratio_for_step(cfg: RealRunConfig, step: int) -> float:
    ratio = cfg.select_ratio
    for start_step, scheduled_ratio in parse_select_ratio_schedule(cfg.select_ratio_schedule):
        if step >= start_step:
            ratio = scheduled_ratio
        else:
            break
    return ratio


def token_losses(logits: torch.Tensor, idx: torch.Tensor, pad_id: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    targets = idx[:, 1:]
    losses = F.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
        targets.contiguous().view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(targets)
    valid = torch.ones_like(targets, dtype=torch.bool)
    if pad_id is not None:
        valid = targets != pad_id
    return losses, valid


@torch.no_grad()
def evaluate(
    model: MiniTransformerLM,
    data: torch.Tensor,
    batch_size: int,
    pad_id: int | None,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_count = 0.0
    for start in range(0, data.size(0), batch_size):
        batch = data[start : start + batch_size].to(device)
        logits, _stats = model(batch)
        losses, valid = token_losses(logits, batch, pad_id)
        preds = logits[:, :-1, :].argmax(dim=-1)
        targets = batch[:, 1:]
        total_loss += losses[valid].sum().item()
        total_correct += (preds[valid] == targets[valid]).float().sum().item()
        total_count += valid.float().sum().item()
    loss = total_loss / max(total_count, 1.0)
    return {"loss": loss, "ppl": math.exp(min(loss, 20.0)), "acc": total_correct / max(total_count, 1.0)}


@torch.no_grad()
def selection_diagnostics(
    model: MiniTransformerLM,
    ref_model: MiniTransformerLM,
    batch: torch.Tensor,
    source_batch: torch.Tensor,
    select_ratio: float,
    pad_id: int | None,
    selection_strategy: str,
) -> dict[str, float]:
    model.eval()
    ref_model.eval()
    logits, _ = model(batch)
    ref_logits, _ = ref_model(batch)
    losses, valid = token_losses(logits, batch, pad_id)
    ref_losses, _ = token_losses(ref_logits, batch, pad_id)
    selected = build_selection_mask(losses, ref_losses, valid, select_ratio, selection_strategy)
    excess = losses - ref_losses
    unselected = valid & ~selected
    target_sources = source_batch[:, 1:]
    selected_count = max(selected.float().sum().item(), 1.0)
    unselected_count = max(unselected.float().sum().item(), 1.0)
    return {
        "diagnostic_selected_fraction": selected.float().sum().item() / max(valid.float().sum().item(), 1.0),
        "selected_model_loss": losses[selected].mean().item() if selected.any() else float("nan"),
        "unselected_model_loss": losses[unselected].mean().item() if unselected.any() else float("nan"),
        "mean_excess_selected": excess[selected].mean().item() if selected.any() else float("nan"),
        "mean_excess_unselected": excess[unselected].mean().item() if unselected.any() else float("nan"),
        "selected_desired_source_fraction": ((target_sources == DESIRED_SOURCE) & selected).float().sum().item() / selected_count,
        "unselected_desired_source_fraction": ((target_sources == DESIRED_SOURCE) & unselected).float().sum().item() / unselected_count,
    }


def build_selection_mask(
    losses: torch.Tensor,
    ref_losses: torch.Tensor,
    valid: torch.Tensor,
    select_ratio: float,
    selection_strategy: str,
) -> torch.Tensor:
    if selection_strategy == "clm":
        return valid
    if selection_strategy == "slm":
        return topk_mask(losses.detach() - ref_losses, valid, select_ratio)
    if selection_strategy == "random":
        return random_valid_mask(valid, select_ratio)
    if selection_strategy == "current_loss":
        return topk_mask(losses.detach(), valid, select_ratio)
    if selection_strategy == "ref_low":
        return topk_mask(-ref_losses, valid, select_ratio)
    raise ValueError(f"Unknown selection strategy: {selection_strategy}")


def to_model_cfg(cfg: RealRunConfig) -> ExperimentConfig:
    return ExperimentConfig(
        seed=cfg.seed,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        ref_steps=cfg.ref_steps,
        steps=cfg.steps,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        n_experts=cfg.n_experts,
        moe_top_k=cfg.moe_top_k,
        moe_aux_weight=cfg.moe_aux_weight,
        select_ratio=cfg.select_ratio,
        slm_warmup_steps=cfg.slm_warmup_steps,
        eval_every=cfg.eval_every,
        device=cfg.device,
        out_dir=cfg.out_dir,
    )


def train_reference(
    cfg: RealRunConfig,
    model_cfg: ExperimentConfig,
    vocab_size: int,
    ref_ids: torch.Tensor,
    desired_eval: torch.Tensor,
    pad_id: int | None,
    device: torch.device,
    out_dir: Path,
) -> MiniTransformerLM:
    checkpoint_path = out_dir / "reference_checkpoint.pt"
    model = MiniTransformerLM(vocab_size, model_cfg, moe=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    start_step = 1
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt["step"]) + 1
        print(f"[reference] resumed from step {ckpt['step']}")
    if start_step > cfg.ref_steps:
        return model
    for step in range(start_step, cfg.ref_steps + 1):
        batch, _sources = sample_batch(ref_ids, torch.zeros_like(ref_ids), cfg.batch_size, device)
        logits, _ = model(batch)
        losses, valid = token_losses(logits, batch, pad_id)
        loss = losses[valid].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step == cfg.ref_steps or step % max(1, cfg.ref_steps // 4) == 0:
            metrics = evaluate(model, desired_eval, cfg.batch_size, pad_id, device)
            print(f"[reference] step={step:04d} loss={loss.item():.4f} desired_eval_loss={metrics['loss']:.4f}")
        if step == cfg.ref_steps or (cfg.save_every > 0 and step % cfg.save_every == 0):
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step}, checkpoint_path)
    return model


def train_candidate(
    name: str,
    cfg: RealRunConfig,
    model_cfg: ExperimentConfig,
    vocab_size: int,
    train_ids: torch.Tensor,
    train_sources: torch.Tensor,
    desired_eval: torch.Tensor,
    general_eval: torch.Tensor,
    ref_model: MiniTransformerLM,
    pad_id: int | None,
    device: torch.device,
    moe: bool,
    selection_strategy: str,
    out_dir: Path,
) -> dict[str, float | int | str]:
    if selection_strategy not in SELECTION_STRATEGIES:
        raise ValueError(f"selection_strategy must be one of {sorted(SELECTION_STRATEGIES)}")
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{name}.pt"
    history_path = out_dir / f"{name}.csv"
    model = MiniTransformerLM(vocab_size, model_cfg, moe=moe).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rows: list[dict[str, float | int | str]] = []
    ref_model.eval()
    start_step = 1
    cumulative_total_tokens = 0
    cumulative_selected_tokens = 0
    elapsed_offset = 0.0
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt["step"]) + 1
        cumulative_total_tokens = int(ckpt.get("cumulative_total_tokens", 0))
        cumulative_selected_tokens = int(ckpt.get("cumulative_selected_tokens", 0))
        elapsed_offset = float(ckpt.get("elapsed_sec", 0.0))
        rows = list(ckpt.get("rows", []))
        print(f"[{name}] resumed from step {ckpt['step']}")
    if start_step > cfg.steps and rows:
        final = rows[-1].copy()
        final["name"] = name
        return final
    start_time = time.perf_counter()

    for step in range(start_step, cfg.steps + 1):
        model.train()
        batch, source_batch = sample_batch(train_ids, train_sources, cfg.batch_size, device)
        logits, stats = model(batch)
        losses, valid = token_losses(logits, batch, pad_id)

        valid_tokens = int(valid.float().sum().item())
        selected_fraction = 1.0
        active_select_ratio = select_ratio_for_step(cfg, step)
        if selection_strategy != "clm" and step > cfg.slm_warmup_steps and active_select_ratio < 0.999:
            with torch.no_grad():
                ref_logits, _ = ref_model(batch)
                ref_losses, _ = token_losses(ref_logits, batch, pad_id)
                selected = build_selection_mask(losses, ref_losses, valid, active_select_ratio, selection_strategy)
            main_loss = losses[selected].mean()
            selected_tokens = int(selected.float().sum().item())
            selected_fraction = selected_tokens / max(valid_tokens, 1)
        else:
            main_loss = losses[valid].mean()
            selected_tokens = valid_tokens

        aux_loss = stats.get("load_balance_loss", torch.tensor(0.0, device=device))
        loss = main_loss + (cfg.moe_aux_weight * aux_loss if moe else 0.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        cumulative_total_tokens += valid_tokens
        cumulative_selected_tokens += selected_tokens

        if step == 1 or step % cfg.eval_every == 0 or step == cfg.steps:
            elapsed_sec = elapsed_offset + time.perf_counter() - start_time
            desired_metrics = evaluate(model, desired_eval, cfg.batch_size, pad_id, device)
            general_metrics = evaluate(model, general_eval, cfg.batch_size, pad_id, device)
            diag = (
                selection_diagnostics(model, ref_model, batch, source_batch, active_select_ratio, pad_id, selection_strategy)
                if selection_strategy != "clm" and active_select_ratio < 0.999
                else {}
            )
            row: dict[str, float | int | str] = {
                "step": step,
                "selection_strategy": selection_strategy,
                "active_select_ratio": active_select_ratio,
                "train_loss": loss.item(),
                "main_loss": main_loss.item(),
                "selected_fraction": selected_fraction,
                "step_valid_tokens": valid_tokens,
                "step_selected_tokens": selected_tokens,
                "cumulative_total_tokens": cumulative_total_tokens,
                "cumulative_selected_tokens": cumulative_selected_tokens,
                "elapsed_sec": elapsed_sec,
                "total_tokens_per_sec": cumulative_total_tokens / max(elapsed_sec, 1e-9),
                "selected_tokens_per_sec": cumulative_selected_tokens / max(elapsed_sec, 1e-9),
                "desired_loss": desired_metrics["loss"],
                "desired_ppl": desired_metrics["ppl"],
                "desired_acc": desired_metrics["acc"],
                "general_loss": general_metrics["loss"],
                "general_ppl": general_metrics["ppl"],
                "general_acc": general_metrics["acc"],
                "load_balance_loss": aux_loss.item() if moe else 0.0,
                "router_entropy": stats.get("router_entropy", torch.tensor(0.0)).item() if moe else 0.0,
            }
            if moe and "expert_fractions" in stats:
                for i, frac in enumerate(stats["expert_fractions"].detach().cpu().tolist()):
                    row[f"expert_fraction_{i}"] = float(frac)
            row.update(diag)
            rows.append(row)
            print(
                f"[{name}] step={step:04d} loss={loss.item():.4f} sel={selected_fraction:.2f} "
                f"desired_loss={desired_metrics['loss']:.4f} general_loss={general_metrics['loss']:.4f}"
            )
        if step == cfg.steps or (cfg.save_every > 0 and step % cfg.save_every == 0):
            elapsed_sec = elapsed_offset + time.perf_counter() - start_time
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "step": step,
                    "cumulative_total_tokens": cumulative_total_tokens,
                    "cumulative_selected_tokens": cumulative_selected_tokens,
                    "elapsed_sec": elapsed_sec,
                    "rows": rows,
                },
                checkpoint_path,
            )

    write_csv(history_path, rows)
    final = rows[-1].copy()
    final["name"] = name
    return final


def write_text_samples(path: Path, title: str, texts: list[str], limit: int = 3) -> None:
    with path.open("w") as f:
        f.write(f"# {title}\n\n")
        for i, text in enumerate(texts[:limit], start=1):
            f.write(f"## Sample {i}\n{text[:2000]}\n\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="small")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--tokenizer-name", dest="tokenizer_name")
    parser.add_argument("--max-vocab-size", dest="max_vocab_size", type=int)
    parser.add_argument("--seq-len", dest="seq_len", type=int)
    parser.add_argument("--batch-size", dest="batch_size", type=int)
    parser.add_argument("--ref-steps", dest="ref_steps", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--eval-every", dest="eval_every", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--d-model", dest="d_model", type=int)
    parser.add_argument("--n-layers", dest="n_layers", type=int)
    parser.add_argument("--n-heads", dest="n_heads", type=int)
    parser.add_argument("--d-ff", dest="d_ff", type=int)
    parser.add_argument("--n-experts", dest="n_experts", type=int)
    parser.add_argument("--moe-top-k", dest="moe_top_k", type=int)
    parser.add_argument("--select-ratio", dest="select_ratio", type=float)
    parser.add_argument("--select-ratio-schedule", dest="select_ratio_schedule")
    parser.add_argument("--slm-warmup-steps", dest="slm_warmup_steps", type=int)
    parser.add_argument("--save-every", dest="save_every", type=int)
    parser.add_argument("--resume-dir", dest="resume_dir")
    parser.add_argument("--desired-mix-fraction", dest="desired_mix_fraction", type=float)
    parser.add_argument("--max-reference-rows", dest="max_reference_rows", type=int)
    parser.add_argument("--max-general-rows", dest="max_general_rows", type=int)
    parser.add_argument("--max-eval-rows", dest="max_eval_rows", type=int)
    parser.add_argument("--max-train-sequences", dest="max_train_sequences", type=int)
    parser.add_argument("--max-ref-sequences", dest="max_ref_sequences", type=int)
    parser.add_argument("--max-eval-sequences", dest="max_eval_sequences", type=int)
    parser.add_argument("--reference-dataset", dest="reference_dataset")
    parser.add_argument("--reference-config", dest="reference_config")
    parser.add_argument("--reference-train-split", dest="reference_train_split")
    parser.add_argument("--reference-eval-split", dest="reference_eval_split")
    parser.add_argument("--reference-text-fields", dest="reference_text_fields")
    parser.add_argument("--general-dataset", dest="general_dataset")
    parser.add_argument("--general-config", dest="general_config")
    parser.add_argument("--general-train-split", dest="general_train_split")
    parser.add_argument("--general-eval-split", dest="general_eval_split")
    parser.add_argument("--general-text-fields", dest="general_text_fields")
    parser.add_argument("--run", choices=["all", "dense", "moe", "dense_slm", "moe_slm", "efficiency_dense", "efficiency_moe", "efficiency"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--out-dir", dest="out_dir")
    return parser.parse_args()


def apply_args(cfg: RealRunConfig, args: argparse.Namespace) -> RealRunConfig:
    for key, value in vars(args).items():
        if value is not None and hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def main() -> None:
    args = parse_args()
    cfg = RealRunConfig()
    for key, value in PRESETS[args.preset].items():
        setattr(cfg, key, value)
    cfg = apply_args(cfg, args)
    if not (0.0 < cfg.select_ratio <= 1.0):
        raise ValueError("--select-ratio must be in (0, 1]")
    if not (0.0 <= cfg.desired_mix_fraction <= 1.0):
        raise ValueError("--desired-mix-fraction must be in [0, 1]")

    set_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"Using device: {device}")

    print("Loading real datasets...")
    reference_train_texts = load_text_rows(
        cfg.reference_dataset,
        cfg.reference_config,
        cfg.reference_train_split,
        cfg.reference_text_fields,
        cfg.max_reference_rows,
    )
    reference_eval_texts = load_text_rows(
        cfg.reference_dataset,
        cfg.reference_config,
        cfg.reference_eval_split,
        cfg.reference_text_fields,
        cfg.max_eval_rows,
    )
    general_train_texts = load_text_rows(
        cfg.general_dataset,
        cfg.general_config,
        cfg.general_train_split,
        cfg.general_text_fields,
        cfg.max_general_rows,
    )
    general_eval_texts = load_text_rows(
        cfg.general_dataset,
        cfg.general_config,
        cfg.general_eval_split,
        cfg.general_text_fields,
        cfg.max_eval_rows,
    )

    if cfg.tokenizer_name == "wordlevel":
        tokenizer: TokenizerLike = WordLevelTokenizer(
            reference_train_texts + reference_eval_texts + general_train_texts + general_eval_texts,
            cfg.max_vocab_size,
        )
    else:
        hf_tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        if hf_tokenizer.pad_token_id is None:
            hf_tokenizer.pad_token = hf_tokenizer.eos_token
        tokenizer = hf_tokenizer
    pad_id = tokenizer.pad_token_id

    train_rows = mix_sources(general_train_texts, reference_train_texts, cfg.desired_mix_fraction, rng)
    ref_rows = [(text, DESIRED_SOURCE) for text in reference_train_texts]
    desired_eval_rows = [(text, DESIRED_SOURCE) for text in reference_eval_texts]
    general_eval_rows = [(text, GENERAL_SOURCE) for text in general_eval_texts]

    ref_ids, ref_sources = pack_texts(tokenizer, ref_rows, cfg.seq_len, cfg.max_ref_sequences)
    train_ids, train_sources = pack_texts(tokenizer, train_rows, cfg.seq_len, cfg.max_train_sequences)
    desired_eval_ids, _ = pack_texts(tokenizer, desired_eval_rows, cfg.seq_len, cfg.max_eval_sequences)
    general_eval_ids, _ = pack_texts(tokenizer, general_eval_rows, cfg.seq_len, cfg.max_eval_sequences)

    if cfg.resume_dir:
        out_dir = Path(cfg.resume_dir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path(cfg.out_dir) / f"rho_moe_real_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    if isinstance(tokenizer, WordLevelTokenizer):
        tokenizer.save(out_dir / "wordlevel_vocab.json")
    write_text_samples(out_dir / "reference_samples.md", "Reference Desired-Domain Samples", reference_train_texts)
    write_text_samples(out_dir / "general_samples.md", "General Pretraining Samples", general_train_texts)

    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Packed ref/train/desired_eval/general_eval sequences: {len(ref_ids)}/{len(train_ids)}/{len(desired_eval_ids)}/{len(general_eval_ids)}")
    print(f"Output dir: {out_dir}")

    model_cfg = to_model_cfg(cfg)
    ref_model = train_reference(cfg, model_cfg, len(tokenizer), ref_ids, desired_eval_ids, pad_id, device, out_dir)

    run_specs = {
        "all": [("dense_clm", False, "clm"), ("dense_slm", False, "slm"), ("moe_clm", True, "clm"), ("moe_slm", True, "slm")],
        "dense": [("dense_clm", False, "clm"), ("dense_slm", False, "slm")],
        "moe": [("moe_clm", True, "clm"), ("moe_slm", True, "slm")],
        "dense_slm": [("dense_slm", False, "slm")],
        "moe_slm": [("moe_slm", True, "slm")],
        "efficiency_dense": [
            ("dense_clm", False, "clm"),
            ("dense_random", False, "random"),
            ("dense_current_loss", False, "current_loss"),
            ("dense_ref_low", False, "ref_low"),
            ("dense_slm", False, "slm"),
        ],
        "efficiency_moe": [
            ("moe_clm", True, "clm"),
            ("moe_random", True, "random"),
            ("moe_slm", True, "slm"),
        ],
        "efficiency": [
            ("dense_clm", False, "clm"),
            ("dense_random", False, "random"),
            ("dense_current_loss", False, "current_loss"),
            ("dense_ref_low", False, "ref_low"),
            ("dense_slm", False, "slm"),
            ("moe_clm", True, "clm"),
            ("moe_random", True, "random"),
            ("moe_slm", True, "slm"),
        ],
    }
    results = []
    for name, moe, selection_strategy in run_specs[cfg.run]:
        print(f"\nTraining {name}...")
        result = train_candidate(
            name,
            cfg,
            model_cfg,
            len(tokenizer),
            train_ids,
            train_sources,
            desired_eval_ids,
            general_eval_ids,
            ref_model,
            pad_id,
            device,
            moe,
            selection_strategy,
            out_dir,
        )
        results.append(result)

    write_csv(out_dir / "summary.csv", results)
    print("\nFinal summary:")
    for row in results:
        print(
            f"{row['name']:>10} | desired_loss={float(row['desired_loss']):.4f} "
            f"desired_acc={float(row['desired_acc']):.3f} general_loss={float(row['general_loss']):.4f} "
            f"selected_fraction={float(row['selected_fraction']):.2f}"
        )
    print(f"\nWrote logs to {out_dir}")


if __name__ == "__main__":
    main()

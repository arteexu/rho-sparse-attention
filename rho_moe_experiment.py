#!/usr/bin/env python3
"""Mini RHO-1 style Selective Language Modeling experiment with MoE support.

This is intentionally compact, but it mirrors the important RHO-1 ingredients:

1. Train a reference LM on clean desired-domain data.
2. Score each pretraining token with reference loss.
3. Train candidate LMs with either CLM or SLM.
4. For SLM, use excess loss: model_loss - reference_loss.
5. Keep full context visible, but mask loss on unselected target positions.
6. Compare dense and MoE transformer blocks.

The data is synthetic so the experiment can run locally and produce interpretable
diagnostics without downloading datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"


@dataclass
class ExperimentConfig:
    seed: int = 7
    seq_len: int = 64
    train_sequences: int = 4096
    ref_sequences: int = 1536
    eval_sequences: int = 512
    batch_size: int = 32
    ref_steps: int = 120
    steps: int = 220
    lr: float = 3e-4
    weight_decay: float = 0.01
    d_model: int = 96
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 192
    dropout: float = 0.0
    n_experts: int = 4
    moe_top_k: int = 2
    moe_aux_weight: float = 0.05
    select_ratio: float = 0.6
    slm_warmup_steps: int = 20
    eval_every: int = 50
    device: str = "auto"
    out_dir: str = "runs"


PRESETS: dict[str, dict[str, object]] = {
    "smoke": {
        "train_sequences": 768,
        "ref_sequences": 384,
        "eval_sequences": 128,
        "batch_size": 16,
        "ref_steps": 35,
        "steps": 60,
        "d_model": 64,
        "n_layers": 1,
        "n_heads": 4,
        "d_ff": 128,
        "eval_every": 30,
        "slm_warmup_steps": 8,
    },
    "small": {
        "train_sequences": 4096,
        "ref_sequences": 1536,
        "eval_sequences": 512,
        "batch_size": 32,
        "ref_steps": 180,
        "steps": 300,
        "d_model": 96,
        "n_layers": 2,
        "n_heads": 4,
        "d_ff": 192,
        "eval_every": 50,
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Vocab:
    def __init__(self, tokens: Iterable[str]):
        unique = [PAD, BOS, EOS]
        seen = set(unique)
        for tok in tokens:
            if tok not in seen:
                unique.append(tok)
                seen.add(tok)
        self.itos = unique
        self.stoi = {tok: idx for idx, tok in enumerate(unique)}

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi[tok] for tok in tokens]

    def decode(self, ids: Iterable[int]) -> list[str]:
        return [self.itos[i] for i in ids]

    def __len__(self) -> int:
        return len(self.itos)


def build_token_inventory() -> tuple[list[str], set[str], set[str]]:
    digits = [str(i) for i in range(40)]
    variables = ["x", "y", "z", "n", "m"]
    ops = ["+", "-", "*", "/", "=", "(", ")", "mod", "then", "therefore", "because"]
    math_words = [
        "solve",
        "add",
        "subtract",
        "multiply",
        "divide",
        "number",
        "sum",
        "product",
        "difference",
        "equation",
        "answer",
        "step",
        "check",
        "gives",
        "carry",
        "remainder",
    ]
    clean_words = ["the", "a", "is", "and", "with", "to", "we", "get", "find", "value"]
    noisy = [
        "<html>",
        "</div>",
        "timestamp",
        "user_492",
        "kudos",
        "permalink",
        "0x7ff",
        "adclick",
        "cookie",
        "nav",
        "random_id",
        "IMG_001",
        "zzqv",
        "lorem",
        "#####",
        "http",
        "utm_source",
        "reply",
        "joined",
        "footer",
    ]
    desired = set(digits + variables + ops + math_words)
    noise = set(noisy)
    all_tokens = digits + variables + ops + math_words + clean_words + noisy
    return all_tokens, desired, noise


def make_clean_math_sequence(seq_len: int, rng: random.Random) -> list[str]:
    """Generate simple math-reasoning-shaped text."""
    a = rng.randint(1, 19)
    b = rng.randint(1, 19)
    op = rng.choice(["+", "-", "*"])
    if op == "+":
        ans = a + b
        verb = "add"
    elif op == "-":
        a, b = max(a, b), min(a, b)
        ans = a - b
        verb = "subtract"
    else:
        ans = a * b
        if ans >= 40:
            ans %= 40
        verb = "multiply"
    toks = [
        BOS,
        "solve",
        "the",
        "equation",
        str(a),
        op,
        str(b),
        "=",
        "x",
        "step",
        "we",
        verb,
        str(a),
        "and",
        str(b),
        "then",
        "x",
        "=",
        str(ans),
        "therefore",
        "answer",
        "=",
        str(ans),
        EOS,
    ]
    return pad_or_trim(toks, seq_len)


def make_mixed_sequence(seq_len: int, rng: random.Random) -> list[str]:
    """Generate text with useful math spans plus web-noise spans."""
    toks: list[str] = [BOS]
    while len(toks) < seq_len - 1:
        choice = rng.random()
        if choice < 0.58:
            span = make_clean_math_sequence(24, rng)[1:-1]
            toks.extend(span[: rng.randint(6, 15)])
        elif choice < 0.78:
            toks.extend(rng.choices(["the", "a", "is", "and", "with", "to", "we", "get"], k=rng.randint(3, 8)))
        else:
            toks.extend(rng.choices(list(NOISE_TOKENS), k=rng.randint(3, 10)))
    toks.append(EOS)
    return pad_or_trim(toks, seq_len)


def make_noise_sequence(seq_len: int, rng: random.Random) -> list[str]:
    toks = [BOS]
    noisy = list(NOISE_TOKENS)
    filler = ["the", "a", "is", "and", "with", "to", "reply", "joined"]
    while len(toks) < seq_len - 1:
        toks.append(rng.choice(noisy if rng.random() < 0.7 else filler))
    toks.append(EOS)
    return pad_or_trim(toks, seq_len)


def pad_or_trim(tokens: list[str], seq_len: int) -> list[str]:
    if len(tokens) > seq_len:
        tokens = tokens[:seq_len]
        tokens[-1] = EOS
    if len(tokens) < seq_len:
        tokens = tokens + [PAD] * (seq_len - len(tokens))
    return tokens


def build_dataset(vocab: Vocab, seqs: list[list[str]], desired_tokens: set[str]) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.tensor([vocab.encode(seq) for seq in seqs], dtype=torch.long)
    desired_mask = torch.tensor([[tok in desired_tokens for tok in seq] for seq in seqs], dtype=torch.bool)
    return ids, desired_mask


def sample_batch(data: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    idx = torch.randint(0, data.size(0), (batch_size,), device=torch.device("cpu"))
    return data[idx].to(device)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.out(y)


class DenseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.net(x), {}


class MoEFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_experts: int, top_k: int, dropout: float):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([DenseFFN(d_model, d_ff, dropout) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, seq_len, d_model = x.shape
        flat = x.reshape(batch * seq_len, d_model)
        logits = self.router(flat)
        probs = F.softmax(logits, dim=-1)
        top_prob, top_idx = torch.topk(probs, k=self.top_k, dim=-1)
        top_prob = top_prob / top_prob.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        out = torch.zeros_like(flat)
        expert_fractions = []
        for expert_idx, expert in enumerate(self.experts):
            route_mask = top_idx == expert_idx
            token_mask = route_mask.any(dim=-1)
            expert_fractions.append(route_mask.float().mean())
            if token_mask.any():
                route_weight = (top_prob * route_mask.float()).sum(dim=-1)
                expert_out, _ = expert(flat[token_mask].view(1, -1, d_model))
                out[token_mask] += expert_out.view(-1, d_model) * route_weight[token_mask, None]

        density_1 = torch.stack(expert_fractions)
        density_proxy = probs.mean(dim=0)
        load_balance_loss = self.n_experts * torch.sum(density_1 * density_proxy)
        entropy = -(probs * (probs + 1e-9).log()).sum(dim=-1).mean()
        stats = {
            "load_balance_loss": load_balance_loss,
            "router_entropy": entropy.detach(),
            "expert_fractions": density_1.detach(),
        }
        return out.view(batch, seq_len, d_model), stats


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ExperimentConfig, moe: bool):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = (
            MoEFFN(cfg.d_model, cfg.d_ff, cfg.n_experts, cfg.moe_top_k, cfg.dropout)
            if moe
            else DenseFFN(cfg.d_model, cfg.d_ff, cfg.dropout)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = x + self.attn(self.ln1(x))
        ffn_out, stats = self.ffn(self.ln2(x))
        x = x + ffn_out
        return x, stats


class MiniTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, cfg: ExperimentConfig, moe: bool = False):
        super().__init__()
        self.moe = moe
        self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg, moe) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, seq_len = idx.shape
        pos = torch.arange(seq_len, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        aux_losses = []
        entropies = []
        expert_fractions = []
        for block in self.blocks:
            x, stats = block(x)
            if "load_balance_loss" in stats:
                aux_losses.append(stats["load_balance_loss"])
                entropies.append(stats["router_entropy"])
                expert_fractions.append(stats["expert_fractions"])
        logits = self.head(self.ln_f(x))
        out_stats: dict[str, torch.Tensor] = {}
        if aux_losses:
            out_stats["load_balance_loss"] = torch.stack(aux_losses).mean()
            out_stats["router_entropy"] = torch.stack(entropies).mean()
            out_stats["expert_fractions"] = torch.stack(expert_fractions).mean(dim=0)
        return logits, out_stats


def token_losses(logits: torch.Tensor, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Predict idx[:, 1:] from logits[:, :-1].
    targets = idx[:, 1:]
    losses = F.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
        targets.contiguous().view(-1),
        ignore_index=0,
        reduction="none",
    ).view_as(targets)
    valid = targets != 0
    return losses, valid


@torch.no_grad()
def evaluate(model: MiniTransformerLM, data: torch.Tensor, batch_size: int, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_count = 0.0
    for start in range(0, data.size(0), batch_size):
        batch = data[start : start + batch_size].to(device)
        logits, stats = model(batch)
        losses, valid = token_losses(logits, batch)
        preds = logits[:, :-1, :].argmax(dim=-1)
        targets = batch[:, 1:]
        total_loss += losses[valid].sum().item()
        total_correct += (preds[valid] == targets[valid]).float().sum().item()
        total_count += valid.float().sum().item()
    return {
        "loss": total_loss / max(total_count, 1.0),
        "ppl": math.exp(min(total_loss / max(total_count, 1.0), 20.0)),
        "acc": total_correct / max(total_count, 1.0),
    }


@torch.no_grad()
def selection_diagnostics(
    model: MiniTransformerLM,
    ref_model: MiniTransformerLM,
    batch: torch.Tensor,
    select_ratio: float,
    desired_token_ids: torch.Tensor,
    noise_token_ids: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    ref_model.eval()
    model_logits, _ = model(batch)
    ref_logits, _ = ref_model(batch)
    model_loss, valid = token_losses(model_logits, batch)
    ref_loss, _ = token_losses(ref_logits, batch)
    excess = model_loss - ref_loss
    selected = topk_mask(excess, valid, select_ratio)
    targets = batch[:, 1:]
    desired_targets = torch.isin(targets, desired_token_ids)
    noise_targets = torch.isin(targets, noise_token_ids)
    selected_loss = model_loss[selected].mean().item() if selected.any() else float("nan")
    unselected = valid & ~selected
    unselected_loss = model_loss[unselected].mean().item() if unselected.any() else float("nan")
    selected_count = max(selected.float().sum().item(), 1.0)
    unselected_count = max(unselected.float().sum().item(), 1.0)
    return {
        "selected_fraction": selected.float().sum().item() / max(valid.float().sum().item(), 1.0),
        "selected_model_loss": selected_loss,
        "unselected_model_loss": unselected_loss,
        "mean_excess_selected": excess[selected].mean().item() if selected.any() else float("nan"),
        "mean_excess_unselected": excess[unselected].mean().item() if unselected.any() else float("nan"),
        "selected_desired_fraction": (selected & desired_targets).float().sum().item() / selected_count,
        "selected_noise_fraction": (selected & noise_targets).float().sum().item() / selected_count,
        "unselected_desired_fraction": (unselected & desired_targets).float().sum().item() / unselected_count,
        "unselected_noise_fraction": (unselected & noise_targets).float().sum().item() / unselected_count,
    }


def topk_mask(score: torch.Tensor, valid: torch.Tensor, select_ratio: float) -> torch.Tensor:
    mask = torch.zeros_like(valid)
    flat_score = score[valid]
    if flat_score.numel() == 0:
        return mask
    k = max(1, int(math.ceil(select_ratio * flat_score.numel())))
    threshold = torch.topk(flat_score, k=k).values.min()
    mask = valid & (score >= threshold)
    # Ties can push the exact fraction above k. That is fine for this experiment.
    return mask


def train_reference(
    cfg: ExperimentConfig,
    vocab_size: int,
    ref_train: torch.Tensor,
    desired_eval: torch.Tensor,
    device: torch.device,
) -> MiniTransformerLM:
    model = MiniTransformerLM(vocab_size, cfg, moe=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    model.train()
    for step in range(1, cfg.ref_steps + 1):
        batch = sample_batch(ref_train, cfg.batch_size, device)
        logits, _ = model(batch)
        losses, valid = token_losses(logits, batch)
        loss = losses[valid].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == cfg.ref_steps or step % max(1, cfg.ref_steps // 3) == 0:
            metrics = evaluate(model, desired_eval, cfg.batch_size, device)
            print(f"[reference] step={step:04d} train_loss={loss.item():.4f} desired_eval_loss={metrics['loss']:.4f}")
    return model


def train_candidate(
    name: str,
    cfg: ExperimentConfig,
    vocab_size: int,
    train_data: torch.Tensor,
    desired_eval: torch.Tensor,
    noise_eval: torch.Tensor,
    ref_model: MiniTransformerLM,
    desired_token_ids: torch.Tensor,
    noise_token_ids: torch.Tensor,
    device: torch.device,
    moe: bool,
    selective: bool,
    out_dir: Path,
) -> dict[str, float]:
    model = MiniTransformerLM(vocab_size, cfg, moe=moe).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history_path = out_dir / f"{name}.csv"
    ref_model.eval()
    rows: list[dict[str, float | int | str]] = []

    for step in range(1, cfg.steps + 1):
        model.train()
        batch = sample_batch(train_data, cfg.batch_size, device)
        logits, stats = model(batch)
        losses, valid = token_losses(logits, batch)

        selected_fraction = 1.0
        if selective and step > cfg.slm_warmup_steps:
            with torch.no_grad():
                ref_logits, _ = ref_model(batch)
                ref_losses, _ = token_losses(ref_logits, batch)
                excess = losses.detach() - ref_losses
                selected = topk_mask(excess, valid, cfg.select_ratio)
            main_loss = losses[selected].mean()
            selected_fraction = selected.float().sum().item() / max(valid.float().sum().item(), 1.0)
        else:
            main_loss = losses[valid].mean()

        aux_loss = stats.get("load_balance_loss", torch.tensor(0.0, device=device))
        loss = main_loss + (cfg.moe_aux_weight * aux_loss if moe else 0.0)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % cfg.eval_every == 0 or step == cfg.steps:
            desired_metrics = evaluate(model, desired_eval, cfg.batch_size, device)
            noise_metrics = evaluate(model, noise_eval, cfg.batch_size, device)
            diag = (
                selection_diagnostics(
                    model,
                    ref_model,
                    batch,
                    cfg.select_ratio,
                    desired_token_ids,
                    noise_token_ids,
                )
                if selective
                else {}
            )
            row: dict[str, float | int | str] = {
                "step": step,
                "train_loss": loss.item(),
                "main_loss": main_loss.item(),
                "selected_fraction": selected_fraction,
                "desired_loss": desired_metrics["loss"],
                "desired_ppl": desired_metrics["ppl"],
                "desired_acc": desired_metrics["acc"],
                "noise_loss": noise_metrics["loss"],
                "noise_ppl": noise_metrics["ppl"],
                "noise_acc": noise_metrics["acc"],
                "load_balance_loss": aux_loss.item() if moe else 0.0,
                "router_entropy": stats.get("router_entropy", torch.tensor(0.0)).item() if moe else 0.0,
            }
            if moe and "expert_fractions" in stats:
                for i, frac in enumerate(stats["expert_fractions"].detach().cpu().tolist()):
                    row[f"expert_fraction_{i}"] = float(frac)
            for key, value in diag.items():
                row[key] = value
            rows.append(row)
            print(
                f"[{name}] step={step:04d} loss={loss.item():.4f} "
                f"sel={selected_fraction:.2f} desired_loss={desired_metrics['loss']:.4f} "
                f"desired_acc={desired_metrics['acc']:.3f} noise_loss={noise_metrics['loss']:.4f}"
            )

    write_csv(history_path, rows)
    final = rows[-1].copy()
    final["name"] = name
    return {k: v for k, v in final.items() if isinstance(v, (int, float, str))}


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def apply_overrides(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    for key, value in vars(args).items():
        if value is not None and hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="small")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seq-len", dest="seq_len", type=int)
    parser.add_argument("--train-sequences", dest="train_sequences", type=int)
    parser.add_argument("--ref-sequences", dest="ref_sequences", type=int)
    parser.add_argument("--eval-sequences", dest="eval_sequences", type=int)
    parser.add_argument("--batch-size", dest="batch_size", type=int)
    parser.add_argument("--ref-steps", dest="ref_steps", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--d-model", dest="d_model", type=int)
    parser.add_argument("--n-layers", dest="n_layers", type=int)
    parser.add_argument("--n-heads", dest="n_heads", type=int)
    parser.add_argument("--d-ff", dest="d_ff", type=int)
    parser.add_argument("--n-experts", dest="n_experts", type=int)
    parser.add_argument("--moe-top-k", dest="moe_top_k", type=int)
    parser.add_argument("--select-ratio", dest="select_ratio", type=float)
    parser.add_argument("--slm-warmup-steps", dest="slm_warmup_steps", type=int)
    parser.add_argument("--eval-every", dest="eval_every", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--out-dir", dest="out_dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig()
    for key, value in PRESETS[args.preset].items():
        setattr(cfg, key, value)
    cfg = apply_overrides(cfg, args)
    if not (0.0 < cfg.select_ratio <= 1.0):
        raise ValueError("--select-ratio must be in (0, 1]")

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"Using device: {device}")

    rng = random.Random(cfg.seed)
    all_tokens, desired_tokens, _noise_tokens = build_token_inventory()
    global NOISE_TOKENS
    NOISE_TOKENS = _noise_tokens
    vocab = Vocab(all_tokens)
    desired_token_ids = torch.tensor(vocab.encode(sorted(desired_tokens)), dtype=torch.long, device=device)
    noise_token_ids = torch.tensor(vocab.encode(sorted(_noise_tokens)), dtype=torch.long, device=device)

    ref_seqs = [make_clean_math_sequence(cfg.seq_len, rng) for _ in range(cfg.ref_sequences)]
    train_seqs = [make_mixed_sequence(cfg.seq_len, rng) for _ in range(cfg.train_sequences)]
    desired_eval_seqs = [make_clean_math_sequence(cfg.seq_len, rng) for _ in range(cfg.eval_sequences)]
    noise_eval_seqs = [make_noise_sequence(cfg.seq_len, rng) for _ in range(cfg.eval_sequences)]

    ref_train, _ = build_dataset(vocab, ref_seqs, desired_tokens)
    train_data, _ = build_dataset(vocab, train_seqs, desired_tokens)
    desired_eval, _ = build_dataset(vocab, desired_eval_seqs, desired_tokens)
    noise_eval, _ = build_dataset(vocab, noise_eval_seqs, desired_tokens)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.out_dir) / f"rho_moe_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (out_dir / "vocab.json").write_text(json.dumps(vocab.itos, indent=2))

    print(f"Vocab size: {len(vocab)}")
    print(f"Output dir: {out_dir}")
    print("Training reference model...")
    ref_model = train_reference(cfg, len(vocab), ref_train, desired_eval, device)

    results = []
    runs = [
        ("dense_clm", False, False),
        ("dense_slm", False, True),
        ("moe_clm", True, False),
        ("moe_slm", True, True),
    ]
    for name, moe, selective in runs:
        print(f"\nTraining {name}...")
        result = train_candidate(
            name=name,
            cfg=cfg,
            vocab_size=len(vocab),
            train_data=train_data,
            desired_eval=desired_eval,
            noise_eval=noise_eval,
            ref_model=ref_model,
            desired_token_ids=desired_token_ids,
            noise_token_ids=noise_token_ids,
            device=device,
            moe=moe,
            selective=selective,
            out_dir=out_dir,
        )
        results.append(result)

    write_csv(out_dir / "summary.csv", results)
    print("\nFinal summary:")
    for row in results:
        print(
            f"{row['name']:>10} | desired_loss={float(row['desired_loss']):.4f} "
            f"desired_acc={float(row['desired_acc']):.3f} noise_loss={float(row['noise_loss']):.4f} "
            f"selected_fraction={float(row['selected_fraction']):.2f}"
        )
    print(f"\nWrote logs to {out_dir}")


if __name__ == "__main__":
    NOISE_TOKENS: set[str] = set()
    main()

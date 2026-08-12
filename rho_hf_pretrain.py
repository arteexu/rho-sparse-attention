#!/usr/bin/env python3
"""AWS-scale RHO-1 style continual pretraining for Hugging Face causal LMs.

This script is meant for 1B-ish model experiments on multi-GPU AWS instances.
It uses a real pretrained CausalLM, a reference CausalLM, streaming text data,
and token-level selective loss masking.

It deliberately avoids Trainer so the RHO-1 mechanics stay visible:

- CLM: train on every target token.
- SLM: train on top-k% tokens by model_loss - reference_loss.
- random/current_loss/ref_low: selection ablations.
- curriculum schedules: e.g. 0:0.6,2000:0.8,4000:1.0.

Launch with torchrun for DDP. Each process owns one GPU and sees a shard of the
streaming dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterator

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


SELECTION_STRATEGIES = {"clm", "slm", "random", "current_loss", "ref_low"}


def is_dist() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main() -> bool:
    return rank() == 0


def setup_dist() -> torch.device:
    if is_dist():
        torch.cuda.set_device(local_rank())
        dist.init_process_group(backend="nccl")
        return torch.device("cuda", local_rank())
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cleanup_dist() -> None:
    if is_dist() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def log(msg: str) -> None:
    if is_main():
        print(msg, flush=True)


def parse_schedule(schedule: str) -> list[tuple[int, float]]:
    if not schedule.strip():
        return []
    points: list[tuple[int, float]] = []
    for raw_part in schedule.split(","):
        part = raw_part.strip()
        if not part:
            continue
        step_s, ratio_s = part.split(":", 1)
        step = int(step_s)
        ratio = float(ratio_s)
        if step < 0 or not (0.0 < ratio <= 1.0):
            raise ValueError("Schedule entries require step >= 0 and ratio in (0, 1]")
        points.append((step, ratio))
    return sorted(points)


def scheduled_ratio(base_ratio: float, schedule: str, step: int) -> float:
    ratio = base_ratio
    for start, value in parse_schedule(schedule):
        if step >= start:
            ratio = value
        else:
            break
    return ratio


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows or not is_main():
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def mirror_small_artifacts(args: argparse.Namespace) -> None:
    """Copy lightweight run metadata to a directory that the block log sync sees."""
    if not args.mirror_dir or not is_main():
        return
    out = Path(args.output_dir)
    mirror = Path(args.mirror_dir)
    mirror.mkdir(parents=True, exist_ok=True)
    for relative in [
        Path("config.json"),
        Path("train_log.csv"),
        Path("checkpoint-last") / "trainer_state.json",
    ]:
        source = out / relative
        if source.exists() and source.is_file():
            target = mirror / relative.name
            shutil.copy2(source, target)


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in {torch.bfloat16, torch.float16}:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError("--dtype must be bf16, fp16, or fp32")


def load_text_stream(args: argparse.Namespace):
    kwargs = {"split": args.dataset_split, "streaming": True}
    if args.dataset_config:
        return load_dataset(args.dataset_name, args.dataset_config, **kwargs)
    return load_dataset(args.dataset_name, **kwargs)


def extract_text(example: dict, text_field: str) -> str:
    value = example
    for part in text_field.split("."):
        value = value[part]
    return str(value)


def packed_batch_iterator(
    args: argparse.Namespace,
    tokenizer,
    device: torch.device,
    seed: int,
) -> Iterator[torch.Tensor]:
    """Yield [micro_batch, seq_len] input batches from a streaming dataset."""
    rng = random.Random(seed + rank())
    dataset = load_text_stream(args)
    token_buffer: list[int] = []
    doc_idx = 0
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Tokenizer must have eos_token_id")

    while True:
        for example in dataset:
            if doc_idx % world_size() != rank():
                doc_idx += 1
                continue
            doc_idx += 1
            text = extract_text(example, args.text_field)
            if args.shuffle_documents and rng.random() < args.document_drop_prob:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
            if not ids:
                continue
            token_buffer.extend(ids)
            token_buffer.append(eos)
            while len(token_buffer) >= args.seq_len * args.micro_batch_size:
                chunk = token_buffer[: args.seq_len * args.micro_batch_size]
                del token_buffer[: args.seq_len * args.micro_batch_size]
                batch = torch.tensor(chunk, dtype=torch.long).view(args.micro_batch_size, args.seq_len)
                yield batch.to(device, non_blocking=True)
        dataset = load_text_stream(args)


def get_token_losses(logits: torch.Tensor, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    targets = input_ids[:, 1:]
    logits = logits[:, :-1, :]
    losses = F.cross_entropy(
        logits.contiguous().view(-1, logits.size(-1)),
        targets.contiguous().view(-1),
        reduction="none",
    ).view_as(targets)
    valid = torch.ones_like(targets, dtype=torch.bool)
    return losses, valid


def topk_mask(score: torch.Tensor, valid: torch.Tensor, ratio: float) -> torch.Tensor:
    if ratio >= 0.999:
        return valid
    flat = score[valid]
    if flat.numel() == 0:
        return torch.zeros_like(valid)
    k = max(1, int(math.ceil(ratio * flat.numel())))
    threshold = torch.topk(flat, k=k).values.min()
    return valid & (score >= threshold)


def random_mask(valid: torch.Tensor, ratio: float) -> torch.Tensor:
    if ratio >= 0.999:
        return valid
    score = torch.rand(valid.shape, device=valid.device).masked_fill(~valid, -float("inf"))
    return topk_mask(score, valid, ratio)


def build_selection(
    strategy: str,
    model_losses: torch.Tensor,
    ref_losses: torch.Tensor | None,
    valid: torch.Tensor,
    ratio: float,
) -> torch.Tensor:
    if strategy == "clm" or ratio >= 0.999:
        return valid
    if strategy == "random":
        return random_mask(valid, ratio)
    if strategy == "current_loss":
        return topk_mask(model_losses.detach(), valid, ratio)
    if ref_losses is None:
        raise ValueError(f"{strategy} requires reference losses")
    if strategy == "slm":
        return topk_mask(model_losses.detach() - ref_losses, valid, ratio)
    if strategy == "ref_low":
        return topk_mask(-ref_losses, valid, ratio)
    raise ValueError(f"Unknown strategy: {strategy}")


@torch.no_grad()
def evaluate_loss(args, model, tokenizer, device: torch.device, dtype: torch.dtype) -> dict[str, float]:
    if args.eval_batches <= 0:
        return {}
    was_training = model.training
    model.eval()
    iterator = packed_batch_iterator(args, tokenizer, device, args.seed + 100_000)
    total_loss = torch.tensor(0.0, device=device)
    total_tokens = torch.tensor(0.0, device=device)
    for _ in range(args.eval_batches):
        batch = next(iterator)
        with autocast_context(device, dtype):
            logits = model(batch).logits
            losses, valid = get_token_losses(logits, batch)
        total_loss += losses[valid].sum()
        total_tokens += valid.sum()
    if is_dist():
        dist.all_reduce(total_loss)
        dist.all_reduce(total_tokens)
    if was_training:
        model.train()
    loss = (total_loss / total_tokens.clamp_min(1)).item()
    return {"eval_loss": loss, "eval_ppl": math.exp(min(loss, 20.0))}


def save_checkpoint(args, model, optimizer, scheduler, step: int, rows: list[dict[str, object]]) -> None:
    out = Path(args.output_dir)
    ckpt = out / "checkpoint-last"
    if is_main():
        ckpt.mkdir(parents=True, exist_ok=True)
        unwrapped = model.module if isinstance(model, DDP) else model
        unwrapped.save_pretrained(ckpt / "model", safe_serialization=True)
        with (ckpt / "trainer_state.json").open("w") as f:
            json.dump({"step": step, "rows": rows}, f, indent=2)
        write_csv(out / "train_log.csv", rows)
    if is_dist():
        dist.barrier()
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
        },
        ckpt / f"optim_rank{rank()}.pt",
    )
    if is_dist():
        dist.barrier()


def maybe_resume(args, model, optimizer, scheduler, device: torch.device) -> tuple[int, list[dict[str, object]]]:
    ckpt = Path(args.output_dir) / "checkpoint-last"
    if not args.resume or not ckpt.exists():
        return 0, []
    opt_path = ckpt / f"optim_rank{rank()}.pt"
    if opt_path.exists():
        state = torch.load(opt_path, map_location=device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = int(state["step"])
    else:
        start_step = 0
    rows: list[dict[str, object]] = []
    state_path = ckpt / "trainer_state.json"
    if is_main() and state_path.exists():
        rows = json.load(state_path.open()).get("rows", [])
    if is_dist():
        payload = [rows]
        dist.broadcast_object_list(payload, src=0)
        rows = payload[0]
    return start_step, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    parser.add_argument("--reference-model-name", default="TinyLlama/TinyLlama_v1.1_math_code")
    parser.add_argument("--dataset-name", default="open-web-math/open-web-math")
    parser.add_argument("--dataset-config", default="")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--strategy", choices=sorted(SELECTION_STRATEGIES), default="slm")
    parser.add_argument("--select-ratio", type=float, default=0.6)
    parser.add_argument("--select-ratio-schedule", default="")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--slm-warmup-steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--attn-implementation", default="")
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-runtime-hours",
        type=float,
        default=0.0,
        help="Stop cleanly after this many hours, saving checkpoint-last first. 0 disables.",
    )
    parser.add_argument(
        "--mirror-dir",
        default="",
        help="Optional local directory for small CSV/config copies, useful for block log sync.",
    )
    parser.add_argument("--shuffle-documents", action="store_true")
    parser.add_argument("--document-drop-prob", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.strategy not in SELECTION_STRATEGIES:
        raise ValueError(f"Unknown strategy {args.strategy}")
    if not (0.0 < args.select_ratio <= 1.0):
        raise ValueError("--select-ratio must be in (0, 1]")

    device = setup_dist()
    dtype = dtype_from_name(args.dtype)
    torch.manual_seed(args.seed + rank())
    random.seed(args.seed + rank())

    out = Path(args.output_dir)
    if is_main():
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if is_main():
        tokenizer.save_pretrained(out / "tokenizer")

    model_load_path = out / "checkpoint-last" / "model"
    model_source = str(model_load_path) if args.resume and model_load_path.exists() else args.model_name
    model_kwargs = {"torch_dtype": dtype}
    ref_kwargs = {"torch_dtype": dtype}
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
        ref_kwargs["attn_implementation"] = args.attn_implementation
    log(f"Loading train model from {model_source}")
    model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs).to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    need_reference = args.strategy in {"slm", "ref_low"}
    reference_model = None
    if need_reference:
        log(f"Loading reference model from {args.reference_model_name}")
        reference_model = AutoModelForCausalLM.from_pretrained(args.reference_model_name, **ref_kwargs).to(device)
        reference_model.eval()
        for p in reference_model.parameters():
            p.requires_grad_(False)

    if is_dist():
        model = DDP(model, device_ids=[local_rank()], output_device=local_rank(), find_unused_parameters=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.max_steps)
    start_step, rows = maybe_resume(args, model, optimizer, scheduler, device)
    if start_step:
        log(f"Resumed from step {start_step}")

    iterator = packed_batch_iterator(args, tokenizer, device, args.seed)
    model.train()
    start_time = time.perf_counter()
    cumulative_total_tokens = int(rows[-1]["cumulative_total_tokens"]) if rows else 0
    cumulative_selected_tokens = int(rows[-1]["cumulative_selected_tokens"]) if rows else 0
    optimizer.zero_grad(set_to_none=True)

    for step in range(start_step + 1, args.max_steps + 1):
        step_loss = torch.tensor(0.0, device=device)
        step_total_tokens = 0
        step_selected_tokens = 0
        active_ratio = scheduled_ratio(args.select_ratio, args.select_ratio_schedule, step)

        for accum_idx in range(args.grad_accum_steps):
            batch = next(iterator)
            with autocast_context(device, dtype):
                logits = model(batch).logits
                losses, valid = get_token_losses(logits, batch)
                ref_losses = None
                if need_reference and args.strategy != "clm" and step > args.slm_warmup_steps and active_ratio < 0.999:
                    assert reference_model is not None
                    with torch.no_grad():
                        ref_logits = reference_model(batch).logits
                        ref_losses, _ = get_token_losses(ref_logits, batch)
                selected = (
                    build_selection(args.strategy, losses, ref_losses, valid, active_ratio)
                    if step > args.slm_warmup_steps
                    else valid
                )
                loss = losses[selected].mean() / args.grad_accum_steps
            loss.backward()
            step_loss += loss.detach() * args.grad_accum_steps
            step_total_tokens += int(valid.sum().item())
            step_selected_tokens += int(selected.sum().item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        stat = torch.tensor([step_loss.item(), step_total_tokens, step_selected_tokens], device=device)
        if is_dist():
            dist.all_reduce(stat, op=dist.ReduceOp.SUM)
        mean_loss = stat[0].item() / world_size()
        total_tokens = int(stat[1].item())
        selected_tokens = int(stat[2].item())
        cumulative_total_tokens += total_tokens
        cumulative_selected_tokens += selected_tokens

        should_log = step == 1 or step % args.log_every == 0 or step % args.eval_every == 0 or step == args.max_steps
        if should_log:
            elapsed = time.perf_counter() - start_time
            row: dict[str, object] = {
                "step": step,
                "strategy": args.strategy,
                "active_select_ratio": active_ratio,
                "train_loss": mean_loss,
                "step_total_tokens": total_tokens,
                "step_selected_tokens": selected_tokens,
                "selected_fraction": selected_tokens / max(total_tokens, 1),
                "cumulative_total_tokens": cumulative_total_tokens,
                "cumulative_selected_tokens": cumulative_selected_tokens,
                "elapsed_sec": elapsed,
                "total_tokens_per_sec": cumulative_total_tokens / max(elapsed, 1e-9),
                "selected_tokens_per_sec": cumulative_selected_tokens / max(elapsed, 1e-9),
                "lr": scheduler.get_last_lr()[0],
            }
            if step % args.eval_every == 0 or step == args.max_steps:
                row.update(evaluate_loss(args, model.module if isinstance(model, DDP) else model, tokenizer, device, dtype))
            rows.append(row)
            if is_main():
                eval_part = f" eval_loss={row.get('eval_loss', float('nan')):.4f}" if "eval_loss" in row else ""
                print(
                    f"step={step} loss={mean_loss:.4f} sel={row['selected_fraction']:.3f} "
                    f"cum_sel={cumulative_selected_tokens} cum_total={cumulative_total_tokens}{eval_part}",
                    flush=True,
                )
                write_csv(out / "train_log.csv", rows)
                mirror_small_artifacts(args)

        if step % args.save_every == 0 or step == args.max_steps:
            save_checkpoint(args, model, optimizer, scheduler, step, rows)
            mirror_small_artifacts(args)

        if args.max_runtime_hours > 0:
            elapsed = time.perf_counter() - start_time
            if elapsed >= args.max_runtime_hours * 3600:
                log(
                    f"Reached max runtime of {args.max_runtime_hours:.2f} hours at step {step}; "
                    "saving checkpoint-last and stopping cleanly."
                )
                save_checkpoint(args, model, optimizer, scheduler, step, rows)
                mirror_small_artifacts(args)
                break

    cleanup_dist()


if __name__ == "__main__":
    main()

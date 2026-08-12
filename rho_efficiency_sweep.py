#!/usr/bin/env python3
"""Run multi-seed RHO-1 efficiency sweeps and aggregate results."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def parse_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return float("nan")
    return float(value)


def mean_std(values: list[float]) -> tuple[float, float]:
    clean = [value for value in values if not math.isnan(value)]
    if not clean:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return clean[0], 0.0
    return statistics.mean(clean), statistics.stdev(clean)


def aggregate_final(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["name"]), []).append(row)

    metric_keys = [
        "desired_loss",
        "desired_acc",
        "general_loss",
        "general_acc",
        "elapsed_sec",
        "cumulative_total_tokens",
        "cumulative_selected_tokens",
        "selected_fraction",
        "selected_desired_source_fraction",
    ]
    output: list[dict[str, object]] = []
    for name, group in sorted(grouped.items()):
        out: dict[str, object] = {"name": name, "n": len(group)}
        for key in metric_keys:
            values = [float(row[key]) for row in group if key in row and row[key] != ""]
            mean, std = mean_std(values)
            out[f"{key}_mean"] = mean
            out[f"{key}_std"] = std
        output.append(out)
    return output


def aggregate_crossings(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["name"]), []).append(row)

    output: list[dict[str, object]] = []
    for name, group in sorted(grouped.items()):
        hits = [row for row in group if str(row["hit_target"]) == "True"]
        out: dict[str, object] = {
            "name": name,
            "n": len(group),
            "hit_rate": len(hits) / max(len(group), 1),
        }
        for key in ["hit_step", "hit_elapsed_sec", "hit_total_tokens", "hit_selected_tokens"]:
            values = [float(row[key]) for row in hits if row.get(key, "") != ""]
            mean, std = mean_std(values)
            out[f"{key}_mean"] = mean
            out[f"{key}_std"] = std
        output.append(out)
    return output


def annotate_efficiency(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Add per-seed target-crossing rows.

    For each seed/family, the target is the final desired_loss of that family's
    CLM baseline. A candidate is more token-efficient if it reaches that target
    with fewer cumulative selected tokens or less elapsed time.
    """
    curves_by_seed_name: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        curves_by_seed_name.setdefault((str(row["seed"]), str(row["name"])), []).append(row)

    crossing_rows: list[dict[str, object]] = []
    for seed in sorted({str(row["seed"]) for row in rows}):
        for family in ["dense", "moe"]:
            clm_name = f"{family}_clm"
            clm_rows = curves_by_seed_name.get((seed, clm_name), [])
            if not clm_rows:
                continue
            target = float(clm_rows[-1]["desired_loss"])
            for (row_seed, name), curve in curves_by_seed_name.items():
                if row_seed != seed or not name.startswith(family):
                    continue
                hit = None
                for point in curve:
                    if float(point["desired_loss"]) <= target:
                        hit = point
                        break
                crossing_rows.append(
                    {
                        "seed": seed,
                        "family": family,
                        "name": name,
                        "target_desired_loss": target,
                        "hit_target": hit is not None,
                        "hit_step": "" if hit is None else hit["step"],
                        "hit_elapsed_sec": "" if hit is None else hit["elapsed_sec"],
                        "hit_total_tokens": "" if hit is None else hit["cumulative_total_tokens"],
                        "hit_selected_tokens": "" if hit is None else hit["cumulative_selected_tokens"],
                    }
                )
    return crossing_rows


def newest_run_dir(seed_dir: Path, before: set[Path]) -> Path:
    candidates = [path for path in seed_dir.glob("rho_moe_real_*") if path not in before and path.is_dir()]
    if not candidates:
        candidates = [path for path in seed_dir.glob("rho_moe_real_*") if path.is_dir()]
    if not candidates:
        raise RuntimeError(f"No rho_moe_real_* run directory found in {seed_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="11,12,13")
    parser.add_argument("--preset", default="small")
    parser.add_argument("--run", default="efficiency_dense")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default="runs/efficiency_sweeps")
    parser.add_argument("--trainer", default="rho_moe_real_train.py")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    args, extra = parse_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    sweep_dir = Path(args.out_dir) / time.strftime("sweep_%Y%m%d_%H%M%S")
    sweep_dir.mkdir(parents=True, exist_ok=True)

    final_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []

    for seed in seeds:
        seed_dir = sweep_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        before = set(seed_dir.glob("rho_moe_real_*"))
        cmd = [
            sys.executable,
            args.trainer,
            "--preset",
            args.preset,
            "--run",
            args.run,
            "--seed",
            str(seed),
            "--device",
            args.device,
            "--out-dir",
            str(seed_dir),
            *extra,
        ]
        print("\n" + " ".join(cmd), flush=True)
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True)
        run_dir = newest_run_dir(seed_dir, before)

        for row in read_csv(run_dir / "summary.csv"):
            row["seed"] = seed
            row["run_dir"] = str(run_dir)
            final_rows.append(row)

        for candidate_csv in sorted(run_dir.glob("*.csv")):
            if candidate_csv.name == "summary.csv":
                continue
            for row in read_csv(candidate_csv):
                row["seed"] = seed
                row["name"] = candidate_csv.stem
                row["run_dir"] = str(run_dir)
                curve_rows.append(row)

    if args.dry_run:
        print(f"\nDry run only. Sweep dir would be {sweep_dir}")
        return

    write_csv(sweep_dir / "final_summary.csv", final_rows)
    write_csv(sweep_dir / "curves.csv", curve_rows)
    write_csv(sweep_dir / "aggregate_final.csv", aggregate_final(final_rows))
    crossing_rows = annotate_efficiency(curve_rows)
    write_csv(sweep_dir / "target_crossings.csv", crossing_rows)
    write_csv(sweep_dir / "aggregate_target_crossings.csv", aggregate_crossings(crossing_rows))

    print(f"\nWrote sweep outputs to {sweep_dir}")
    print("Key files:")
    print(f"  {sweep_dir / 'aggregate_final.csv'}")
    print(f"  {sweep_dir / 'target_crossings.csv'}")
    print(f"  {sweep_dir / 'curves.csv'}")


if __name__ == "__main__":
    main()

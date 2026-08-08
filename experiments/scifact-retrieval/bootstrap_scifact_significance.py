#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


METRICS = ("mrr", "ndcg@10", "map@1000", "recall@10")
COMPARISONS = (
    ("e5_minus_bm25", "e5", "bm25"),
    ("rrf_minus_e5", "rrf", "e5"),
    ("cross_encoder_minus_rrf", "cross_encoder", "rrf"),
    ("cross_encoder_minus_bm25", "cross_encoder", "bm25"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paired bootstrap significance testing for SciFact per-claim metrics."
    )
    p.add_argument("--bm25", type=Path, required=True)
    p.add_argument("--e5", type=Path, required=True)
    p.add_argument("--rrf", type=Path, required=True)
    p.add_argument("--cross-encoder", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--bootstrap-samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        claim_id = str(row.get("claim_id", "")).strip()
        if not claim_id:
            raise ValueError(f"{path}:{line_no}: missing claim_id")
        if claim_id in rows:
            raise ValueError(f"{path}: duplicate claim_id {claim_id}")
        missing = [metric for metric in METRICS if metric not in row]
        if missing:
            raise ValueError(f"{path}:{line_no}: missing metrics {missing}")
        rows[claim_id] = row
    return rows


def holm_adjust(p_values: list[float]) -> list[float]:
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, p_values[index] * (n - rank))
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    diffs = a - b
    observed = float(np.mean(diffs))
    n = len(diffs)
    means = np.empty(samples, dtype=np.float64)

    start = 0
    chunk = 2000
    while start < samples:
        current = min(chunk, samples - start)
        idx = rng.integers(0, n, size=(current, n))
        means[start:start + current] = diffs[idx].mean(axis=1)
        start += current

    ci_low, ci_high = np.percentile(means, [2.5, 97.5])
    p_nonpos = (np.count_nonzero(means <= 0) + 1) / (samples + 1)
    p_nonneg = (np.count_nonzero(means >= 0) + 1) / (samples + 1)
    p_value = min(1.0, 2.0 * min(p_nonpos, p_nonneg))
    return observed, float(ci_low), float(ci_high), float(p_value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    systems = {
        "bm25": load_jsonl(args.bm25),
        "e5": load_jsonl(args.e5),
        "rrf": load_jsonl(args.rrf),
        "cross_encoder": load_jsonl(args.cross_encoder),
    }

    claim_ids = sorted(systems["bm25"])
    expected = set(claim_ids)
    for name, rows in systems.items():
        if set(rows) != expected:
            raise ValueError(f"Claim IDs differ for {name}")

    print(f"Aligned claims: {len(claim_ids)}")
    if len(claim_ids) != 188:
        print(f"WARNING: expected 188 claims, found {len(claim_ids)}")

    merged_rows: list[dict[str, Any]] = []
    for claim_id in claim_ids:
        row: dict[str, Any] = {"claim_id": claim_id}
        for system_name, system_rows in systems.items():
            for metric in METRICS:
                row[f"{system_name}_{metric}"] = float(system_rows[claim_id][metric])
        merged_rows.append(row)
    write_csv(args.output_dir / "per_claim_metrics_merged.csv", merged_rows)

    rng = np.random.default_rng(args.seed)
    results: list[dict[str, Any]] = []

    for metric in METRICS:
        metric_rows: list[dict[str, Any]] = []
        for comparison_name, system_a, system_b in COMPARISONS:
            a = np.array(
                [float(systems[system_a][cid][metric]) for cid in claim_ids],
                dtype=float,
            )
            b = np.array(
                [float(systems[system_b][cid][metric]) for cid in claim_ids],
                dtype=float,
            )
            observed, ci_low, ci_high, p_value = bootstrap(
                a, b, args.bootstrap_samples, rng
            )
            diffs = a - b
            metric_rows.append(
                {
                    "comparison": comparison_name,
                    "metric": metric,
                    "system_a": system_a,
                    "system_b": system_b,
                    "mean_a": float(a.mean()),
                    "mean_b": float(b.mean()),
                    "observed_difference": observed,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": p_value,
                    "wins": int(np.sum(diffs > 1e-12)),
                    "losses": int(np.sum(diffs < -1e-12)),
                    "ties": int(np.sum(np.abs(diffs) <= 1e-12)),
                    "examples": len(claim_ids),
                }
            )

        adjusted = holm_adjust([float(r["p_value"]) for r in metric_rows])
        for row, adj in zip(metric_rows, adjusted):
            row["holm_adjusted_p_value"] = adj
            row["significant_0_05"] = adj < 0.05
            results.append(row)

    csv_path = args.output_dir / "paired_bootstrap_comparisons.csv"
    write_csv(csv_path, results)

    report = {
        "method": {
            "test": "paired nonparametric bootstrap over SciFact claims",
            "bootstrap_samples": args.bootstrap_samples,
            "confidence_interval": "95% percentile interval",
            "multiple_testing": "Holm correction across four planned comparisons per metric",
            "seed": args.seed,
        },
        "claims": len(claim_ids),
        "metrics": list(METRICS),
        "comparisons": results,
    }
    report_path = args.output_dir / "paired_bootstrap_scifact.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nSCIFACT PAIRED BOOTSTRAP COMPLETED")
    for metric in METRICS:
        print(f"\n===== {metric} =====")
        print(f"{'Comparison':<34}{'Diff':>10}{'CI low':>10}{'CI high':>10}{'p Holm':>10}{'Sig':>7}")
        for row in results:
            if row["metric"] == metric:
                print(
                    f"{row['comparison']:<34}"
                    f"{row['observed_difference']:>10.4f}"
                    f"{row['ci_low']:>10.4f}"
                    f"{row['ci_high']:>10.4f}"
                    f"{row['holm_adjusted_p_value']:>10.4f}"
                    f"{str(row['significant_0_05']):>7}"
                )

    print("\nReport:", report_path)
    print("Comparison table:", csv_path)
    print("Merged per-claim metrics:", args.output_dir / "per_claim_metrics_merged.csv")


if __name__ == "__main__":
    main()

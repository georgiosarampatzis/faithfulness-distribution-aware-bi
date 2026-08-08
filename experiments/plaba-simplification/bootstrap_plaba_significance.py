#!/usr/bin/env python3
"""
Paired bootstrap significance analysis for PLABA grouped multi-reference SARI.

Comparisons:
1. selector_v2 vs beam4
2. full_consistency_guard vs beam4
3. full_consistency_guard vs selector_v2
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import evaluate
import numpy as np

SYSTEM_ORDER = ["beam4", "selector_v2", "full_consistency_guard"]
COMPARISONS = [
    ("selector_v2", "beam4"),
    ("full_consistency_guard", "beam4"),
    ("full_consistency_guard", "selector_v2"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beam4", type=Path, required=True)
    parser.add_argument("--selector-v2", type=Path, required=True)
    parser.add_argument("--full-guard", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        example_id = str(row.get("example_id", "")).strip()
        if not example_id:
            raise ValueError(f"{path}:{line_number}: missing example_id")
        if example_id in rows:
            raise ValueError(f"{path}: duplicate example_id {example_id}")
        rows[example_id] = row
    return rows


def extract_source(row: dict[str, Any]) -> str:
    for key in ("source_text", "source", "input_text"):
        value = row.get(key)
        if value is not None:
            return str(value).strip()
    raise KeyError("Could not find source text field.")


def extract_prediction(row: dict[str, Any]) -> str:
    for key in ("prediction", "output_text", "generated_text"):
        value = row.get(key)
        if value is not None:
            return str(value).strip()
    raise KeyError("Could not find prediction field.")


def extract_references(row: dict[str, Any]) -> list[str]:
    references = row.get("references")
    if isinstance(references, list) and references:
        cleaned = [str(reference).strip() for reference in references if str(reference).strip()]
        if cleaned:
            return cleaned
    for key in ("reference", "target_text"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return [str(value).strip()]
    raise KeyError("Could not find references field.")


def percentile_interval(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    return (
        float(np.quantile(values, alpha / 2.0)),
        float(np.quantile(values, 1.0 - alpha / 2.0)),
    )


def bootstrap_two_sided_pvalue(bootstrap_differences: np.ndarray) -> float:
    count_nonpositive = int(np.count_nonzero(bootstrap_differences <= 0.0))
    count_nonnegative = int(np.count_nonzero(bootstrap_differences >= 0.0))
    denominator = len(bootstrap_differences) + 1
    lower_tail = (count_nonpositive + 1) / denominator
    upper_tail = (count_nonnegative + 1) / denominator
    return float(min(1.0, 2.0 * min(lower_tail, upper_tail)))


def holm_adjust(named_pvalues: list[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(named_pvalues, key=lambda item: item[1])
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running_max = 0.0

    for rank, (name, pvalue) in enumerate(ordered):
        candidate = min(1.0, (total - rank) * pvalue)
        running_max = max(running_max, candidate)
        adjusted[name] = running_max

    return adjusted


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 1000:
        raise ValueError("--bootstrap-samples should be at least 1000.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "beam4": args.beam4,
        "selector_v2": args.selector_v2,
        "full_consistency_guard": args.full_guard,
    }
    systems = {name: load_jsonl(path) for name, path in paths.items()}

    example_ids = list(systems["beam4"].keys())
    expected_ids = set(example_ids)

    for system_name, rows in systems.items():
        actual_ids = set(rows.keys())
        if actual_ids != expected_ids:
            missing = sorted(expected_ids - actual_ids)
            extra = sorted(actual_ids - expected_ids)
            raise ValueError(
                f"Example IDs differ for {system_name}. Missing={missing[:5]}, extra={extra[:5]}"
            )

    metric = evaluate.load("sari")
    per_example_rows: list[dict[str, Any]] = []

    print(f"Computing grouped multi-reference SARI for {len(example_ids)} examples...")

    for example_index, example_id in enumerate(example_ids, start=1):
        anchor = systems["beam4"][example_id]
        source = extract_source(anchor)
        references = extract_references(anchor)
        output_row: dict[str, Any] = {"example_id": example_id}

        for system_name in SYSTEM_ORDER:
            row = systems[system_name][example_id]
            if extract_source(row) != source:
                raise ValueError(f"Source mismatch for {example_id}/{system_name}")
            prediction = extract_prediction(row)
            score = float(
                metric.compute(
                    sources=[source],
                    predictions=[prediction],
                    references=[references],
                )["sari"]
            )
            output_row[f"{system_name}_sari"] = score

        per_example_rows.append(output_row)

        if example_index % 25 == 0 or example_index == len(example_ids):
            print(f"Scored: {example_index}/{len(example_ids)}")

    score_arrays = {
        system_name: np.asarray(
            [row[f"{system_name}_sari"] for row in per_example_rows],
            dtype=np.float64,
        )
        for system_name in SYSTEM_ORDER
    }

    rng = np.random.default_rng(args.seed)
    n_examples = len(example_ids)
    indices = rng.integers(
        0,
        n_examples,
        size=(args.bootstrap_samples, n_examples),
    )

    system_results: dict[str, Any] = {}
    for system_name in SYSTEM_ORDER:
        scores = score_arrays[system_name]
        bootstrap_means = scores[indices].mean(axis=1)
        ci_low, ci_high = percentile_interval(bootstrap_means)
        system_results[system_name] = {
            "examples": n_examples,
            "mean_grouped_sari": float(scores.mean()),
            "standard_deviation_across_examples": float(scores.std(ddof=1)),
            "bootstrap_95_ci": {"lower": ci_low, "upper": ci_high},
        }

    comparison_results: dict[str, Any] = {}
    raw_pvalues: list[tuple[str, float]] = []

    for system_a, system_b in COMPARISONS:
        comparison_name = f"{system_a}_minus_{system_b}"
        paired_differences = score_arrays[system_a] - score_arrays[system_b]
        bootstrap_differences = paired_differences[indices].mean(axis=1)
        ci_low, ci_high = percentile_interval(bootstrap_differences)
        pvalue = bootstrap_two_sided_pvalue(bootstrap_differences)

        comparison_results[comparison_name] = {
            "system_a": system_a,
            "system_b": system_b,
            "mean_difference": float(paired_differences.mean()),
            "bootstrap_95_ci": {"lower": ci_low, "upper": ci_high},
            "bootstrap_two_sided_pvalue_raw": pvalue,
            "wins": int(np.count_nonzero(paired_differences > 0)),
            "losses": int(np.count_nonzero(paired_differences < 0)),
            "ties": int(np.count_nonzero(paired_differences == 0)),
            "significant_raw_0_05": bool(pvalue < 0.05),
        }
        raw_pvalues.append((comparison_name, pvalue))

    adjusted_pvalues = holm_adjust(raw_pvalues)
    for comparison_name, adjusted_pvalue in adjusted_pvalues.items():
        comparison_results[comparison_name]["holm_adjusted_pvalue"] = adjusted_pvalue
        comparison_results[comparison_name]["significant_holm_0_05"] = bool(
            adjusted_pvalue < 0.05
        )

    report = {
        "protocol": (
            "Paired bootstrap resampling over grouped PLABA validation abstracts. "
            "Each bootstrap sample draws abstracts with replacement and preserves "
            "pairing across systems. The primary measure is per-abstract multi-reference SARI."
        ),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "systems": system_results,
        "comparisons": comparison_results,
        "multiple_comparison_correction": (
            "Holm correction across the three predefined pairwise comparisons."
        ),
        "interpretation_note": (
            "The full-consistency system includes deliberate source fallbacks. "
            "These are validation-set estimates and do not replace untouched test evaluation."
        ),
    }

    report_path = args.output_dir / "paired_bootstrap_grouped_sari.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    scores_path = args.output_dir / "per_example_grouped_sari.csv"
    with scores_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "example_id",
                "beam4_sari",
                "selector_v2_sari",
                "full_consistency_guard_sari",
            ],
        )
        writer.writeheader()
        writer.writerows(per_example_rows)

    comparison_path = args.output_dir / "paired_bootstrap_comparisons.csv"
    with comparison_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "comparison",
                "system_a",
                "system_b",
                "mean_difference",
                "ci_lower",
                "ci_upper",
                "raw_pvalue",
                "holm_adjusted_pvalue",
                "wins",
                "losses",
                "ties",
                "significant_holm_0_05",
            ],
        )
        writer.writeheader()
        for comparison_name, values in comparison_results.items():
            writer.writerow(
                {
                    "comparison": comparison_name,
                    "system_a": values["system_a"],
                    "system_b": values["system_b"],
                    "mean_difference": values["mean_difference"],
                    "ci_lower": values["bootstrap_95_ci"]["lower"],
                    "ci_upper": values["bootstrap_95_ci"]["upper"],
                    "raw_pvalue": values["bootstrap_two_sided_pvalue_raw"],
                    "holm_adjusted_pvalue": values["holm_adjusted_pvalue"],
                    "wins": values["wins"],
                    "losses": values["losses"],
                    "ties": values["ties"],
                    "significant_holm_0_05": values["significant_holm_0_05"],
                }
            )

    print("\nPAIRED BOOTSTRAP COMPLETED")
    print(
        f"{'Comparison':<54}{'Diff':>9}{'CI low':>10}"
        f"{'CI high':>10}{'p Holm':>10}{'Sig':>7}"
    )
    for comparison_name, values in comparison_results.items():
        print(
            f"{comparison_name:<54}"
            f"{values['mean_difference']:>9.3f}"
            f"{values['bootstrap_95_ci']['lower']:>10.3f}"
            f"{values['bootstrap_95_ci']['upper']:>10.3f}"
            f"{values['holm_adjusted_pvalue']:>10.4f}"
            f"{str(values['significant_holm_0_05']):>7}"
        )

    print("Report:", report_path)
    print("Per-example scores:", scores_path)
    print("Comparison table:", comparison_path)


if __name__ == "__main__":
    main()

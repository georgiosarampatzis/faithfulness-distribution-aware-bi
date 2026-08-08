#!/usr/bin/env python3
"""
Evaluate predefined reliability-first selection policies on the PLABA
development candidates produced by build_plaba_selector.py v2.

References are used only to evaluate the held-out selections and the source
fallback. Candidate ranking uses cross-validated predicted SARI and
reference-free source-candidate features.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable

import evaluate


CANDIDATE_ORDER = {"beam4": 0, "greedy": 1, "concise": 2}
EPSILON = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        row = json.loads(line)
        required = {
            "example_id",
            "candidate_type",
            "source_text",
            "prediction",
            "references",
            "target_sari",
            "predicted_sari_cv",
            "char_similarity",
            "semantic_cosine",
            "number_precision",
            "number_recall",
            "negation_precision",
            "negation_recall",
            "uncertainty_precision",
            "uncertainty_recall",
            "exact_copy",
        }
        missing = required - set(row)

        if missing:
            raise ValueError(
                f"{path}:{line_number}: missing fields {sorted(missing)}"
            )

        rows.append(row)

    return rows


def is_one(value: Any) -> bool:
    return float(value) >= 1.0 - EPSILON


def best_predicted(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            float(row["predicted_sari_cv"]),
            -CANDIDATE_ORDER[row["candidate_type"]],
        ),
    )


def source_fallback(
    example_id: str,
    source: str,
    references: list[str],
    source_sari: float,
) -> dict[str, Any]:
    return {
        "example_id": example_id,
        "candidate_type": "source_fallback",
        "source_text": source,
        "prediction": source,
        "references": references,
        "target_sari": source_sari,
        "predicted_sari_cv": None,
        "char_similarity": 1.0,
        "semantic_cosine": 1.0,
        "number_precision": 1.0,
        "number_recall": 1.0,
        "negation_precision": 1.0,
        "negation_recall": 1.0,
        "uncertainty_precision": 1.0,
        "uncertainty_recall": 1.0,
        "exact_copy": 1.0,
        "readability_gain": 0.0,
        "is_fallback": True,
    }


def choose_with_guard(
    rows: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
    fallback: dict[str, Any],
    *,
    avoid_exact_within_delta: float | None = None,
) -> dict[str, Any]:
    eligible = [row for row in rows if predicate(row)]

    if not eligible:
        return fallback

    best = best_predicted(eligible)

    if avoid_exact_within_delta is None or not bool(best["exact_copy"]):
        return best

    alternatives = [
        row
        for row in eligible
        if not bool(row["exact_copy"])
        and float(row["predicted_sari_cv"])
        >= float(best["predicted_sari_cv"]) - avoid_exact_within_delta
    ]

    if not alternatives:
        return best

    return max(
        alternatives,
        key=lambda row: (
            float(row["predicted_sari_cv"]),
            -float(row["char_similarity"]),
            -CANDIDATE_ORDER[row["candidate_type"]],
        ),
    )


def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
    similarities = [float(row["char_similarity"]) for row in selected]
    saris = [float(row["target_sari"]) for row in selected]

    def violations(
        precision_key: str,
        recall_key: str | None = None,
    ) -> int:
        count = 0
        for row in selected:
            if not is_one(row[precision_key]):
                count += 1
                continue
            if recall_key is not None and not is_one(row[recall_key]):
                count += 1
        return count

    return {
        "examples": len(selected),
        "mean_heldout_sari": mean(saris),
        "selected_counts": dict(
            Counter(row["candidate_type"] for row in selected)
        ),
        "fallback_count": sum(
            row["candidate_type"] == "source_fallback"
            for row in selected
        ),
        "copy_analysis": {
            "exact_source_copies": sum(
                bool(row["exact_copy"]) for row in selected
            ),
            "mean_source_similarity": mean(similarities),
            "median_source_similarity": median(similarities),
            "similarity_gte_0_99": sum(
                value >= 0.99 for value in similarities
            ),
            "similarity_gte_0_95": sum(
                value >= 0.95 for value in similarities
            ),
            "similarity_gte_0_90": sum(
                value >= 0.90 for value in similarities
            ),
        },
        "reliability_proxies": {
            "unsupported_number_violations": violations(
                "number_precision"
            ),
            "number_omission_violations": violations(
                "number_recall"
            ),
            "negation_consistency_violations": violations(
                "negation_precision",
                "negation_recall",
            ),
            "uncertainty_consistency_violations": violations(
                "uncertainty_precision",
                "uncertainty_recall",
            ),
            "mean_semantic_cosine": mean(
                float(row["semantic_cosine"]) for row in selected
            ),
        },
        "mean_readability_gain_fkgl": mean(
            float(row.get("readability_gain", 0.0))
            for row in selected
        ),
    }


def write_predictions(
    path: Path,
    selected: list[dict[str, Any]],
    policy_name: str,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in selected:
            output = {
                "example_id": row["example_id"],
                "source_text": row["source_text"],
                "prediction": row["prediction"],
                "references": row["references"],
                "selected_from": row["candidate_type"],
                "selection_policy": policy_name,
                "predicted_sari_cv": row.get("predicted_sari_cv"),
                "actual_sari_for_analysis_only": row["target_sari"],
                "char_similarity": row["char_similarity"],
                "semantic_cosine": row["semantic_cosine"],
                "number_precision": row["number_precision"],
                "number_recall": row["number_recall"],
                "negation_precision": row["negation_precision"],
                "negation_recall": row["negation_recall"],
                "uncertainty_precision": row["uncertainty_precision"],
                "uncertainty_recall": row["uncertainty_recall"],
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.features)
    by_example: dict[str, list[dict[str, Any]]] = {}

    for row in rows:
        by_example.setdefault(row["example_id"], []).append(row)

    expected_types = set(CANDIDATE_ORDER)
    for example_id, candidates in by_example.items():
        actual_types = {row["candidate_type"] for row in candidates}
        if actual_types != expected_types:
            raise ValueError(
                f"{example_id}: expected {sorted(expected_types)}, "
                f"found {sorted(actual_types)}"
            )

    print("Computing source-fallback SARI values...")
    sari_metric = evaluate.load("sari")
    fallback_by_example: dict[str, dict[str, Any]] = {}

    for index, example_id in enumerate(sorted(by_example), start=1):
        anchor = by_example[example_id][0]
        source = str(anchor["source_text"])
        references = [str(value) for value in anchor["references"]]

        source_sari = float(
            sari_metric.compute(
                sources=[source],
                predictions=[source],
                references=[references],
            )["sari"]
        )

        fallback_by_example[example_id] = source_fallback(
            example_id,
            source,
            references,
            source_sari,
        )

        if index % 25 == 0 or index == len(by_example):
            print(f"Fallback SARI: {index}/{len(by_example)}")

    policies: dict[
        str,
        tuple[
            Callable[[dict[str, Any]], bool],
            float | None,
        ],
    ] = {
        "sari_only": (
            lambda row: True,
            None,
        ),
        "number_precision_guard": (
            lambda row: is_one(row["number_precision"]),
            None,
        ),
        "number_and_negation_guard": (
            lambda row: (
                is_one(row["number_precision"])
                and is_one(row["negation_precision"])
                and is_one(row["negation_recall"])
            ),
            None,
        ),
        "number_negation_uncertainty_precision_guard": (
            lambda row: (
                is_one(row["number_precision"])
                and is_one(row["negation_precision"])
                and is_one(row["negation_recall"])
                and is_one(row["uncertainty_precision"])
            ),
            None,
        ),
        "full_consistency_guard": (
            lambda row: (
                is_one(row["number_precision"])
                and is_one(row["negation_precision"])
                and is_one(row["negation_recall"])
                and is_one(row["uncertainty_precision"])
                and is_one(row["uncertainty_recall"])
            ),
            None,
        ),
        "full_guard_semantic_0.98": (
            lambda row: (
                is_one(row["number_precision"])
                and is_one(row["negation_precision"])
                and is_one(row["negation_recall"])
                and is_one(row["uncertainty_precision"])
                and is_one(row["uncertainty_recall"])
                and float(row["semantic_cosine"]) >= 0.98
            ),
            None,
        ),
        "number_guard_nonexact_delta_0.25": (
            lambda row: is_one(row["number_precision"]),
            0.25,
        ),
        "number_negation_guard_nonexact_delta_0.25": (
            lambda row: (
                is_one(row["number_precision"])
                and is_one(row["negation_precision"])
                and is_one(row["negation_recall"])
            ),
            0.25,
        ),
    }

    report = {
        "note": (
            "This is development-set policy selection. Candidate quality "
            "predictions are cross-validated. References are used only for "
            "held-out evaluation and source-fallback SARI. Lock one policy "
            "before applying the final trained selector to the untouched test set."
        ),
        "policies": {},
    }

    ordered_examples = sorted(by_example)

    for policy_name, (predicate, delta) in policies.items():
        selected = [
            choose_with_guard(
                by_example[example_id],
                predicate,
                fallback_by_example[example_id],
                avoid_exact_within_delta=delta,
            )
            for example_id in ordered_examples
        ]

        report["policies"][policy_name] = summarize(selected)

        write_predictions(
            args.output_dir / f"{policy_name}_predictions.jsonl",
            selected,
            policy_name,
        )

    report_path = args.output_dir / "reliability_policy_comparison.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"\n{'Policy':<48}"
        f"{'SARI':>8}"
        f"{'Fall':>7}"
        f"{'Exact':>7}"
        f"{'Num+':>7}"
        f"{'Neg':>7}"
        f"{'Unc':>7}"
    )

    for policy_name, values in report["policies"].items():
        reliability = values["reliability_proxies"]
        copy = values["copy_analysis"]

        print(
            f"{policy_name:<48}"
            f"{values['mean_heldout_sari']:>8.3f}"
            f"{values['fallback_count']:>7}"
            f"{copy['exact_source_copies']:>7}"
            f"{reliability['unsupported_number_violations']:>7}"
            f"{reliability['negation_consistency_violations']:>7}"
            f"{reliability['uncertainty_consistency_violations']:>7}"
        )

    print("Saved:", report_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path):
    rows = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        required = {
            "example_id",
            "candidate_type",
            "target_sari",
            "char_similarity",
            "exact_copy",
            "number_precision",
        }
        missing = required - set(row)
        if missing:
            raise ValueError(
                f"{path}:{line_no}: missing {sorted(missing)}"
            )
        rows.append(row)
    return rows


def best_by_sari(rows):
    order = {"beam4": 0, "greedy": 1, "concise": 2}
    return max(
        rows,
        key=lambda row: (
            float(row["target_sari"]),
            -order.get(row["candidate_type"], 99),
        ),
    )


def evaluate_constraint(by_example, name, predicate):
    selected = []
    unavailable = 0

    for example_id in sorted(by_example):
        candidates = [
            row for row in by_example[example_id]
            if predicate(row)
        ]
        if not candidates:
            unavailable += 1
            candidates = by_example[example_id]
        selected.append(best_by_sari(candidates))

    similarities = [
        float(row["char_similarity"]) for row in selected
    ]
    number_violations = sum(
        float(row["number_precision"]) < 0.999999
        for row in selected
    )

    return {
        "constraint": name,
        "examples": len(selected),
        "examples_without_eligible_candidate": unavailable,
        "oracle_mean_sari": mean(
            float(row["target_sari"]) for row in selected
        ),
        "selected_counts": dict(
            Counter(row["candidate_type"] for row in selected)
        ),
        "copy_analysis": {
            "exact_source_copies": sum(
                float(row["exact_copy"]) >= 0.5
                for row in selected
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
        "number_precision_violations": number_violations,
    }


def main():
    args = parse_args()
    rows = load_rows(args.features)

    by_example = {}
    for row in rows:
        by_example.setdefault(row["example_id"], []).append(row)

    constraints = [
        ("unconstrained", lambda row: True),
        (
            "number_precision_1",
            lambda row: float(row["number_precision"]) >= 0.999999,
        ),
        (
            "not_exact_copy",
            lambda row: float(row["exact_copy"]) < 0.5,
        ),
        (
            "similarity_lt_0.99",
            lambda row: float(row["char_similarity"]) < 0.99,
        ),
        (
            "similarity_lt_0.95",
            lambda row: float(row["char_similarity"]) < 0.95,
        ),
        (
            "similarity_lt_0.90",
            lambda row: float(row["char_similarity"]) < 0.90,
        ),
        (
            "num1_and_similarity_lt_0.99",
            lambda row: (
                float(row["number_precision"]) >= 0.999999
                and float(row["char_similarity"]) < 0.99
            ),
        ),
        (
            "num1_and_similarity_lt_0.95",
            lambda row: (
                float(row["number_precision"]) >= 0.999999
                and float(row["char_similarity"]) < 0.95
            ),
        ),
        (
            "num1_and_similarity_lt_0.90",
            lambda row: (
                float(row["number_precision"]) >= 0.999999
                and float(row["char_similarity"]) < 0.90
            ),
        ),
    ]

    report = {
        "examples": len(by_example),
        "candidate_rows": len(rows),
        "note": (
            "References are used here only for oracle ceiling analysis. "
            "These results are not deployable selector performance."
        ),
        "constraints": [
            evaluate_constraint(by_example, name, predicate)
            for name, predicate in constraints
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"{'Constraint':<31}"
        f"{'SARI':>9}"
        f"{'NoCand':>8}"
        f"{'Exact':>8}"
        f"{'>=.99':>8}"
        f"{'>=.95':>8}"
        f"{'NumV':>7}"
    )
    for item in report["constraints"]:
        copy = item["copy_analysis"]
        print(
            f"{item['constraint']:<31}"
            f"{item['oracle_mean_sari']:>9.3f}"
            f"{item['examples_without_eligible_candidate']:>8}"
            f"{copy['exact_source_copies']:>8}"
            f"{copy['similarity_gte_0_99']:>8}"
            f"{copy['similarity_gte_0_95']:>8}"
            f"{item['number_precision_violations']:>7}"
        )

    print("Saved:", args.output)


if __name__ == "__main__":
    main()

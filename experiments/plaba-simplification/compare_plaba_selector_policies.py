#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

CANDIDATE_ORDER = {"beam4": 0, "greedy": 1, "concise": 2}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def load_jsonl(path: Path):
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        required = {
            "example_id", "candidate_type", "source_text", "prediction",
            "references", "target_sari", "predicted_sari_cv",
            "char_similarity", "number_precision",
        }
        missing = required - set(row)
        if missing:
            raise ValueError(
                f"{path}:{line_number}: missing fields {sorted(missing)}"
            )
        rows.append(row)
    return rows


def reliable_pool(rows):
    reliable = [
        row for row in rows
        if float(row.get("number_precision", 0.0)) >= 0.999999
    ]
    return reliable if reliable else rows


def predicted_best(rows):
    return max(
        rows,
        key=lambda row: (
            float(row["predicted_sari_cv"]),
            -CANDIDATE_ORDER[row["candidate_type"]],
        ),
    )


def select_sari_only(rows):
    return predicted_best(rows)


def select_delta_lowcopy(rows, delta):
    pool = reliable_pool(rows)
    best_predicted = max(float(row["predicted_sari_cv"]) for row in pool)
    shortlist = [
        row for row in pool
        if float(row["predicted_sari_cv"]) >= best_predicted - delta
    ]
    return min(
        shortlist,
        key=lambda row: (
            float(row["char_similarity"]),
            -float(row["predicted_sari_cv"]),
            CANDIDATE_ORDER[row["candidate_type"]],
        ),
    )


def select_avoid_threshold(rows, threshold):
    pool = reliable_pool(rows)
    preferred = [
        row for row in pool
        if float(row["char_similarity"]) < threshold
        and float(row.get("exact_copy", 0.0)) < 0.5
    ]
    return predicted_best(preferred if preferred else pool)


def select_reliable_delta(rows, delta=0.5):
    pool = reliable_pool(rows)
    best_predicted = max(float(row["predicted_sari_cv"]) for row in pool)
    shortlist = [
        row for row in pool
        if float(row["predicted_sari_cv"]) >= best_predicted - delta
    ]

    non_near = [
        row for row in shortlist
        if float(row["char_similarity"]) < 0.99
        and float(row.get("exact_copy", 0.0)) < 0.5
    ]
    if non_near:
        shortlist = non_near

    return min(
        shortlist,
        key=lambda row: (
            float(row["char_similarity"]),
            -float(row.get("semantic_cosine", 0.0)),
            -float(row["predicted_sari_cv"]),
            CANDIDATE_ORDER[row["candidate_type"]],
        ),
    )


def summarize(selected):
    similarities = [float(row["char_similarity"]) for row in selected]
    saris = [float(row["target_sari"]) for row in selected]
    num_precision = [float(row.get("number_precision", 0.0)) for row in selected]
    readability = [float(row.get("readability_gain", 0.0)) for row in selected]
    semantic = [float(row.get("semantic_cosine", 0.0)) for row in selected]

    return {
        "examples": len(selected),
        "mean_heldout_sari": mean(saris),
        "selected_counts": dict(Counter(
            row["candidate_type"] for row in selected
        )),
        "copy_analysis": {
            "exact_source_copies": sum(
                float(row.get("exact_copy", 0.0)) >= 0.5
                for row in selected
            ),
            "mean_source_similarity": mean(similarities),
            "median_source_similarity": median(similarities),
            "similarity_gte_0_99": sum(x >= 0.99 for x in similarities),
            "similarity_gte_0_95": sum(x >= 0.95 for x in similarities),
            "similarity_gte_0_90": sum(x >= 0.90 for x in similarities),
        },
        "faithfulness_proxies": {
            "mean_number_precision": mean(num_precision),
            "number_precision_violations": sum(x < 0.999999 for x in num_precision),
            "mean_semantic_cosine": mean(semantic),
        },
        "mean_readability_gain_fkgl": mean(readability),
    }


def write_predictions(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            output = {
                "example_id": row["example_id"],
                "source_text": row["source_text"],
                "prediction": row["prediction"],
                "references": row["references"],
                "selected_from": row["candidate_type"],
                "selector_fold": row.get("fold"),
                "predicted_sari_cv": row["predicted_sari_cv"],
                "actual_sari_for_analysis_only": row["target_sari"],
                "char_similarity": row["char_similarity"],
                "number_precision": row.get("number_precision"),
                "semantic_cosine": row.get("semantic_cosine"),
            }
            for key in ("pmid", "question"):
                if key in row:
                    output[key] = row[key]
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.features)

    by_example = {}
    for row in rows:
        by_example.setdefault(row["example_id"], []).append(row)

    for example_id, candidates in by_example.items():
        types = {row["candidate_type"] for row in candidates}
        expected = set(CANDIDATE_ORDER)
        if types != expected:
            raise ValueError(
                f"{example_id}: expected {sorted(expected)}, got {sorted(types)}"
            )

    policies = {
        "sari-only": select_sari_only,
        "delta-0.25-lowcopy": lambda rows: select_delta_lowcopy(rows, 0.25),
        "delta-0.50-lowcopy": lambda rows: select_delta_lowcopy(rows, 0.50),
        "delta-1.00-lowcopy": lambda rows: select_delta_lowcopy(rows, 1.00),
        "avoid-similarity-0.99": lambda rows: select_avoid_threshold(rows, 0.99),
        "avoid-similarity-0.95": lambda rows: select_avoid_threshold(rows, 0.95),
        "reliable-delta-0.50": lambda rows: select_reliable_delta(rows, 0.50),
    }

    report = {
        "note": (
            "This is development-set policy analysis. Candidate SARI predictions "
            "remain cross-validated, but choosing a policy after comparing this "
            "report is hyperparameter tuning on development data. The selected "
            "policy must be locked before test evaluation."
        ),
        "policies": {},
    }

    ordered_examples = sorted(by_example)
    for policy_name, selector in policies.items():
        selected = [
            selector(by_example[example_id])
            for example_id in ordered_examples
        ]
        report["policies"][policy_name] = summarize(selected)
        write_predictions(
            args.output_dir / f"{policy_name}-predictions.jsonl",
            selected,
        )

    report_path = args.output_dir / "policy_comparison.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Saved:", report_path)


if __name__ == "__main__":
    main()

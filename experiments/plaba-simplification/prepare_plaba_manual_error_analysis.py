#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

EPS = 1e-9
CHECKS = (
    "number_precision",
    "negation_precision",
    "negation_recall",
    "uncertainty_precision",
    "uncertainty_recall",
)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--beam4", type=Path, required=True)
    p.add_argument("--selector", type=Path, required=True)
    p.add_argument("--guard", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sample-size", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not str(row.get("example_id", "")).strip():
            raise ValueError(f"{path}:{n}: missing example_id")
        rows.append(row)
    return rows


def index(rows: list[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for row in rows:
        eid = str(row["example_id"])
        if eid in out:
            raise ValueError(f"{path}: duplicate example_id {eid}")
        out[eid] = row
    return out


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def get_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in row:
            return norm(row[key])
    raise KeyError(f"Missing one of {keys}")


def pred(row: dict[str, Any]) -> str:
    return get_text(row, ("prediction", "output_text", "generated_text"))


def source(row: dict[str, Any]) -> str:
    return get_text(row, ("source_text", "source", "input_text"))


def refs(row: dict[str, Any]) -> list[str]:
    values = row.get("references")
    if isinstance(values, list):
        return [norm(x) for x in values if norm(x)]
    for key in ("reference", "target_text"):
        if key in row and norm(row[key]):
            return [norm(row[key])]
    return []


def failures(row: dict[str, Any]) -> list[str]:
    return [key for key in CHECKS if float(row.get(key, 0.0)) < 1.0 - EPS]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0].keys()) if rows else ["example_id"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_sample(
    rows: list[dict[str, Any]], size: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[row["category"]].append(row)
    for values in groups.values():
        rng.shuffle(values)

    selected = list(groups["source_fallback"])
    quotas = {
        "guard_differs_from_beam4": 20,
        "selector_differs_from_beam4": 15,
        "agreement_control": 8,
    }
    for category, quota in quotas.items():
        remaining = max(0, size - len(selected))
        selected.extend(groups[category][: min(quota, remaining)])

    used = {row["example_id"] for row in selected}
    pool = [
        row
        for category in (
            "guard_differs_from_beam4",
            "selector_differs_from_beam4",
            "agreement_control",
        )
        for row in groups[category]
        if row["example_id"] not in used
    ]
    rng.shuffle(pool)
    selected.extend(pool[: max(0, size - len(selected))])
    rng.shuffle(selected)
    return selected


def main() -> None:
    a = args()
    a.output_dir.mkdir(parents=True, exist_ok=True)

    beam = index(read_jsonl(a.beam4), a.beam4)
    selector = index(read_jsonl(a.selector), a.selector)
    guard = index(read_jsonl(a.guard), a.guard)
    if set(beam) != set(selector) or set(beam) != set(guard):
        raise ValueError("Example IDs differ across prediction files")

    feature_map: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in read_jsonl(a.features):
        feature_map[str(row["example_id"])][str(row["candidate_type"])] = row

    comparisons = []
    fallback_rows = []
    for eid in sorted(beam):
        b, s, g = beam[eid], selector[eid], guard[eid]
        bp, sp, gp = pred(b), pred(s), pred(g)
        selected_from = str(g.get("selected_from", ""))

        if selected_from == "source_fallback":
            category = "source_fallback"
        elif gp != bp:
            category = "guard_differs_from_beam4"
        elif sp != bp:
            category = "selector_differs_from_beam4"
        else:
            category = "agreement_control"

        bfeat = feature_map[eid].get("beam4", {})
        comparisons.append(
            {
                "example_id": eid,
                "category": category,
                "source_text": source(b),
                "references": " ||| ".join(refs(b)),
                "beam4_prediction": bp,
                "selector_prediction": sp,
                "guard_prediction": gp,
                "selector_selected_from": s.get("selected_from", ""),
                "guard_selected_from": selected_from,
                "beam4_failed_guard_checks": ";".join(failures(bfeat)),
                "beam4_equals_selector": int(bp == sp),
                "beam4_equals_guard": int(bp == gp),
                "selector_equals_guard": int(sp == gp),
            }
        )

        if selected_from == "source_fallback":
            for ctype in ("beam4", "greedy", "concise"):
                feat = feature_map[eid][ctype]
                fallback_rows.append(
                    {
                        "example_id": eid,
                        "candidate_type": ctype,
                        "prediction": pred(feat),
                        "failed_guard_checks": ";".join(failures(feat)),
                        "number_precision": feat.get("number_precision"),
                        "number_recall": feat.get("number_recall"),
                        "negation_precision": feat.get("negation_precision"),
                        "negation_recall": feat.get("negation_recall"),
                        "uncertainty_precision": feat.get("uncertainty_precision"),
                        "uncertainty_recall": feat.get("uncertainty_recall"),
                        "semantic_cosine": feat.get("semantic_cosine"),
                    }
                )

    sample = make_sample(comparisons, a.sample_size, a.seed)
    rng = random.Random(a.seed)
    blinded, references, key = [], [], []

    for i, row in enumerate(sample, 1):
        systems = ["beam4", "selector_v2", "full_guard"]
        rng.shuffle(systems)
        mapping = dict(zip(("A", "B", "C"), systems))
        outputs = {
            "beam4": row["beam4_prediction"],
            "selector_v2": row["selector_prediction"],
            "full_guard": row["guard_prediction"],
        }
        sid = f"P{i:03d}"
        blinded.append(
            {
                "sample_id": sid,
                "example_id": row["example_id"],
                "case_category": row["category"],
                "source_text": row["source_text"],
                "output_A": outputs[mapping["A"]],
                "output_B": outputs[mapping["B"]],
                "output_C": outputs[mapping["C"]],
                "preferred_output": "",
                "A_faithfulness_1_5": "",
                "B_faithfulness_1_5": "",
                "C_faithfulness_1_5": "",
                "A_readability_1_5": "",
                "B_readability_1_5": "",
                "C_readability_1_5": "",
                "A_completeness_1_5": "",
                "B_completeness_1_5": "",
                "C_completeness_1_5": "",
                "number_error_A": "",
                "number_error_B": "",
                "number_error_C": "",
                "negation_error_A": "",
                "negation_error_B": "",
                "negation_error_C": "",
                "uncertainty_error_A": "",
                "uncertainty_error_B": "",
                "uncertainty_error_C": "",
                "unsupported_information_A": "",
                "unsupported_information_B": "",
                "unsupported_information_C": "",
                "important_omission_A": "",
                "important_omission_B": "",
                "important_omission_C": "",
                "annotator_notes": "",
            }
        )
        references.append(
            {
                "sample_id": sid,
                "example_id": row["example_id"],
                "human_references": row["references"],
            }
        )
        key.append(
            {
                "sample_id": sid,
                "example_id": row["example_id"],
                "output_A_system": mapping["A"],
                "output_B_system": mapping["B"],
                "output_C_system": mapping["C"],
                "case_category": row["category"],
                "selector_selected_from": row["selector_selected_from"],
                "guard_selected_from": row["guard_selected_from"],
                "beam4_failed_guard_checks": row["beam4_failed_guard_checks"],
            }
        )

    write_csv(a.output_dir / "all_test_comparisons.csv", comparisons)
    write_csv(a.output_dir / "fallback_reasons.csv", fallback_rows)
    write_csv(a.output_dir / "manual_annotation_blinded.csv", blinded)
    write_csv(a.output_dir / "manual_annotation_references.csv", references)
    write_csv(a.output_dir / "manual_annotation_key.csv", key)

    summary = {
        "examples": len(comparisons),
        "sample_size": len(sample),
        "seed": a.seed,
        "all_case_counts": dict(Counter(x["category"] for x in comparisons)),
        "sample_case_counts": dict(Counter(x["category"] for x in sample)),
        "fallback_candidate_failure_counts": dict(
            Counter(
                check
                for row in fallback_rows
                for check in str(row["failed_guard_checks"]).split(";")
                if check
            )
        ),
        "instructions": {
            "preferred_output": "A, B, C, TIE, or NONE",
            "ratings": "1=worst, 5=best",
            "binary_errors": "1=yes, 0=no",
            "blinding": (
                "Keep manual_annotation_key.csv hidden until annotation is complete. "
                "Use the references sheet only for second-pass adjudication."
            ),
        },
    }
    (a.output_dir / "manual_error_analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nMANUAL ERROR-ANALYSIS PACKAGE COMPLETED")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for name in (
        "all_test_comparisons.csv",
        "fallback_reasons.csv",
        "manual_annotation_blinded.csv",
        "manual_annotation_references.csv",
        "manual_annotation_key.csv",
        "manual_error_analysis_summary.json",
    ):
        print("Saved:", a.output_dir / name)


if __name__ == "__main__":
    main()

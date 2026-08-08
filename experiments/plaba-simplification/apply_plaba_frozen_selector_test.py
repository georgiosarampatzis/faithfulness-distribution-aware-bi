#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import build_plaba_selector as bps  # noqa: E402

CANDIDATE_ORDER = {"beam4": 0, "greedy": 1, "concise": 2}
EPSILON = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the frozen PLABA selector-v2 model and the locked "
            "full-consistency guard to untouched test candidates. "
            "No references or SARI targets are used."
        )
    )
    parser.add_argument("--beam4", type=Path, required=True)
    parser.add_argument("--greedy", type=Path, required=True)
    parser.add_argument("--concise", type=Path, required=True)
    parser.add_argument("--selector-model", type=Path, required=True)
    parser.add_argument("--selector-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--semantic-model", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_one(value: Any) -> bool:
    return float(value) >= 1.0 - EPSILON


def passes_full_consistency_guard(row: dict[str, Any]) -> bool:
    return (
        is_one(row["number_precision"])
        and is_one(row["negation_precision"])
        and is_one(row["negation_recall"])
        and is_one(row["uncertainty_precision"])
        and is_one(row["uncertainty_recall"])
    )


def choose_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            float(row["predicted_sari"]),
            -CANDIDATE_ORDER[row["candidate_type"]],
        ),
    )


def clean_feature_record(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "original_row"}


def selected_output(selected: dict[str, Any], *, policy: str) -> dict[str, Any]:
    output = dict(selected["original_row"])
    output.update(
        {
            "selected_from": selected["candidate_type"],
            "selection_policy": policy,
            "predicted_sari": float(selected["predicted_sari"]),
            "char_similarity": float(selected["char_similarity"]),
            "semantic_cosine": float(selected["semantic_cosine"]),
            "number_precision": float(selected["number_precision"]),
            "number_recall": float(selected["number_recall"]),
            "negation_precision": float(selected["negation_precision"]),
            "negation_recall": float(selected["negation_recall"]),
            "uncertainty_precision": float(selected["uncertainty_precision"]),
            "uncertainty_recall": float(selected["uncertainty_recall"]),
            "is_source_fallback": False,
        }
    )
    return output


def fallback_output(
    anchor: dict[str, Any], *, example_id: str, source: str
) -> dict[str, Any]:
    output = dict(anchor["original_row"])
    output["prediction"] = source
    output.update(
        {
            "example_id": example_id,
            "source_text": source,
            "selected_from": "source_fallback",
            "selection_policy": "full_consistency_guard",
            "predicted_sari": None,
            "char_similarity": 1.0,
            "semantic_cosine": 1.0,
            "number_precision": 1.0,
            "number_recall": 1.0,
            "negation_precision": 1.0,
            "negation_recall": 1.0,
            "uncertainty_precision": 1.0,
            "uncertainty_recall": 1.0,
            "is_source_fallback": True,
        }
    )
    return output


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(args.selector_config.read_text(encoding="utf-8"))
    feature_names = list(config["feature_names"])
    semantic_model_name = args.semantic_model or config["semantic_model"]

    systems = {
        "beam4": bps.load_jsonl(args.beam4),
        "greedy": bps.load_jsonl(args.greedy),
        "concise": bps.load_jsonl(args.concise),
    }

    example_ids = list(systems["beam4"])
    expected_ids = set(example_ids)
    for candidate_type, rows in systems.items():
        if set(rows) != expected_ids:
            raise ValueError(
                f"Example IDs differ for {candidate_type}: "
                f"expected {len(expected_ids)}, found {len(rows)}"
            )

    records: list[dict[str, Any]] = []
    semantic_sources: list[str] = []
    semantic_predictions: list[str] = []

    for example_id in example_ids:
        anchor = systems["beam4"][example_id]
        source = bps.source_of(anchor)

        for candidate_type in bps.CANDIDATES:
            row = systems[candidate_type][example_id]
            if bps.norm_space(bps.source_of(row)) != bps.norm_space(source):
                raise ValueError(f"Source mismatch for {example_id}/{candidate_type}")

            prediction = bps.prediction_of(row)
            record: dict[str, Any] = {
                "example_id": example_id,
                "candidate_type": candidate_type,
                "source_text": source,
                "prediction": prediction,
                "original_row": row,
            }
            record.update(bps.feature_dict(source, prediction))
            for candidate_name in bps.CANDIDATES:
                record[f"candidate_is_{candidate_name}"] = float(
                    candidate_type == candidate_name
                )

            records.append(record)
            semantic_sources.append(source)
            semantic_predictions.append(prediction)

    print(f"Examples: {len(example_ids)}")
    print(f"Candidate rows: {len(records)}")
    print(f"Loading semantic model: {semantic_model_name}")

    encoder = SentenceTransformer(semantic_model_name, device=args.device)
    source_embeddings = encoder.encode(
        [f"query: {text}" for text in semantic_sources],
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    prediction_embeddings = encoder.encode(
        [f"query: {text}" for text in semantic_predictions],
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    for record, score in zip(
        records, np.sum(source_embeddings * prediction_embeddings, axis=1)
    ):
        record["semantic_cosine"] = float(score)

    missing_features = sorted(
        {
            feature
            for record in records
            for feature in feature_names
            if feature not in record
        }
    )
    if missing_features:
        raise ValueError(f"Missing selector features: {missing_features}")

    X = np.asarray(
        [[float(record[name]) for name in feature_names] for record in records],
        dtype=np.float64,
    )

    selector = joblib.load(args.selector_model)
    expected_feature_count = getattr(selector, "n_features_in_", None)
    if expected_feature_count is not None and int(expected_feature_count) != len(
        feature_names
    ):
        raise ValueError(
            "Selector/config feature mismatch: "
            f"model expects {expected_feature_count}, "
            f"config contains {len(feature_names)}"
        )

    predicted_sari = selector.predict(X)
    for record, score in zip(records, predicted_sari):
        record["predicted_sari"] = float(score)

    by_example: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_example.setdefault(record["example_id"], []).append(record)

    selector_outputs: list[dict[str, Any]] = []
    guarded_outputs: list[dict[str, Any]] = []
    selector_counts: Counter[str] = Counter()
    guarded_counts: Counter[str] = Counter()
    fallback_count = 0

    for example_id in example_ids:
        candidates = by_example[example_id]
        if {row["candidate_type"] for row in candidates} != set(CANDIDATE_ORDER):
            raise ValueError(f"{example_id}: incomplete candidate set")

        selector_choice = choose_best(candidates)
        selector_counts[selector_choice["candidate_type"]] += 1
        selector_outputs.append(selected_output(selector_choice, policy="selector_v2"))

        eligible = [row for row in candidates if passes_full_consistency_guard(row)]
        if eligible:
            guarded_choice = choose_best(eligible)
            guarded_counts[guarded_choice["candidate_type"]] += 1
            guarded_outputs.append(
                selected_output(
                    guarded_choice, policy="full_consistency_guard"
                )
            )
        else:
            fallback_count += 1
            guarded_counts["source_fallback"] += 1
            guarded_outputs.append(
                fallback_output(
                    candidates[0],
                    example_id=example_id,
                    source=candidates[0]["source_text"],
                )
            )

    features_path = args.output_dir / "test_selector_features.jsonl"
    selector_path = args.output_dir / "test_selector_v2_predictions.jsonl"
    guard_path = args.output_dir / "test_full_consistency_guard_predictions.jsonl"

    write_jsonl(features_path, [clean_feature_record(record) for record in records])
    write_jsonl(selector_path, selector_outputs)
    write_jsonl(guard_path, guarded_outputs)

    summary = {
        "method": (
            "Frozen selector-v2 and locked full-consistency guard. "
            "No references or SARI targets were read or computed."
        ),
        "examples": len(example_ids),
        "candidate_rows": len(records),
        "semantic_model": semantic_model_name,
        "feature_count": len(feature_names),
        "selector_selected_counts": dict(selector_counts),
        "guard_selected_counts": dict(guarded_counts),
        "guard_fallback_count": fallback_count,
        "input_sha256": {
            "beam4": sha256(args.beam4),
            "greedy": sha256(args.greedy),
            "concise": sha256(args.concise),
            "selector_model": sha256(args.selector_model),
            "selector_config": sha256(args.selector_config),
        },
        "output_sha256": {
            "features": sha256(features_path),
            "selector_predictions": sha256(selector_path),
            "guard_predictions": sha256(guard_path),
        },
    }

    summary_path = args.output_dir / "test_frozen_selection_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nFROZEN TEST SELECTION COMPLETED")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Features: {features_path}")
    print(f"Selector predictions: {selector_path}")
    print(f"Guard predictions: {guard_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

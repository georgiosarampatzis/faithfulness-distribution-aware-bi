#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean, median

import evaluate


ORDER = ["beam4", "greedy", "balanced", "concise"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beam4", type=Path, required=True)
    parser.add_argument("--greedy", type=Path, required=True)
    parser.add_argument("--balanced", type=Path, required=True)
    parser.add_argument("--concise", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path):
    rows = {}
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        example_id = str(row.get("example_id", "")).strip()
        if not example_id:
            raise ValueError(f"{path}:{line_no}: missing example_id")
        if example_id in rows:
            raise ValueError(f"{path}: duplicate example_id {example_id}")
        rows[example_id] = row
    return rows


def source_of(row):
    for key in ("source_text", "source", "input_text"):
        value = row.get(key)
        if value is not None:
            return str(value).strip()
    raise KeyError("No source field found")


def prediction_of(row):
    for key in ("prediction", "output_text", "generated_text"):
        value = row.get(key)
        if value is not None:
            return str(value).strip()
    raise KeyError("No prediction field found")


def references_of(row):
    refs = row.get("references")
    if isinstance(refs, list) and refs:
        return [str(ref).strip() for ref in refs if str(ref).strip()]
    for key in ("reference", "target_text"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return [str(value).strip()]
    raise KeyError("No references found")


def normalize_space(text):
    return re.sub(r"\s+", " ", text).strip()


def char_similarity(source, prediction):
    return SequenceMatcher(
        None,
        normalize_space(source).lower(),
        normalize_space(prediction).lower(),
        autojunk=False,
    ).ratio()


def summarize_selected(selected):
    similarities = [item["similarity"] for item in selected]
    return {
        "mean_sari": mean(item["sari"] for item in selected),
        "selected_counts": dict(Counter(
            item["candidate_type"] for item in selected
        )),
        "copy_analysis": {
            "exact_source_copies": sum(
                normalize_space(item["source"]) ==
                normalize_space(item["prediction"])
                for item in selected
            ),
            "mean_source_similarity": mean(similarities),
            "median_source_similarity": median(similarities),
            "similarity_gte_0_99": sum(x >= 0.99 for x in similarities),
            "similarity_gte_0_95": sum(x >= 0.95 for x in similarities),
            "similarity_gte_0_90": sum(x >= 0.90 for x in similarities),
        },
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "beam4": args.beam4,
        "greedy": args.greedy,
        "balanced": args.balanced,
        "concise": args.concise,
    }
    systems = {name: load_jsonl(path) for name, path in paths.items()}

    example_ids = list(systems["beam4"])
    expected = set(example_ids)
    for name, rows in systems.items():
        if set(rows) != expected:
            raise ValueError(f"Example IDs differ for {name}")

    sari_metric = evaluate.load("sari")
    per_example = {}
    candidate_scores = {name: [] for name in ORDER}
    strict_wins = Counter()
    tied_best_counts = Counter()
    tie_examples = 0
    balanced_vs_beam = Counter()

    print("Computing four-candidate SARI scores...")
    for index, example_id in enumerate(example_ids, start=1):
        anchor = systems["beam4"][example_id]
        source = source_of(anchor)
        references = references_of(anchor)
        items = []

        for name in ORDER:
            row = systems[name][example_id]
            row_source = source_of(row)
            if normalize_space(row_source) != normalize_space(source):
                raise ValueError(f"Source mismatch: {example_id}/{name}")

            prediction = prediction_of(row)
            sari = float(sari_metric.compute(
                sources=[source],
                predictions=[prediction],
                references=[references],
            )["sari"])
            similarity = char_similarity(source, prediction)

            item = {
                "candidate_type": name,
                "source": source,
                "prediction": prediction,
                "references": references,
                "sari": sari,
                "similarity": similarity,
                "original_row": row,
            }
            items.append(item)
            candidate_scores[name].append(sari)

        max_sari = max(item["sari"] for item in items)
        winners = [
            item for item in items
            if abs(item["sari"] - max_sari) < 1e-9
        ]
        for item in winners:
            tied_best_counts[item["candidate_type"]] += 1
        if len(winners) == 1:
            strict_wins[winners[0]["candidate_type"]] += 1
        else:
            tie_examples += 1

        beam_sari = next(
            item["sari"] for item in items
            if item["candidate_type"] == "beam4"
        )
        balanced_sari = next(
            item["sari"] for item in items
            if item["candidate_type"] == "balanced"
        )
        if balanced_sari > beam_sari + 1e-9:
            balanced_vs_beam["balanced_better"] += 1
        elif beam_sari > balanced_sari + 1e-9:
            balanced_vs_beam["beam4_better"] += 1
        else:
            balanced_vs_beam["tie"] += 1

        per_example[example_id] = items

        if index % 25 == 0 or index == len(example_ids):
            print(f"Scored: {index}/{len(example_ids)}")

    rank = {name: idx for idx, name in enumerate(ORDER)}

    def choose_oracle(items, predicate=lambda item: True):
        eligible = [item for item in items if predicate(item)]
        if not eligible:
            return None
        return max(
            eligible,
            key=lambda item: (
                item["sari"],
                -rank[item["candidate_type"]],
            ),
        )

    unconstrained = []
    no_exact = []
    lt_099 = []
    lt_095 = []
    unavailable = Counter()

    for example_id in example_ids:
        items = per_example[example_id]
        unconstrained.append(choose_oracle(items))

        selections = {
            "no_exact_copy": choose_oracle(
                items,
                lambda item: normalize_space(item["source"]) !=
                             normalize_space(item["prediction"]),
            ),
            "similarity_lt_0.99": choose_oracle(
                items, lambda item: item["similarity"] < 0.99
            ),
            "similarity_lt_0.95": choose_oracle(
                items, lambda item: item["similarity"] < 0.95
            ),
        }

        for name, selected in selections.items():
            if selected is None:
                unavailable[name] += 1
            else:
                if name == "no_exact_copy":
                    no_exact.append(selected)
                elif name == "similarity_lt_0.99":
                    lt_099.append(selected)
                else:
                    lt_095.append(selected)

    def covered_summary(selected, total):
        summary = summarize_selected(selected) if selected else {}
        summary["covered_examples"] = len(selected)
        summary["total_examples"] = total
        summary["coverage_rate"] = len(selected) / total
        return summary

    report = {
        "examples": len(example_ids),
        "candidate_mean_sari": {
            name: mean(candidate_scores[name]) for name in ORDER
        },
        "strict_wins": dict(strict_wins),
        "best_including_ties": dict(tied_best_counts),
        "tie_examples": tie_examples,
        "balanced_vs_beam4": dict(balanced_vs_beam),
        "four_way_oracle": summarize_selected(unconstrained),
        "constrained_oracles_without_fallback": {
            "no_exact_copy": covered_summary(
                no_exact, len(example_ids)
            ),
            "similarity_lt_0.99": covered_summary(
                lt_099, len(example_ids)
            ),
            "similarity_lt_0.95": covered_summary(
                lt_095, len(example_ids)
            ),
        },
        "unavailable_examples": dict(unavailable),
        "note": (
            "All oracle results use references and are upper-bound analyses, "
            "not deployable selector performance. Constrained results exclude "
            "examples with no eligible candidate instead of silently falling "
            "back to an unconstrained output."
        ),
    }

    summary_path = args.output_dir / "four_candidate_oracle_summary.json"
    summary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    predictions_path = args.output_dir / "four_way_oracle_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for example_id, selected in zip(example_ids, unconstrained):
            output = dict(selected["original_row"])
            output["selected_from"] = selected["candidate_type"]
            output["oracle_sari"] = selected["sari"]
            output["char_similarity"] = selected["similarity"]
            output["candidate_sari"] = {
                item["candidate_type"]: item["sari"]
                for item in per_example[example_id]
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")

    print("\nFOUR-CANDIDATE ORACLE COMPLETED")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Summary:", summary_path)
    print("Predictions:", predictions_path)


if __name__ == "__main__":
    main()

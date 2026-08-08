#!/usr/bin/env python3
import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import evaluate
import numpy as np
from rouge_score import rouge_scorer


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {exc}"
                ) from exc
    return rows


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def rouge_means(predictions: list[str], references: list[list[str]]) -> dict:
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )
    values = {name: [] for name in ["rouge1", "rouge2", "rougeL"]}

    for prediction, refs in zip(predictions, references):
        scores = scorer.score_multi(refs, prediction)
        for name in values:
            values[name].append(scores[name].fmeasure * 100.0)

    return {
        f"{name}_f1": float(np.mean(metric_values))
        for name, metric_values in values.items()
    }


def duplicate_missing_second_reference(
    references: list[list[str]],
) -> list[list[str]]:
    result = []
    for refs in references:
        if not refs:
            raise ValueError("An example has no references.")
        if len(refs) == 1:
            result.append([refs[0], refs[0]])
        else:
            result.append(refs[:2])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate grouped PLABA validation predictions."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.predictions)

    sources = [str(row["source_text"]) for row in rows]
    predictions = [str(row["prediction"]) for row in rows]
    grouped_references = [
        [str(reference) for reference in row["references"]]
        for row in rows
    ]

    sari_metric = evaluate.load("sari")
    bleu_metric = evaluate.load("sacrebleu")

    # Secondary analysis: one example per PMID with all available references.
    grouped_refs_for_bleu = duplicate_missing_second_reference(
        grouped_references
    )
    grouped = {
        "examples": len(rows),
        "protocol": "one prediction per PMID; all available references",
        "sari": float(
            sari_metric.compute(
                sources=sources,
                predictions=predictions,
                references=grouped_references,
            )["sari"]
        ),
        "sacrebleu": float(
            bleu_metric.compute(
                predictions=predictions,
                references=grouped_refs_for_bleu,
            )["score"]
        ),
        **rouge_means(predictions, grouped_references),
    }

    # Primary PLABA-compatible analysis: expand each adaptation as one row.
    row_sources = []
    row_predictions = []
    row_references = []

    for source, prediction, refs in zip(
        sources,
        predictions,
        grouped_references,
    ):
        for reference in refs:
            row_sources.append(source)
            row_predictions.append(prediction)
            row_references.append([reference])

    row_level = {
        "examples": len(row_predictions),
        "protocol": (
            "PLABA-compatible row-level evaluation; repeated source/output "
            "for abstracts with multiple human adaptations"
        ),
        "sari": float(
            sari_metric.compute(
                sources=row_sources,
                predictions=row_predictions,
                references=row_references,
            )["sari"]
        ),
        "sacrebleu": float(
            bleu_metric.compute(
                predictions=row_predictions,
                references=row_references,
            )["score"]
        ),
        **rouge_means(row_predictions, row_references),
    }

    similarities = [
        SequenceMatcher(
            None,
            normalize(source),
            normalize(prediction),
        ).ratio()
        for source, prediction in zip(sources, predictions)
    ]

    exact = sum(
        normalize(source) == normalize(prediction)
        for source, prediction in zip(sources, predictions)
    )

    copy_analysis = {
        "examples": len(rows),
        "exact_source_copies": int(exact),
        "exact_source_copy_rate": float(exact / len(rows)),
        "similarity_gte_0_99": int(sum(x >= 0.99 for x in similarities)),
        "similarity_gte_0_95": int(sum(x >= 0.95 for x in similarities)),
        "similarity_gte_0_90": int(sum(x >= 0.90 for x in similarities)),
        "mean_source_similarity": float(np.mean(similarities)),
        "median_source_similarity": float(np.median(similarities)),
        "outputs_at_512_token_cap": int(
            sum(
                int(row.get("prediction_token_count", 0)) >= 512
                for row in rows
            )
        ),
    }

    result = {
        "system": "FLAN-T5-base 512/512, beam=4",
        "primary_row_level": row_level,
        "secondary_grouped_multireference": grouped,
        "copy_analysis": copy_analysis,
    }

    output_path = args.output_dir / "automatic_metrics.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import hashlib
import json
import re
from collections import Counter
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


DATA_DIR = None
OUTPUT_DIR = None
TOKENIZER_NAME = "google/flan-t5-base"

PROMPT_TEMPLATE = """Rewrite the full biomedical abstract in plain language for a general reader at approximately an eighth-grade reading level. Use the consumer question only as context; do not merely answer the question. Preserve important facts, numbers, comparisons, causal direction, and uncertainty. Explain medical terms and abbreviations. Do not add unsupported information.

Consumer question: {question}

Biomedical abstract: {abstract}

Plain-language adaptation:"""


def clean_text(value) -> str:
    if pd.isna(value):
        return ""

    return re.sub(r"\s+", " ", str(value)).strip()


def make_prompt(question: str, abstract: str) -> str:
    return PROMPT_TEMPLATE.format(
        question=clean_text(question),
        abstract=clean_text(abstract),
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def length_statistics(lengths: list[int]) -> dict:
    values = np.asarray(lengths, dtype=np.int64)

    return {
        "count": int(len(values)),
        "minimum": int(values.min()),
        "mean": float(values.mean()),
        "median": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "maximum": int(values.max()),
        "over_512": int((values > 512).sum()),
        "over_640": int((values > 640).sum()),
        "over_768": int((values > 768).sum()),
        "over_1024": int((values > 1024).sum()),
    }


def unique_preserving_order(values: list[str]) -> list[str]:
    seen = set()
    result = []

    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)

    return result


def load_split(split: str) -> pd.DataFrame:
    path = DATA_DIR / f"{split}.csv"

    frame = pd.read_csv(
        path,
        dtype={"pmid": str},
    )

    required_columns = {
        "question",
        "pmid",
        "input_text",
        "target_text",
        "Adaptation_Version",
        "Question_Type",
    }

    missing = required_columns - set(frame.columns)

    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}"
        )

    for column in required_columns:
        frame[column] = frame[column].map(clean_text)

    missing_pmid = frame["pmid"].eq("")

    # Documented correction for the single missing PMID in train.csv.
    if split == "train" and missing_pmid.any():
        source_prefix = (
            "The most important causes of cough were: "
            "cough variant asthma (CVA)"
        )

        correction_mask = (
            missing_pmid
            & frame["input_text"].str.startswith(source_prefix)
        )

        if int(correction_mask.sum()) != 1:
            raise ValueError(
                "Expected exactly one row for the documented PMID correction."
            )

        frame.loc[correction_mask, "pmid"] = "19957776"
        print("Applied PMID correction: train row 622 -> 19957776")

    if frame["pmid"].eq("").any():
        missing_rows = frame.index[frame["pmid"].eq("")].tolist()
        raise ValueError(
            f"Empty PMID remains in {split}: {missing_rows}"
        )

    if (frame["input_text"] == "").any():
        raise ValueError(f"Empty input_text found in {split}.")

    if (frame["target_text"] == "").any():
        raise ValueError(f"Empty target_text found in {split}.")

    return frame


def create_training_rows(frame: pd.DataFrame) -> list[dict]:
    rows = []

    for index, row in frame.reset_index(drop=True).iterrows():
        rows.append(
            {
                "example_id": f"train-{index + 1:04d}",
                "pmid": row["pmid"],
                "question": row["question"],
                "question_type": row["Question_Type"],
                "adaptation_version": row["Adaptation_Version"],
                "source_text": row["input_text"],
                "prompt": make_prompt(
                    row["question"],
                    row["input_text"],
                ),
                "target_text": row["target_text"],
            }
        )

    return rows


def create_grouped_rows(
    frame: pd.DataFrame,
    split: str,
) -> list[dict]:
    grouped_rows = []

    for pmid, group in frame.groupby("pmid", sort=False):
        sources = unique_preserving_order(
            group["input_text"].tolist()
        )
        questions = unique_preserving_order(
            group["question"].tolist()
        )
        question_types = unique_preserving_order(
            group["Question_Type"].tolist()
        )

        if len(sources) != 1:
            raise ValueError(
                f"PMID {pmid} in {split} has "
                f"{len(sources)} different source texts."
            )

        if len(questions) != 1:
            raise ValueError(
                f"PMID {pmid} in {split} has "
                f"{len(questions)} different questions."
            )

        references = unique_preserving_order(
            group["target_text"].tolist()
        )

        grouped_rows.append(
            {
                "example_id": f"{split}-{len(grouped_rows) + 1:04d}",
                "pmid": pmid,
                "question": questions[0],
                "question_types": question_types,
                "source_text": sources[0],
                "prompt": make_prompt(
                    questions[0],
                    sources[0],
                ),
                "references": references,
                "reference_count": len(references),
                "adaptation_versions": unique_preserving_order(
                    group["Adaptation_Version"].tolist()
                ),
            }
        )

    return grouped_rows


def token_lengths(
    tokenizer,
    texts: list[str],
) -> list[int]:
    lengths = []

    for text in texts:
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            verbose=False,
        )
        lengths.append(len(encoded["input_ids"]))

    return lengths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare PLABA train, validation, and test data."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing train.csv, val.csv, and test.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <data-dir>/processed.",
    )
    parser.add_argument(
        "--tokenizer-name",
        default="google/flan-t5-base",
        help="Tokenizer used for the preparation statistics.",
    )
    return parser.parse_args()


def main() -> None:
    global DATA_DIR, OUTPUT_DIR, TOKENIZER_NAME

    args = parse_args()

    DATA_DIR = args.data_dir
    OUTPUT_DIR = (
        args.output_dir
        if args.output_dir is not None
        else DATA_DIR / "processed"
    )
    TOKENIZER_NAME = args.tokenizer_name

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = {
        split: load_split(split)
        for split in ["train", "val", "test"]
    }

    train_rows = create_training_rows(frames["train"])
    val_rows = create_grouped_rows(frames["val"], "val")
    test_rows = create_grouped_rows(frames["test"], "test")

    output_files = {
        "train": OUTPUT_DIR / "train_rows.jsonl",
        "val": OUTPUT_DIR / "val_grouped.jsonl",
        "test": OUTPUT_DIR / "test_grouped.jsonl",
    }

    write_jsonl(output_files["train"], train_rows)
    write_jsonl(output_files["val"], val_rows)
    write_jsonl(output_files["test"], test_rows)

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        use_fast=True,
    )

    # We are measuring lengths, not feeding them to the model here.
    tokenizer.model_max_length = 1_000_000

    train_prompt_lengths = token_lengths(
        tokenizer,
        [row["prompt"] for row in train_rows],
    )
    train_target_lengths = token_lengths(
        tokenizer,
        [row["target_text"] for row in train_rows],
    )

    val_prompt_lengths = token_lengths(
        tokenizer,
        [row["prompt"] for row in val_rows],
    )
    val_reference_lengths = token_lengths(
        tokenizer,
        [
            reference
            for row in val_rows
            for reference in row["references"]
        ],
    )

    test_prompt_lengths = token_lengths(
        tokenizer,
        [row["prompt"] for row in test_rows],
    )
    test_reference_lengths = token_lengths(
        tokenizer,
        [
            reference
            for row in test_rows
            for reference in row["references"]
        ],
    )

    report = {
        "tokenizer": TOKENIZER_NAME,
        "prompt_template": PROMPT_TEMPLATE,
        "train": {
            "training_rows": len(train_rows),
            "unique_pmids": len(
                {row["pmid"] for row in train_rows}
            ),
            "prompt_token_lengths": length_statistics(
                train_prompt_lengths
            ),
            "target_token_lengths": length_statistics(
                train_target_lengths
            ),
        },
        "validation": {
            "unique_pmids": len(val_rows),
            "total_references": sum(
                row["reference_count"] for row in val_rows
            ),
            "references_per_pmid": dict(
                sorted(
                    Counter(
                        row["reference_count"]
                        for row in val_rows
                    ).items()
                )
            ),
            "prompt_token_lengths": length_statistics(
                val_prompt_lengths
            ),
            "reference_token_lengths": length_statistics(
                val_reference_lengths
            ),
        },
        "test": {
            "unique_pmids": len(test_rows),
            "total_references": sum(
                row["reference_count"] for row in test_rows
            ),
            "references_per_pmid": dict(
                sorted(
                    Counter(
                        row["reference_count"]
                        for row in test_rows
                    ).items()
                )
            ),
            "prompt_token_lengths": length_statistics(
                test_prompt_lengths
            ),
            "reference_token_lengths": length_statistics(
                test_reference_lengths
            ),
        },
    }

    report_path = OUTPUT_DIR / "preparation_report.json"

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(
            report,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    checksum_path = OUTPUT_DIR / "SHA256SUMS.txt"

    with checksum_path.open("w", encoding="utf-8") as handle:
        for path in [
            output_files["train"],
            output_files["val"],
            output_files["test"],
            report_path,
        ]:
            handle.write(
                f"{sha256(path)}  {path.name}\n"
            )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nCreated:")
    for path in output_files.values():
        print(path)
    print(report_path)
    print(checksum_path)


if __name__ == "__main__":
    main()

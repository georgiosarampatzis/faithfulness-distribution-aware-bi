#!/usr/bin/env python3

import json
from collections import Counter
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


DATA_DIR = None
OUTPUT_DIR = None
MODEL_NAME = "google/flan-t5-base"

PROMPT = (
    "Rewrite the following biomedical abstract in plain language for a "
    "non-specialist reader. Preserve important facts, numbers, comparisons, "
    "and uncertainty. Do not add unsupported information.\n\n"
)


def summarize_lengths(lengths: list[int]) -> dict:
    array = np.asarray(lengths, dtype=np.int64)

    return {
        "count": int(len(array)),
        "minimum": int(array.min()),
        "mean": float(array.mean()),
        "median": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "maximum": int(array.max()),
        "over_256": int((array > 256).sum()),
        "over_384": int((array > 384).sum()),
        "over_512": int((array > 512).sum()),
        "over_768": int((array > 768).sum()),
        "over_1024": int((array > 1024).sum()),
    }


def overlap(a: set[str], b: set[str]) -> list[str]:
    return sorted(a & b)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect and audit the PLABA dataset."
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
        required=True,
        help="Directory for the PLABA audit report.",
    )
    return parser.parse_args()


def main() -> None:
    global DATA_DIR, OUTPUT_DIR

    args = parse_args()
    DATA_DIR = args.data_dir
    OUTPUT_DIR = args.output_dir

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = {}

    for split in ["train", "val", "test"]:
        path = DATA_DIR / f"{split}.csv"
        frame = pd.read_csv(path)

        required = {
            "question",
            "pmid",
            "input_text",
            "target_text",
            "Adaptation_Version",
            "Question_Type",
        }

        missing_columns = required - set(frame.columns)

        if missing_columns:
            raise ValueError(
                f"{path} is missing columns: {sorted(missing_columns)}"
            )

        frame["pmid"] = frame["pmid"].astype(str)
        frame["input_text"] = frame["input_text"].fillna("").astype(str)
        frame["target_text"] = frame["target_text"].fillna("").astype(str)
        frame["question"] = frame["question"].fillna("").astype(str)

        frames[split] = frame

    print("Loading tokenizer:", MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    report = {
        "model_tokenizer": MODEL_NAME,
        "tokenizer_model_max_length": int(tokenizer.model_max_length),
        "splits": {},
        "cross_split_overlap": {},
    }

    for split, frame in frames.items():
        source_texts = [
            PROMPT + text.strip()
            for text in frame["input_text"]
        ]

        target_texts = [
            text.strip()
            for text in frame["target_text"]
        ]

        source_lengths = [
            len(
                tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    truncation=False,
                )
            )
            for text in source_texts
        ]

        target_lengths = [
            len(
                tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    truncation=False,
                )
            )
            for text in target_texts
        ]

        adaptation_counts = Counter(
            frame["Adaptation_Version"].astype(str)
        )

        question_type_counts = Counter(
            frame["Question_Type"].astype(str)
        )

        split_report = {
            "rows": int(len(frame)),
            "unique_pmids": int(frame["pmid"].nunique()),
            "unique_questions": int(frame["question"].nunique()),
            "unique_source_texts": int(frame["input_text"].nunique()),
            "unique_target_texts": int(frame["target_text"].nunique()),
            "duplicate_source_rows": int(
                len(frame) - frame["input_text"].nunique()
            ),
            "empty_sources": int(
                (frame["input_text"].str.strip() == "").sum()
            ),
            "empty_targets": int(
                (frame["target_text"].str.strip() == "").sum()
            ),
            "adaptation_versions": dict(
                sorted(adaptation_counts.items())
            ),
            "question_types": dict(
                sorted(question_type_counts.items())
            ),
            "source_token_lengths_with_prompt": summarize_lengths(
                source_lengths
            ),
            "target_token_lengths": summarize_lengths(
                target_lengths
            ),
        }

        report["splits"][split] = split_report

    split_pairs = [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ]

    for left, right in split_pairs:
        left_frame = frames[left]
        right_frame = frames[right]

        pmid_overlap = overlap(
            set(left_frame["pmid"]),
            set(right_frame["pmid"]),
        )

        source_overlap = overlap(
            set(left_frame["input_text"]),
            set(right_frame["input_text"]),
        )

        question_overlap = overlap(
            set(left_frame["question"]),
            set(right_frame["question"]),
        )

        report["cross_split_overlap"][f"{left}_vs_{right}"] = {
            "pmid_count": len(pmid_overlap),
            "source_text_count": len(source_overlap),
            "question_count": len(question_overlap),
            "example_pmids": pmid_overlap[:10],
            "example_questions": question_overlap[:10],
        }

    report_path = OUTPUT_DIR / "plaba_data_audit.json"

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(
            report,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print("\nPLABA DATA AUDIT")
    print("=" * 72)

    for split, info in report["splits"].items():
        print(f"\n{split.upper()}")
        print("Rows:", info["rows"])
        print("Unique PMIDs:", info["unique_pmids"])
        print("Unique questions:", info["unique_questions"])
        print("Duplicate source rows:", info["duplicate_source_rows"])
        print(
            "Source token lengths:",
            info["source_token_lengths_with_prompt"],
        )
        print(
            "Target token lengths:",
            info["target_token_lengths"],
        )

    print("\nCROSS-SPLIT OVERLAP")

    for pair, info in report["cross_split_overlap"].items():
        print(f"\n{pair}")
        print("PMID overlap:", info["pmid_count"])
        print("Exact source overlap:", info["source_text_count"])
        print("Question overlap:", info["question_count"])

    print("\nSaved:", report_path)


if __name__ == "__main__":
    main()

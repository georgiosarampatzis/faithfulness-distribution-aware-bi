#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate plain-language adaptations for grouped PLABA examples."
    )
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batched(rows: list[dict], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start:start + batch_size]


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if not args.data_file.exists():
        raise FileNotFoundError(args.data_file)
    if not args.model_dir.exists():
        raise FileNotFoundError(args.model_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    rows = load_jsonl(args.data_file)
    print("Examples:", len(rows))
    print("Model:", args.model_dir)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Maximum source length:", args.max_source_length)
    print("Maximum generated tokens:", args.max_new_tokens)
    print("Beam size:", args.num_beams)
    print("Length penalty:", args.length_penalty)
    print("Sampling:", args.do_sample)
    print("Top-p:", args.top_p)
    print("Temperature:", args.temperature)
    print("No-repeat n-gram size:", args.no_repeat_ngram_size)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_dir,
        dtype=dtype,
    ).to("cuda")
    model.eval()
    model.config.use_cache = True

    shared_encoder = (
        model.shared.weight.data_ptr()
        == model.encoder.embed_tokens.weight.data_ptr()
    )
    shared_decoder = (
        model.shared.weight.data_ptr()
        == model.decoder.embed_tokens.weight.data_ptr()
    )

    print("Shared = encoder embeddings:", shared_encoder)
    print("Shared = decoder embeddings:", shared_decoder)

    if not shared_encoder or not shared_decoder:
        raise RuntimeError("T5 embedding weights are not tied correctly.")
    if not torch.isfinite(model.shared.weight).all():
        raise RuntimeError("Non-finite values found in shared embeddings.")

    output_path = args.output_dir / "predictions.jsonl"
    metadata_path = args.output_dir / "generation_configuration.json"

    configuration = {
        "data_file": str(args.data_file),
        "model_dir": str(args.model_dir),
        "examples": len(rows),
        "max_source_length": args.max_source_length,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "num_beams": args.num_beams,
        "length_penalty": args.length_penalty,
        "do_sample": args.do_sample,
        "top_p": args.top_p if args.do_sample else None,
        "temperature": args.temperature if args.do_sample else None,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "early_stopping": args.num_beams > 1,
        "seed": args.seed,
        "dtype": str(dtype),
    }

    metadata_path.write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    generated_records = []
    batches = list(batched(rows, args.batch_size))

    for batch in tqdm(batches, desc="Generating"):
        prompts = [str(row["prompt"]) for row in batch]

        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=args.max_source_length,
            return_tensors="pt",
        )
        encoded = {key: value.to("cuda") for key, value in encoded.items()}

        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "num_beams": args.num_beams,
            "do_sample": args.do_sample,
        }

        if args.num_beams > 1:
            generation_kwargs["early_stopping"] = True
            generation_kwargs["length_penalty"] = args.length_penalty

        if args.do_sample:
            generation_kwargs["top_p"] = args.top_p
            generation_kwargs["temperature"] = args.temperature

        if args.no_repeat_ngram_size > 0:
            generation_kwargs["no_repeat_ngram_size"] = (
                args.no_repeat_ngram_size
            )

        with torch.inference_mode():
            generated_ids = model.generate(
                **encoded,
                **generation_kwargs,
            )

        predictions = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        for row, prediction, token_ids in zip(batch, predictions, generated_ids):
            generated_records.append(
                {
                    "example_id": row["example_id"],
                    "pmid": row["pmid"],
                    "question": row["question"],
                    "source_text": row["source_text"],
                    "prediction": prediction.strip(),
                    "prediction_token_count": int(
                        (token_ids != tokenizer.pad_token_id).sum().item()
                    ),
                    "references": row["references"],
                    "reference_count": row["reference_count"],
                }
            )

    with output_path.open("w", encoding="utf-8") as handle:
        for record in generated_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    lengths = [record["prediction_token_count"] for record in generated_records]
    empty_predictions = sum(
        not record["prediction"].strip() for record in generated_records
    )

    summary = {
        "examples": len(generated_records),
        "empty_predictions": empty_predictions,
        "minimum_prediction_tokens": int(min(lengths)),
        "mean_prediction_tokens": float(np.mean(lengths)),
        "median_prediction_tokens": float(np.median(lengths)),
        "maximum_prediction_tokens": int(max(lengths)),
        "predictions_file": str(output_path),
    }

    summary_path = args.output_dir / "generation_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nGENERATION COMPLETED")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Configuration:", metadata_path)
    print("Predictions:", output_path)


if __name__ == "__main__":
    main()

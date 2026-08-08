#!/usr/bin/env python3

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


class TokenizedDataset(Dataset):
    def __init__(self, features: list[dict[str, Any]]) -> None:
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.features[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune FLAN-T5 on the PLABA dataset."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing the processed PLABA JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--model-name",
        default="google/flan-t5-base",
    )
    parser.add_argument(
        "--max-source-length",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--max-target-length",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-5,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def prepare_raw_examples(
    data_dir: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train_rows = load_jsonl(data_dir / "train_rows.jsonl")
    validation_rows = load_jsonl(data_dir / "val_grouped.jsonl")

    train_examples = [
        {
            "prompt": str(row["prompt"]),
            "target": str(row["target_text"]),
        }
        for row in train_rows
    ]

    # Each human reference is retained as a separate validation target.
    validation_examples = []

    for row in validation_rows:
        prompt = str(row["prompt"])

        for reference in row["references"]:
            validation_examples.append(
                {
                    "prompt": prompt,
                    "target": str(reference),
                }
            )

    return train_examples, validation_examples


def tokenize_examples(
    tokenizer,
    examples: list[dict[str, str]],
    max_source_length: int,
    max_target_length: int,
) -> TokenizedDataset:
    prompts = [row["prompt"] for row in examples]
    targets = [row["target"] for row in examples]

    model_inputs = tokenizer(
        prompts,
        max_length=max_source_length,
        truncation=True,
        padding=False,
        add_special_tokens=True,
    )

    target_tokens = tokenizer(
        text_target=targets,
        max_length=max_target_length,
        truncation=True,
        padding=False,
        add_special_tokens=True,
    )

    features = []

    for index in range(len(examples)):
        features.append(
            {
                "input_ids": model_inputs["input_ids"][index],
                "attention_mask": model_inputs["attention_mask"][index],
                "labels": target_tokens["input_ids"][index],
            }
        )

    return TokenizedDataset(features)


def count_truncated(
    tokenizer,
    texts: list[str],
    maximum_length: int,
) -> int:
    count = 0

    for text in texts:
        length = len(
            tokenizer(
                text,
                truncation=False,
                add_special_tokens=True,
                verbose=False,
            )["input_ids"]
        )

        if length > maximum_length:
            count += 1

    return count


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    print("Model:", args.model_name)
    print("GPU:", torch.cuda.get_device_name(0))
    print("BF16 supported:", torch.cuda.is_bf16_supported())
    print("Maximum source length:", args.max_source_length)
    print("Maximum target length:", args.max_target_length)

    train_examples, validation_examples = prepare_raw_examples(
        args.data_dir
    )

    print("Training examples:", len(train_examples))
    print("Validation examples:", len(validation_examples))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
    )

    # Prevent warnings while manually checking untruncated lengths.
    tokenizer.model_max_length = 1_000_000

    train_source_truncated = count_truncated(
        tokenizer,
        [row["prompt"] for row in train_examples],
        args.max_source_length,
    )
    train_target_truncated = count_truncated(
        tokenizer,
        [row["target"] for row in train_examples],
        args.max_target_length,
    )
    validation_source_truncated = count_truncated(
        tokenizer,
        [row["prompt"] for row in validation_examples],
        args.max_source_length,
    )
    validation_target_truncated = count_truncated(
        tokenizer,
        [row["target"] for row in validation_examples],
        args.max_target_length,
    )

    print("Truncated training sources:", train_source_truncated)
    print("Truncated training targets:", train_target_truncated)
    print("Truncated validation sources:", validation_source_truncated)
    print("Truncated validation targets:", validation_target_truncated)

    train_dataset = tokenize_examples(
        tokenizer,
        train_examples,
        args.max_source_length,
        args.max_target_length,
    )
    validation_dataset = tokenize_examples(
        tokenizer,
        validation_examples,
        args.max_source_length,
        args.max_target_length,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
    )

    model.config.use_cache = False

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    run_configuration = {
        "model": args.model_name,
        "max_source_length": args.max_source_length,
        "max_target_length": args.max_target_length,
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": (
            args.gradient_accumulation_steps
        ),
        "effective_train_batch_size": (
            args.train_batch_size
            * args.gradient_accumulation_steps
        ),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": 20,
        "seed": args.seed,
        "gradient_checkpointing": args.gradient_checkpointing,
        "bf16": True,
        "truncation": {
            "train_sources": train_source_truncated,
            "train_targets": train_target_truncated,
            "validation_sources": validation_source_truncated,
            "validation_targets": validation_target_truncated,
        },
    }

    with (
        args.output_dir / "run_configuration.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            run_configuration,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),

        do_train=True,
        do_eval=True,

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=5,

        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=20,
        lr_scheduler_type="linear",

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
        ),

        max_grad_norm=1.0,

        bf16=True,
        tf32=True,

        gradient_checkpointing=args.gradient_checkpointing,

        predict_with_generate=False,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        save_total_limit=1,
        save_only_model=True,

        dataloader_num_workers=0,
        dataloader_pin_memory=True,

        optim="adamw_torch",

        seed=args.seed,
        data_seed=args.seed,

        report_to="none",
        disable_tqdm=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    train_result = trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint
    )

    trainer.save_state()
    trainer.save_metrics("train", train_result.metrics)

    # load_best_model_at_end has already restored the best checkpoint.
    best_model_dir = args.output_dir / "best_model"

    trainer.model.config.use_cache = True
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(best_model_dir)

    validation_metrics = trainer.evaluate()
    trainer.save_metrics("best_validation", validation_metrics)

    summary = {
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "best_model_directory": str(best_model_dir),
        "training_metrics": train_result.metrics,
        "best_validation_metrics": validation_metrics,
        "run_configuration": run_configuration,
    }

    summary_path = args.output_dir / "training_summary.json"

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            summary,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print("\nTRAINING COMPLETED")
    print("Best checkpoint:", trainer.state.best_model_checkpoint)
    print("Best validation loss:", trainer.state.best_metric)
    print("Best model:", best_model_dir)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()

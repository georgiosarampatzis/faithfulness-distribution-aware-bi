# Faithfulness- and Distribution-Aware Text Analytics for Business Intelligence: From Evidence Retrieval to Plain-Language Adaptation

Reproducibility repository accompanying the manuscript:

**Faithfulness- and Distribution-Aware Text Analytics for Business Intelligence: From Evidence Retrieval to Plain-Language Adaptation**

## Overview

This repository contains experimental code and sanitized reproducibility artifacts for the component-wise SciFact retrieval and PLABA plain-language adaptation experiments reported in the manuscript.

## Components

- SciFact: BM25, E5, Reciprocal Rank Fusion, and cross-encoder reranking.
- PLABA: fine-tuned FLAN-T5-base, Beam-4, Greedy and Concise decoding, Random Forest candidate scoring, consistency screening, and source fallback.

The SciFact and PLABA experiments are evaluated independently; the manuscript does not claim a single cross-dataset end-to-end benchmark.

## Repository Contents

- experiments/scifact-retrieval : retrieval and evaluation scripts
- experiments/plaba-simplification : adaptation, selector, guard and evaluation scripts
- artifacts/scifact : sanitized retrieval metrics, rankings and statistical outputs
- artifacts/plaba : sanitized configurations, metrics, selector artifacts and statistical outputs
- dashboard : Streamlit integration prototype
- requirements-lock.txt : frozen Python environment snapshot
- DATA_AND_ARTIFACTS.md : data and artifact policy

## Environment

Reported experiments used Python 3.10.12, PyTorch 2.6.0, CUDA 12.4, Transformers 5.14.1, Sentence Transformers 5.6.1, Scikit-learn 1.7.2, NumPy 2.2.6, Pandas 2.3.3 and SciPy 1.15.3.

Training and inference used one NVIDIA RTX A6000 GPU with approximately 48 GB of memory.

## Data

Raw SciFact and PLABA datasets are not redistributed. They must be obtained from their original providers under the corresponding licenses and access conditions.

Per-example review artifacts have been sanitized to remove source texts, reference adaptations, consumer questions and SciFact claim text.

## Frozen PLABA Policy

The full-consistency policy requires numerical precision, negation precision, negation recall, uncertainty precision and uncertainty recall to equal 1.0. If no generated candidate satisfies all five requirements, the original source is returned.

These deterministic checks are consistency indicators and do not establish complete factual correctness.

## Model Weights

Large fine-tuned FLAN-T5 checkpoint files are not included in the reproducibility repository. Configuration metadata, metrics, selector artifacts and frozen hashes are retained.

## Anonymity

The manuscript was submitted to Information (MDPI) with author attribution.

## Reproduction

All commands below are intended to be executed from the repository root.

### SciFact Retrieval

The SciFact pipeline consists of BM25 retrieval, E5 dense retrieval, Reciprocal Rank Fusion (RRF), cross-encoder reranking, and paired bootstrap significance testing.

#### BM25

```bash
python experiments/scifact-retrieval/run_bm25.py \
  --data-dir data/scifact \
  --output-dir results/scifact/bm25 \
  --top-k 1000
```

#### E5 Dense Retrieval

```bash
python experiments/scifact-retrieval/run_e5.py \
  --data-dir data/scifact \
  --output-dir results/scifact/e5 \
  --model intfloat/e5-base-v2 \
  --batch-size 32 \
  --top-k 1000
```

Use `--rebuild-cache` when document embeddings must be recomputed.

#### Reciprocal Rank Fusion

```bash
python experiments/scifact-retrieval/run_rrf.py \
  --data-dir data/scifact \
  --bm25-rankings results/scifact/bm25/rankings.jsonl \
  --e5-rankings results/scifact/e5/rankings.jsonl \
  --output-dir results/scifact/rrf \
  --rrf-k 60 \
  --top-k 1000
```

#### Cross-Encoder Reranking

```bash
python experiments/scifact-retrieval/run_cross_encoder.py \
  --data-dir data/scifact \
  --bm25-rankings results/scifact/bm25/rankings.jsonl \
  --e5-rankings results/scifact/e5/rankings.jsonl \
  --rrf-rankings results/scifact/rrf/rankings.jsonl \
  --output-dir results/scifact/cross-encoder \
  --model cross-encoder/ms-marco-MiniLM-L6-v2 \
  --candidate-depth 100 \
  --batch-size 64 \
  --max-length 512 \
  --top-k 1000
```

#### SciFact Statistical Significance

```bash
python experiments/scifact-retrieval/bootstrap_scifact_significance.py \
  --bm25 results/scifact/bm25/per_claim_metrics.jsonl \
  --e5 results/scifact/e5/per_claim_metrics.jsonl \
  --rrf results/scifact/rrf/per_claim_metrics.jsonl \
  --cross-encoder results/scifact/cross-encoder/per_claim_metrics.jsonl \
  --output-dir results/scifact/significance \
  --bootstrap-samples 10000 \
  --seed 42
```

### PLABA Plain-Language Simplification

The PLABA pipeline consists of dataset auditing and preparation, FLAN-T5 fine-tuning, three decoding configurations, validation-based selector training, reliability analysis, frozen test-set application, and paired bootstrap significance testing.

#### Dataset Audit

```bash
python experiments/plaba-simplification/inspect_plaba.py \
  --data-dir data/plaba \
  --output-dir results/plaba/data-audit
```

#### Data Preparation

```bash
python experiments/plaba-simplification/prepare_plaba.py \
  --data-dir data/plaba \
  --output-dir data/plaba/processed \
  --tokenizer-name google/flan-t5-base
```

#### FLAN-T5 Fine-Tuning

The reported model uses `google/flan-t5-base` with maximum source and target lengths of 768 tokens.

```bash
python experiments/plaba-simplification/train_flan_t5.py \
  --data-dir data/plaba/processed \
  --output-dir results/plaba/flan-t5-base-768 \
  --model-name google/flan-t5-base \
  --max-source-length 768 \
  --max-target-length 768 \
  --train-batch-size 4 \
  --eval-batch-size 4 \
  --gradient-accumulation-steps 4 \
  --epochs 5 \
  --learning-rate 3e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.1 \
  --seed 42
```

The effective training batch size is 16.

#### Validation Candidate Generation

Beam-4:

```bash
python experiments/plaba-simplification/generate_plaba.py \
  --data-file data/plaba/processed/val_grouped.jsonl \
  --model-dir results/plaba/flan-t5-base-768/best_model \
  --output-dir results/plaba/flan-t5-base-768/validation-beam4 \
  --max-source-length 768 \
  --max-new-tokens 768 \
  --batch-size 2 \
  --num-beams 4 \
  --length-penalty 1.0 \
  --no-repeat-ngram-size 0 \
  --seed 42
```

Greedy:

```bash
python experiments/plaba-simplification/generate_plaba.py \
  --data-file data/plaba/processed/val_grouped.jsonl \
  --model-dir results/plaba/flan-t5-base-768/best_model \
  --output-dir results/plaba/flan-t5-base-768/validation-greedy \
  --max-source-length 768 \
  --max-new-tokens 768 \
  --batch-size 4 \
  --num-beams 1 \
  --length-penalty 1.0 \
  --no-repeat-ngram-size 0 \
  --seed 42
```

Concise beam:

```bash
python experiments/plaba-simplification/generate_plaba.py \
  --data-file data/plaba/processed/val_grouped.jsonl \
  --model-dir results/plaba/flan-t5-base-768/best_model \
  --output-dir results/plaba/flan-t5-base-768/validation-concise-beam \
  --max-source-length 768 \
  --max-new-tokens 512 \
  --batch-size 2 \
  --num-beams 4 \
  --length-penalty 0.0 \
  --no-repeat-ngram-size 3 \
  --seed 42
```

#### Validation Evaluation

For example:

```bash
python experiments/plaba-simplification/evaluate_plaba.py \
  --predictions results/plaba/flan-t5-base-768/validation-beam4/predictions.jsonl \
  --output-dir results/plaba/flan-t5-base-768/validation-beam4
```

The same evaluation command can be applied to the greedy and concise-beam predictions.

#### Validation-Based Selector

```bash
python experiments/plaba-simplification/build_plaba_selector.py \
  --beam4 results/plaba/flan-t5-base-768/validation-beam4/predictions.jsonl \
  --greedy results/plaba/flan-t5-base-768/validation-greedy/predictions.jsonl \
  --concise results/plaba/flan-t5-base-768/validation-concise-beam/predictions.jsonl \
  --output-dir results/plaba/flan-t5-base-768/selector-cv-v2 \
  --semantic-model intfloat/e5-base-v2 \
  --batch-size 32 \
  --folds 5 \
  --seed 42 \
  --device cuda
```

#### Reliability Policies

```bash
python experiments/plaba-simplification/evaluate_plaba_reliability_policies_v2.py \
  --features results/plaba/flan-t5-base-768/selector-cv-v2/selector_features.jsonl \
  --output-dir results/plaba/flan-t5-base-768/selector-reliability-v2
```

#### Test Candidate Generation

Beam-4:

```bash
python experiments/plaba-simplification/generate_plaba.py \
  --data-file data/plaba/processed/test_grouped.jsonl \
  --model-dir results/plaba/flan-t5-base-768/best_model \
  --output-dir results/plaba/flan-t5-base-768/test-beam4 \
  --max-source-length 768 \
  --max-new-tokens 768 \
  --batch-size 2 \
  --num-beams 4 \
  --length-penalty 1.0 \
  --no-repeat-ngram-size 0 \
  --seed 42
```

Greedy:

```bash
python experiments/plaba-simplification/generate_plaba.py \
  --data-file data/plaba/processed/test_grouped.jsonl \
  --model-dir results/plaba/flan-t5-base-768/best_model \
  --output-dir results/plaba/flan-t5-base-768/test-greedy \
  --max-source-length 768 \
  --max-new-tokens 768 \
  --batch-size 4 \
  --num-beams 1 \
  --length-penalty 1.0 \
  --no-repeat-ngram-size 0 \
  --seed 42
```

Concise beam:

```bash
python experiments/plaba-simplification/generate_plaba.py \
  --data-file data/plaba/processed/test_grouped.jsonl \
  --model-dir results/plaba/flan-t5-base-768/best_model \
  --output-dir results/plaba/flan-t5-base-768/test-concise-beam \
  --max-source-length 768 \
  --max-new-tokens 512 \
  --batch-size 2 \
  --num-beams 4 \
  --length-penalty 0.0 \
  --no-repeat-ngram-size 3 \
  --seed 42
```

#### Frozen Selector on the Test Set

```bash
python experiments/plaba-simplification/apply_plaba_frozen_selector_test.py \
  --beam4 results/plaba/flan-t5-base-768/test-beam4/predictions.jsonl \
  --greedy results/plaba/flan-t5-base-768/test-greedy/predictions.jsonl \
  --concise results/plaba/flan-t5-base-768/test-concise-beam/predictions.jsonl \
  --selector-model results/plaba/flan-t5-base-768/selector-cv-v2/selector_model.joblib \
  --selector-config results/plaba/flan-t5-base-768/selector-cv-v2/selector_configuration.json \
  --output-dir results/plaba/flan-t5-base-768/test-final \
  --semantic-model intfloat/e5-base-v2 \
  --batch-size 32 \
  --device cuda
```

The frozen selection stage does not use test references or SARI targets for candidate selection.

#### Final Test Evaluation

For example:

```bash
python experiments/plaba-simplification/evaluate_plaba.py \
  --predictions results/plaba/flan-t5-base-768/test-final/test_selector_v2_predictions.jsonl \
  --output-dir results/plaba/flan-t5-base-768/test-evaluation/selector-v2
```

#### Test-Set Statistical Significance

```bash
python experiments/plaba-simplification/bootstrap_plaba_significance.py \
  --beam4 results/plaba/flan-t5-base-768/test-beam4/predictions.jsonl \
  --selector-v2 results/plaba/flan-t5-base-768/test-final/test_selector_v2_predictions.jsonl \
  --full-guard results/plaba/flan-t5-base-768/test-final/test_full_consistency_guard_predictions.jsonl \
  --output-dir results/plaba/flan-t5-base-768/test-significance \
  --bootstrap-samples 10000 \
  --seed 42
```

### Reproducibility Artifacts

The repository includes compact experiment outputs required to inspect the reported results, including retrieval rankings, per-claim metrics, PLABA generated candidates, selector artifacts, reliability-policy outputs, final test outputs, and paired-bootstrap significance results.

Large model checkpoints, cached embedding matrices, temporary files, and execution logs are intentionally excluded.

Raw datasets are not redistributed where redistribution is restricted by their original licenses. They must be obtained from their original sources and placed under the corresponding `data/` directories before running the pipelines.

Randomized stages use seed `42`. Neural experiments require a CUDA-capable GPU.

## Held-Out External Evaluation: SimpleText2024

The accompanying manuscript additionally evaluates the
frozen PLABA adaptation pipeline on SimpleText2024,
using evaluation material distributed through BioCLEAR.

The evaluation comprises 103 source documents and
648 sentence inputs, including 578 referenced sentence
instances used for the primary sentence-level evaluation.

The FLAN-T5-base checkpoint, decoding configurations,
35-feature Random Forest selector, and consistency
policy are transferred without additional training,
parameter tuning, or threshold modification.

This experiment assesses cross-dataset robustness
rather than an official SimpleText shared-task submission.

External-transfer artifacts should be documented separately
when included in the repository.

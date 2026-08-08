# Data and Artifact Policy

## Third-Party Data

This study uses previously released research resources including SciFact and PLABA. Raw third-party datasets are not redistributed in this repository.

Users wishing to reproduce experiments from raw inputs should obtain the datasets from their original providers and comply with the corresponding licenses and access conditions.

## SciFact Artifacts

The artifacts/scifact directory contains derived retrieval metrics, sanitized rankings, per-claim evaluation metrics, and paired-bootstrap statistical outputs.

SciFact claim text and source-document text are not redistributed.

## PLABA Artifacts

The artifacts/plaba directory contains frozen generation configurations, training and validation summaries, automatic evaluation metrics, Random Forest selector artifacts, consistency-policy summaries, accessibility diagnostics, paired-bootstrap outputs, frozen policy summaries, cryptographic hashes, and sanitized per-example records.

Raw biomedical abstracts, consumer questions, human reference adaptations, and raw PLABA dataset files are not redistributed.

Generated prediction text is excluded from the sanitized per-example release where appropriate. SHA-256 identifiers are retained to support artifact identity checks.

## Large Model Files

Fine-tuned FLAN-T5 model weights and intermediate checkpoints are excluded from the anonymous review repository because of their size.

Configuration metadata, evaluation outputs, selector artifacts, and frozen hashes required to audit the reported experiments are retained.

## Frozen Artifact Integrity

SHA-256 values associated with the frozen PLABA candidate files are stored under artifacts/plaba/frozen/.

Original local absolute paths appearing in experiment records were replaced with the neutral placeholder <PROJECT_ROOT> for anonymous review.

This path sanitization does not alter reported experimental values or the recorded hashes of the original frozen candidate files.

## Double-Blind Review

Author identities, institutional information, email addresses, server names, and local user paths have been removed from the anonymous repository.

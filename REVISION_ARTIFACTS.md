# Revision Artifacts

This directory documents analyses added during peer review.

## Numerical-recall sensitivity analysis

Location:

`artifacts/revision/number-recall-sensitivity/`

The original frozen consistency policy required perfect numerical precision while treating numerical recall as diagnostic.

A reviewer-requested post-hoc sensitivity analysis additionally requires numerical recall = 1, without changing generated candidates, candidate-quality predictions, candidate ordering, or other consistency checks.

Main results:

- Original eligible candidates: 271/330
- Strict numerical-recall eligible candidates: 243/330
- Original source fallbacks: 7/110
- Strict numerical-recall source fallbacks: 11/110
- Final decisions changed: 4/110
- Original grouped SARI: 46.240
- Strict grouped SARI: 46.448

This is a post-hoc sensitivity analysis and does not replace the frozen primary policy.

## Manual diagnostic audit

Location:

`artifacts/revision/manual-audit/`

A targeted blinded audit was constructed to examine potential false rejections and false acceptances of the consistency policy.

The diagnostic sample contains:

- 7 source-fallback cases
- 8 consistency-guard versus Beam-4 disagreement cases
- 5 agreement-control cases
- 20 cases in total

The sample deliberately oversamples disagreement and fallback cases. Its proportions must not be interpreted as population-level estimates of consistency-guard accuracy.

# Adaptive Guardrail Review Worksheet

This worksheet covers the headline `official_mem0` vs `official_mem0_same_evidence_adaptive` comparison. The `initial_*` columns are provisional review labels and the `author_*` columns are intentionally blank for final confirmation.

CSV: `results/official_mem0_basecompact_full_20260502T072459Z/submission_checks/adaptive_guardrail_review.csv`

## Coverage

- All 11 automatic disagreement cases.
- Deterministic agreement sample: 10 shared-correct and 10 shared-wrong cases, seed 20260503.

## Initial Summary

| group | rows | auto top5 | auto adaptive | initial top5 | initial adaptive |
|---|---:|---:|---:|---:|---:|
| auto_adaptive_win | 5 | 0 | 5 | 1 | 5 |
| auto_top5_win | 6 | 6 | 0 | 4 | 0 |
| shared_correct_sample | 10 | 10 | 10 | 10 | 10 |
| shared_wrong_sample | 10 | 0 | 0 | 2 | 2 |
| disagreement cases only | 11 | 6 | 5 | 5 | 5 |
| total worksheet | 31 | 16 | 15 | 17 | 17 |

Do not cite the `initial_*` labels as author review until the blank `author_*` columns are filled or confirmed.

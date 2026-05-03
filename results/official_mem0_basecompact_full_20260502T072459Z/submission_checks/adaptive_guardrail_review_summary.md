# Adaptive Guardrail Audit

This author-confirmed audit covers the headline `official_mem0` vs `official_mem0_same_evidence_adaptive` comparison.

CSV: `results/official_mem0_basecompact_full_20260502T072459Z/submission_checks/adaptive_guardrail_review.csv`

## Coverage

- All 11 automatic disagreement cases between official Mem0 top-5 and the same-evidence adaptive policy.
- Deterministic agreement sample: 10 shared-correct and 10 shared-wrong cases, seed 20260503.

## Audit Summary

| group | rows | auto top5 | auto adaptive | audit top5 | audit adaptive |
|---|---:|---:|---:|---:|---:|
| auto_adaptive_win | 5 | 0 | 5 | 1 | 5 |
| auto_top5_win | 6 | 6 | 0 | 4 | 0 |
| shared_correct_sample | 10 | 10 | 10 | 10 | 10 |
| shared_wrong_sample | 10 | 0 | 0 | 2 | 2 |
| disagreement cases only | 11 | 6 | 5 | 5 | 5 |
| total worksheet | 31 | 16 | 15 | 17 | 17 |

On the 11 automatic disagreement cases, the automatic proxy scorer reports 6 top-5 wins and 5 adaptive wins; author-confirmed audit labels this subset as 5 correct for top-5 and 5 correct for adaptive. On the full 31-case worksheet, author-confirmed audit labels both policies correct on 17 cases.

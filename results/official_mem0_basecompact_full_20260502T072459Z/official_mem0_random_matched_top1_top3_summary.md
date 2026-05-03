# Matched-Budget Random Top1/Top3 Control

Deterministic control seed: 20260503
Route budget: 105 top1 cases and 206 top3 cases.
Case JSONL: `results/official_mem0_basecompact_full_20260502T072459Z/official_mem0_random_matched_top1_top3_cases.jsonl`
Random draw distribution: `results/official_mem0_basecompact_full_20260502T072459Z/official_mem0_random_matched_top1_top3_distribution.csv`

## Point Comparisons

| policy | correct | reader tokens | reader-token reduction | reader-visible context | context reduction | tokens/correct | wins/losses |
|---|---:|---:|---:|---:|---:|---:|---:|
| official_mem0 | 62/311 | 151558 | 0.00% | 34079 | 0.00% | 2444.5 | 0/0 |
| official_mem0_top1 | 50/311 | 65812 | 56.58% | 6722 | 80.28% | 1316.2 | 3/15 |
| official_mem0_random_matched_top1_top3 | 56/311 | 95230 | 37.17% | 15897 | 53.35% | 1700.5 | 3/9 |
| official_mem0_same_evidence_adaptive | 61/311 | 95466 | 37.01% | 15907 | 53.32% | 1565.0 | 5/6 |
| official_mem0_top3 | 60/311 | 110058 | 27.38% | 20452 | 39.99% | 1834.3 | 3/5 |

## Random Route Distribution

- Correct answers: mean=56.64, median=57, 95% interval=[53, 60], min=49, max=63.
- Reader total tokens: mean=95116.01, median=95113, 95% interval=[94880, 95370], min=94710, max=95538.
- Reader-visible retrieved-context tokens: mean=15815.21, median=15816, 95% interval=[15676, 15956], min=15560, max=16082.

Adaptive correct count is greater than or equal to 99.57% of random matched-budget draws.

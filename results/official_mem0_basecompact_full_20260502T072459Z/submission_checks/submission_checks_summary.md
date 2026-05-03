# Official Mem0 Submission Checks

Sources: results/official_mem0_basecompact_full_20260502T072459Z

## Policy Efficiency

| policy | rows | correct | reader tokens | delta tokens | tokens/correct | correct/100k |
|---|---:|---:|---:|---:|---:|---:|
| official_mem0 | 311 | 62 | 151558 | --- | 2444.48 | 40.91 |
| official_mem0_odv2_selective | 311 | 58 | 86838 | -42.70 | 1497.21 | 66.79 |
| official_mem0_top1 | 311 | 50 | 65812 | -56.58 | 1316.24 | 75.97 |
| official_mem0_top2 | 311 | 54 | 86002 | -43.25 | 1592.63 | 62.79 |
| official_mem0_top3 | 311 | 60 | 110058 | -27.38 | 1834.30 | 54.52 |
| official_mem0_top4 | 311 | 59 | 130157 | -14.12 | 2206.05 | 45.33 |

## Bootstrap CIs

| metric | 2.5% | median | 97.5% |
|---|---:|---:|---:|
| accuracy_delta_pct_points | -4.1801 | -1.2862 | 1.6077 |
| base_tokens_per_correct | 1991.4211 | 2446.5000 | 3149.4375 |
| reader_token_reduction_pct | 41.9122 | 42.7229 | 43.4016 |
| target_tokens_per_correct | 1206.9861 | 1498.9138 | 1940.3111 |
| tokens_per_correct_reduction_pct | 28.6213 | 38.7507 | 47.1313 |

## Manual Audit

Wrote 50 prioritized cases to `manual_audit_sample.csv`.
Use this to test whether the local span scorer is undercounting correctness.

## State Guard Isolation

Wrote 2 rows where ODV2 removed stale official Mem0 evidence.
If this count is tiny, claim token savings as ranked evidence compaction rather than validity reasoning.

## What This Supports

- Token-savings claim: supported by paired reader-token totals.
- Cost-normalized utility claim: supported by tokens per correct answer.
- Accuracy-improvement claim: not supported unless additional validation changes the result.
- Systems claim: still needs cache-free wall-clock benchmark output.

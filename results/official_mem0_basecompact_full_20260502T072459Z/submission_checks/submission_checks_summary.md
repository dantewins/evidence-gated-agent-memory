# Official Mem0 Submission Checks

Sources: results/official_mem0_basecompact_full_20260502T072459Z

## Policy Efficiency

| policy | rows | correct | reader tokens | delta tokens | tokens/correct | correct/100k |
|---|---:|---:|---:|---:|---:|---:|
| official_mem0 | 311 | 62 | 151558 | --- | 2444.48 | 40.91 |
| official_mem0_odv2_selective | 311 | 58 | 86838 | -42.70 | 1497.21 | 66.79 |
| official_mem0_same_evidence_adaptive | 311 | 61 | 95466 | -37.01 | 1565.02 | 63.90 |
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
| accuracy_delta_pct_points | -2.5723 | -0.3215 | 1.6077 |
| base_tokens_per_correct | 1991.4211 | 2446.5000 | 3149.4375 |
| reader_token_reduction_pct | 35.4819 | 36.9965 | 38.5655 |
| target_tokens_per_correct | 1274.8289 | 1567.8525 | 2008.3830 |
| tokens_per_correct_reduction_pct | 28.5183 | 36.0509 | 42.6360 |
| accuracy_delta_pct_points | -6.7524 | -3.8585 | -1.2862 |
| base_tokens_per_correct | 1991.4211 | 2446.5000 | 3149.4375 |
| reader_token_reduction_pct | 56.2804 | 56.5777 | 56.8561 |
| target_tokens_per_correct | 1043.2063 | 1317.6200 | 1771.5946 |
| tokens_per_correct_reduction_pct | 36.5017 | 46.1878 | 53.1889 |
| accuracy_delta_pct_points | -5.1447 | -2.5723 | -0.3215 |
| base_tokens_per_correct | 1991.4211 | 2446.5000 | 3149.4375 |
| reader_token_reduction_pct | 42.9529 | 43.2548 | 43.5538 |
| target_tokens_per_correct | 1274.5373 | 1594.6852 | 2097.4878 |
| tokens_per_correct_reduction_pct | 25.2078 | 34.8818 | 42.2621 |
| accuracy_delta_pct_points | -2.5723 | -0.6431 | 0.9646 |
| base_tokens_per_correct | 1991.4211 | 2446.5000 | 3149.4375 |
| reader_token_reduction_pct | 27.1029 | 27.3799 | 27.6568 |
| target_tokens_per_correct | 1485.5676 | 1835.6333 | 2352.4468 |
| tokens_per_correct_reduction_pct | 17.4299 | 24.9831 | 31.4503 |
| accuracy_delta_pct_points | -2.5723 | -0.9646 | 0.6431 |
| base_tokens_per_correct | 1991.4211 | 2446.5000 | 3149.4375 |
| reader_token_reduction_pct | 13.8536 | 14.1208 | 14.3850 |
| target_tokens_per_correct | 1783.4247 | 2207.0847 | 2845.9783 |
| tokens_per_correct_reduction_pct | 1.0313 | 9.8638 | 16.9536 |

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

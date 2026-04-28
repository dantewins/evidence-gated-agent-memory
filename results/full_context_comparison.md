# Full Benchmark Context

## Summary

ODV2 selective retrieval does **not** beat Mem0 globally. The usable result is narrow:
on LongMemEval `knowledge-update`, ODV2 preserves Mem0 correctness on the cases
where validity gating applies while reducing prompt context.

## Positive Result: LongMemEval Knowledge-Update

| Slice | n | Delta Accuracy vs Mem0 | Delta Stale Exposure vs Mem0 | Delta Prompt Tokens vs Mem0 | Wins | Losses | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ODV2 intervened | 6 | 0.000 | 0.000 | -256.33 | 0 | 0 | usable |
| Mem0 exposed stale state | 4 | 0.000 | 0.000 | -310.25 | 0 | 0 | usable |

## LongMemEval Temporal-Reasoning

| Method | n | Accuracy | Exact Accuracy | Retrieval Hit | Stale Exposure | Context Tokens | Memory Tokens | Delta Accuracy vs Mem0 | Delta Context Tokens vs Mem0 | Wins | Losses | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| mem0 | 133 | 0.226 | 0.113 | 0.008 | 0.000 | 1438.10 | 670.85 | 0.000 | 0.00 | 0 | 0 | baseline |
| odv2_mem0_selective | 133 | 0.135 | 0.053 | 0.045 | 0.000 | 1062.54 | 216.17 | -0.090 | -375.56 | 7 | 19 | not usable |
| odv2_dense_compact | 133 | 0.150 | 0.075 | 0.045 | 0.000 | 1067.06 | 291.89 | -0.075 | -371.04 | 7 | 17 | not usable |
| odv2_mem0_temporal_prune | 133 | 0.135 | 0.053 | 0.045 | 0.000 | 1062.54 | 216.17 | -0.090 | -375.56 | 7 | 19 | not usable |
| odv2_recovery | 133 | 0.135 | 0.053 | 0.045 | 0.000 | 1062.54 | 216.17 | -0.090 | -375.56 | 7 | 19 | not usable |

## LoCoMo Aggregate

| Method | Accuracy | Exact Accuracy | Retrieval Hit | Stale Exposure | Context Tokens | Memory Tokens | Latency ms | Delta Accuracy vs Mem0 | Delta Context Tokens vs Mem0 | Wins | Losses | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| mem0 | 0.243 | 0.121 | 0.008 | 0.137 | 661.32 | 145.79 | 82.12 | 0.000 | 0.00 | 0 | 0 | baseline |
| odv2_mem0_selective | 0.240 | 0.113 | 0.013 | 0.137 | 641.13 | 130.56 | 87.34 | -0.003 | -20.19 | 7 | 11 | not usable |
| strong_retrieval | 0.080 | 0.031 | 0.089 | 0.765 | 886.73 | 121.65 | 99.76 |  |  |  |  | baseline ablation |
| dense_retrieval | 0.213 | 0.103 | 0.077 | 0.616 | 802.94 | 141.46 | 95.07 |  |  |  |  | baseline ablation |
| mem0_validity_guard | 0.198 | 0.099 | 0.023 | 0.749 | 1431.05 | 256.74 | 141.09 |  |  |  |  | negative ablation |
| odv2_mem0_hybrid | 0.178 | 0.088 | 0.023 | 0.764 | 808.15 | 143.86 | 100.90 |  |  |  |  | negative ablation |
| odv2_mem0_temporal_prune | 0.228 | 0.115 | 0.013 | 0.137 | 698.48 | 136.79 | 87.05 | -0.015 | 37.16 |  |  | not usable |
| odv2_recovery | 0.193 | 0.097 | 0.015 | 0.422 | 690.32 | 125.53 | 92.38 |  |  |  |  | negative ablation |
| odv2_dense | 0.111 | 0.047 | 0.121 | 0.765 | 1342.69 | 182.01 | 132.32 |  |  |  |  | negative ablation |
| odv2_dense_compact | 0.069 | 0.027 | 0.068 | 0.765 | 577.90 | 68.28 | 73.50 |  |  |  |  | negative ablation |

## LoCoMo Validity Slices

| Slice | n | Delta Accuracy vs Mem0 | Delta Stale Exposure vs Mem0 | Delta Prompt Tokens vs Mem0 | Wins | Losses | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ODV2 intervened | 217 | -0.018 | 0.000 | -143.50 | 7 | 11 | not usable |
| Mem0 exposed stale state | 212 | -0.019 | 0.000 | -143.86 | 7 | 11 | not usable |
| Mem0 same-key conflict | 0 | 0.000 | 0.000 | 0.00 | 0 | 0 | no cases |

## Claim

ODV2 is not a replacement for Mem0. The defensible result is that ODV2 can act
as a conservative validity gate on update-sensitive LongMemEval knowledge-update
cases, preserving Mem0 correctness with zero paired losses while reducing prompt
context.

## Limitations

- ODV2 does not improve broad benchmark accuracy over Mem0.
- Temporal-reasoning is not a positive result; reducing historical context hurts accuracy.
- LoCoMo is not a positive result; ODV2 reduces some context but introduces paired losses.

# Cache-Free Reader Systems Benchmark

Shared cases replayed: 64

| policy | cases | correct | wall ms | examples/s | tokens/s | mean ms/case | peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| official_mem0 | 64 | 9 | 4005.50 | 15.9780 | 7962.05 | 62.59 | 16179.3 |
| official_mem0_same_evidence_adaptive | 64 | 10 | 2378.25 | 26.9106 | 8307.39 | 37.16 | 15686.1 |
| official_mem0_top1 | 64 | 10 | 1677.78 | 38.1455 | 7805.53 | 26.22 | 15219.0 |
| official_mem0_top2 | 64 | 9 | 2138.86 | 29.9225 | 8386.26 | 33.42 | 15440.1 |
| official_mem0_top3 | 64 | 9 | 2532.75 | 25.2690 | 8932.97 | 39.57 | 15686.1 |

Cache is disabled in `LocalHFReasoner` for this benchmark. The rows replay existing retrieved records and do not rerun Mem0 ingestion or extraction.

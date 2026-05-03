# Cache-Free Reader Systems Benchmark

Shared cases replayed: 64

| policy | cases | correct | wall ms | examples/s | tokens/s | mean ms/case | peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| official_mem0 | 64 | 9 | 3614.20 | 17.7079 | 8824.09 | 56.47 | 16179.3 |
| official_mem0_odv2_selective | 64 | 9 | 2374.55 | 26.9525 | 7638.92 | 37.10 | 15440.1 |
| official_mem0_top2 | 64 | 9 | 2167.52 | 29.5269 | 8275.37 | 33.87 | 15440.1 |
| official_mem0_top3 | 64 | 9 | 2552.07 | 25.0777 | 8865.36 | 39.88 | 15686.1 |

Cache is disabled in `LocalHFReasoner` for this benchmark. The rows replay existing retrieved records and do not rerun Mem0 ingestion or extraction.

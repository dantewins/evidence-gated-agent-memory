from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: python scripts/summarize_manifests.py <manifest.json> [<manifest.json> ...]")
        return 1

    rows: list[dict[str, object]] = []
    for raw_path in argv:
        path = Path(raw_path)
        payload = json.loads(path.read_text())
        benchmark = str(payload.get("benchmark", path.stem))
        for metric in payload.get("metrics", []):
            row = dict(metric)
            row["benchmark"] = benchmark
            rows.append(row)

    if not rows:
        print("No metric rows found.")
        return 1

    policies = sorted({str(row["policy_name"]) for row in rows})
    benchmarks = sorted({str(row["benchmark"]) for row in rows})

    print("| policy | benchmark | accuracy | exact_accuracy | retrieval_hit | stale_exposure | ctx_tokens | memory_tokens | latency_ms | snapshot |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for policy in policies:
        for benchmark in benchmarks:
            match = next(
                (
                    row
                    for row in rows
                    if str(row["policy_name"]) == policy and str(row["benchmark"]) == benchmark
                ),
                None,
            )
            if match is None:
                continue
            print(
                f"| {policy} | {benchmark} | "
                f"{float(match['accuracy']):.3f} | "
                f"{float(match.get('exact_match_accuracy', match['accuracy'])):.3f} | "
                f"{float(match.get('retrieval_hit_rate', 0.0)):.3f} | "
                f"{float(match.get('stale_state_exposure_rate', 0.0)):.3f} | "
                f"{float(match['avg_context_tokens']):.2f} | "
                f"{float(match.get('avg_retrieved_context_tokens', match['avg_context_tokens'])):.2f} | "
                f"{float(match['avg_query_latency_ms']):.2f} | "
                f"{float(match['avg_snapshot_size']):.2f} |"
            )

    print()
    print("| policy | mean_accuracy | mean_retrieval_hit | mean_stale_exposure | mean_ctx_tokens | mean_latency_ms |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for policy in policies:
        policy_rows = [row for row in rows if str(row["policy_name"]) == policy]
        mean_accuracy = sum(float(row["accuracy"]) for row in policy_rows) / len(policy_rows)
        mean_retrieval_hit = sum(float(row.get("retrieval_hit_rate", 0.0)) for row in policy_rows) / len(policy_rows)
        mean_stale_exposure = sum(float(row.get("stale_state_exposure_rate", 0.0)) for row in policy_rows) / len(policy_rows)
        mean_context = sum(float(row["avg_context_tokens"]) for row in policy_rows) / len(policy_rows)
        mean_latency = sum(float(row["avg_query_latency_ms"]) for row in policy_rows) / len(policy_rows)
        print(
            f"| {policy} | {mean_accuracy:.3f} | {mean_retrieval_hit:.3f} | "
            f"{mean_stale_exposure:.3f} | {mean_context:.2f} | {mean_latency:.2f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

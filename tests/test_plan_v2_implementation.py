import json

from memory_inference.agent import AgentRunner
from memory_inference.benchmarks.longmemeval_preprocess import load_preprocessed_longmemeval, preprocess_longmemeval
from memory_inference.benchmarks.locomo_preprocess import load_preprocessed_locomo, preprocess_locomo
from memory_inference.benchmarks.revision_synthetic import RevisionBenchmarkConfig, build_revision_benchmark
from memory_inference.consolidation.offline_delta_v2 import OfflineDeltaConsolidationPolicyV2
from memory_inference.consolidation.recency_salience import RecencySalienceMemoryPolicy
from memory_inference.consolidation.revision_types import QueryMode
from memory_inference.llm.deterministic_reader import DeterministicValidityReader
from memory_inference.llm.fixed_prompt_reader import FixedPromptReader
from memory_inference.llm.mock_consolidator import MockConsolidator


def test_recency_salience_prioritizes_recent_exact_match() -> None:
    scenario = next(
        scenario
        for scenario in build_revision_benchmark(RevisionBenchmarkConfig(entities=2))
        if scenario.scenario_id.startswith("S1")
    )
    query = scenario.batch.queries[0]
    policy = RecencySalienceMemoryPolicy()
    policy.ingest(scenario.batch.updates)
    top = policy.retrieve(query.entity, query.attribute, top_k=1).entries[0]
    assert top.entity == query.entity
    assert top.attribute == query.attribute


def test_deterministic_reader_handles_history_queries() -> None:
    scenario = next(
        scenario
        for scenario in build_revision_benchmark(RevisionBenchmarkConfig(entities=1))
        if scenario.scenario_id.startswith("S3")
    )
    query = scenario.batch.queries[0]
    query.query_mode = QueryMode.HISTORY
    query.answer = next(
        entry.value for entry in scenario.batch.updates
        if entry.entity == query.entity and entry.attribute == query.attribute
    )
    runner = AgentRunner(
        policy=RecencySalienceMemoryPolicy(),
        reasoner=DeterministicValidityReader(),
    )
    examples = runner.run_batches([scenario.batch])
    assert examples[0].prediction == query.answer


def test_fixed_prompt_reader_is_stable_stub() -> None:
    reader = FixedPromptReader(prompt_template="Use only memory.")
    assert reader.prompt_template == "Use only memory."


def test_offline_delta_respects_maintenance_frequency() -> None:
    scenario = next(
        scenario
        for scenario in build_revision_benchmark(RevisionBenchmarkConfig(entities=1))
        if scenario.scenario_id.startswith("S2")
    )
    policy = OfflineDeltaConsolidationPolicyV2(
        consolidator=MockConsolidator(),
        maintenance_frequency=2,
    )
    policy.ingest(scenario.batch.updates)
    policy.maybe_consolidate()
    assert policy.retrieve("user_00", scenario.batch.queries[0].attribute).entries == []
    policy.maybe_consolidate()
    assert policy.retrieve("user_00", scenario.batch.queries[0].attribute).entries


def test_revision_benchmark_includes_long_gap_and_alias_scenarios() -> None:
    scenarios = build_revision_benchmark(RevisionBenchmarkConfig(entities=1))
    scenario_ids = {scenario.scenario_id.split("_")[0] for scenario in scenarios}
    assert "S6" in scenario_ids
    assert "S7" in scenario_ids


def test_longmemeval_preprocess_round_trips_normalized_schema(tmp_path) -> None:
    path = tmp_path / "longmemeval.json"
    output = tmp_path / "longmemeval.normalized.json"
    payload = [
        {
            "question_id": "q_001",
            "question_type": "knowledge-update",
            "question": "What city does the user live in now?",
            "answer": "Boston",
            "haystack_sessions": [
                {"role": "user", "content": "I moved to Boston."},
            ],
            "haystack_session_ids": ["sess_1"],
            "haystack_dates": ["2024-01-20"],
        }
    ]
    path.write_text(json.dumps(payload))
    batches = preprocess_longmemeval(path, output)
    reloaded = load_preprocessed_longmemeval(output)
    assert len(batches) == 1
    assert reloaded[0].queries[0].attribute == "home_city"
    assert reloaded[0].queries[0].query_mode == QueryMode.CURRENT_STATE


def test_locomo_preprocess_round_trips_normalized_schema(tmp_path) -> None:
    path = tmp_path / "locomo.json"
    output = tmp_path / "locomo.normalized.json"
    payload = [
        {
            "sample_id": "sample_001",
            "conversation": {
                "session_1": [
                    {"dia_id": 0, "speaker": "Alice", "text": "I got a new job at Google."},
                ],
                "session_1_date_time": "2024-01-10",
            },
            "event_summary": {
                "Alice": ["Got a job at Google"],
            },
            "qa": [
                {
                    "question": "Where does Alice work now?",
                    "answer": "Google",
                    "category": 1,
                }
            ],
        }
    ]
    path.write_text(json.dumps(payload))
    batches = preprocess_locomo(path, output)
    reloaded = load_preprocessed_locomo(output)
    assert len(batches) == 1
    assert reloaded[0].queries[0].attribute == "employer"
    assert reloaded[0].queries[0].answer == "Google"

from memory_inference.domain.enums import QueryMode
from memory_inference.memory.policies.presets import (
    mem0_all_features_policy,
    mem0_archive_conflict_policy,
    mem0_history_aware_policy,
    mem0_policy,
)
from tests.factories import make_query, make_record


class FakeDenseEncoder:
    def encode_query(self, text: str) -> tuple[float, ...]:
        return self._encode(text)

    def encode_passage(self, text: str) -> tuple[float, ...]:
        return self._encode(text)

    def encode_passages(self, texts) -> list[tuple[float, ...]]:
        return [self._encode(text) for text in texts]

    def similarity(self, left, right) -> float:
        return sum(left_value * right_value for left_value, right_value in zip(left, right))

    def _encode(self, text: str) -> tuple[float, ...]:
        lower = text.lower()
        return (
            1.0 if "google" in lower else 0.0,
            1.0 if "meta" in lower else 0.0,
            1.0 if "employer" in lower or "work" in lower or "job" in lower else 0.0,
            1.0 if "boston" in lower else 0.0,
            1.0 if "seattle" in lower else 0.0,
        )


def test_mem0_preset_flags_match_expected_behavior_surface() -> None:
    base = mem0_policy(encoder=FakeDenseEncoder())
    history = mem0_history_aware_policy(encoder=FakeDenseEncoder())
    archive = mem0_archive_conflict_policy(encoder=FakeDenseEncoder())
    full = mem0_all_features_policy(encoder=FakeDenseEncoder())

    assert base.name == "mem0"
    assert base.history_enabled is False
    assert base.archive_conflict_enabled is False

    assert history.name == "mem0_history_aware"
    assert history.history_enabled is True
    assert history.archive_conflict_enabled is False

    assert archive.name == "mem0_archive_conflict"
    assert archive.history_enabled is False
    assert archive.archive_conflict_enabled is True

    assert full.name == "mem0_all_features"
    assert full.history_enabled is True
    assert full.archive_conflict_enabled is True


def test_mem0_full_preset_uses_augmented_state_path() -> None:
    policy = mem0_all_features_policy(encoder=FakeDenseEncoder())
    policy.ingest(
        [
            make_record(
                entry_id="old",
                entity="user",
                attribute="employer",
                value="Google",
                timestamp=1,
                session_id="s",
                metadata={"memory_kind": "state"},
            ),
            make_record(
                entry_id="new",
                entity="user",
                attribute="employer",
                value="Meta",
                timestamp=2,
                session_id="s",
                metadata={"memory_kind": "state"},
            ),
        ]
    )

    result = policy.retrieve_for_query(
        make_query(
            query_id="q",
            entity="user",
            attribute="employer",
            question="Where does the user work now?",
            answer="Meta",
            timestamp=3,
            session_id="s",
            query_mode=QueryMode.CURRENT_STATE,
        ),
        top_k=2,
    )

    assert result.debug["retrieval_mode"] == "mem0_state_augmented"
    assert {entry.value for entry in result.entries} == {"Google", "Meta"}

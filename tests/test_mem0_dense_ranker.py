from memory_inference.domain.enums import QueryMode
from memory_inference.memory.retrieval import DenseRanker
from tests.factories import make_query, make_record


class FakeDenseEncoder:
    def encode_query(self, text: str) -> tuple[float, ...]:
        return self._encode(text)

    def encode_passage(self, text: str) -> tuple[float, ...]:
        return self._encode(text)

    def encode_passages(self, texts) -> list[tuple[float, ...]]:
        return [self._encode(text) for text in texts]

    def similarity(self, left, right) -> float:
        return sum(a * b for a, b in zip(left, right))

    def _encode(self, text: str) -> tuple[float, ...]:
        lower = text.lower()
        return (
            1.0 if "google" in lower else 0.0,
            1.0 if "meta" in lower else 0.0,
            1.0 if "employer" in lower or "work" in lower or "job" in lower else 0.0,
            1.0 if "boston" in lower else 0.0,
        )


def test_dense_ranker_prefers_best_semantic_query_match() -> None:
    ranker = DenseRanker(encoder=FakeDenseEncoder())
    candidates = [
        make_record(
            entry_id="google",
            entity="user",
            attribute="employer",
            value="Google",
            timestamp=1,
            session_id="s",
        ),
        make_record(
            entry_id="boston",
            entity="user",
            attribute="home_city",
            value="Boston",
            timestamp=2,
            session_id="s",
        ),
    ]
    for entry in candidates:
        ranker.index(entry)

    query = make_query(
        query_id="q",
        entity="user",
        attribute="employer",
        question="Where does the user work at Google?",
        answer="Google",
        timestamp=3,
        session_id="s",
    )
    ranked = ranker.rank_query(query, candidates, entity_matches=lambda left, right: left == right)

    assert ranked[0].entry_id == "google"


def test_dense_ranker_history_mode_prefers_older_match_when_similarity_ties() -> None:
    ranker = DenseRanker(encoder=FakeDenseEncoder())
    candidates = [
        make_record(
            entry_id="old",
            entity="user",
            attribute="employer",
            value="Google",
            timestamp=1,
            session_id="s",
        ),
        make_record(
            entry_id="new",
            entity="user",
            attribute="employer",
            value="Google",
            timestamp=5,
            session_id="s",
        ),
    ]
    for entry in candidates:
        ranker.index(entry)

    query = make_query(
        query_id="q",
        entity="user",
        attribute="employer",
        question="What was the user's previous employer?",
        answer="Google",
        timestamp=6,
        session_id="s",
        query_mode=QueryMode.HISTORY,
    )
    ranked = ranker.rank_query(
        query,
        candidates,
        entity_matches=lambda left, right: left == right,
        history=True,
    )

    assert ranked[0].entry_id == "old"


def test_dense_ranker_rebuilds_entry_vector_after_removal() -> None:
    ranker = DenseRanker(encoder=FakeDenseEncoder())
    entry = make_record(
        entry_id="google",
        entity="user",
        attribute="employer",
        value="Google",
        timestamp=1,
        session_id="s",
    )

    ranker.index(entry)
    cached_vector = ranker.entry_vector(entry)
    ranker.remove(entry.entry_id)
    rebuilt_vector = ranker.entry_vector(entry)

    assert entry.entry_id in ranker.entry_vectors
    assert rebuilt_vector == cached_vector


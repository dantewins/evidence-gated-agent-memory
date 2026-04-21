from dataclasses import fields

from memory_inference.domain.query import RuntimeQuery


def test_runtime_query_has_no_gold_answer_field() -> None:
    field_names = {field.name for field in fields(RuntimeQuery)}
    assert "answer" not in field_names
    assert "gold_answer" not in field_names

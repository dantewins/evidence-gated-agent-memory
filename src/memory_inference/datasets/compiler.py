from __future__ import annotations

import logging
from typing import Iterable

from memory_inference.annotation.fact_extractor import extract_structured_facts
from memory_inference.annotation.provenance import compact_support_text
from memory_inference.annotation.query_intent import (
    choose_query_attribute,
    infer_locomo_query_entity,
    infer_longmemeval_query_entity,
    locomo_query_mode,
    longmemeval_query_mode,
    should_skip_locomo_question,
)
from memory_inference.annotation.salience import estimate_confidence, estimate_importance
from memory_inference.datasets.normalized_io import (
    ANNOTATION_VERSION,
    BENCHMARK_SOURCE_VERSION,
    COMPILER_VERSION,
    SCHEMA_VERSION,
    NormalizedDataset,
    NormalizedRecord,
)
from memory_inference.domain.benchmark import (
    ExperimentCase,
    ExperimentContext,
    RawLoCoMoSample,
    RawLongMemEvalRecord,
)
from memory_inference.domain.memory import MemoryRecord
from memory_inference.domain.query import RuntimeQuery
from memory_inference.evaluation.targets import EvalTarget

logger = logging.getLogger(__name__)


def compile_locomo_samples(
    samples: Iterable[RawLoCoMoSample],
    *,
    split: str = "default",
) -> NormalizedDataset:
    records: list[NormalizedRecord] = []
    dropped = 0
    warnings: list[str] = []
    total_updates = 0
    total_cases = 0

    for idx, sample in enumerate(samples):
        try:
            context = _compile_locomo_context(sample)
            cases = _compile_locomo_cases(sample, context)
            if not cases:
                continue
            total_updates += len(context.updates)
            total_cases += len(cases)
            records.append(
                NormalizedRecord(
                    schema_version=SCHEMA_VERSION,
                    source_dataset="locomo",
                    source_split=split,
                    source_record_id=sample.sample_id,
                    context=context,
                    cases=cases,
                    preprocessing_metadata={
                        "sample_id": sample.sample_id,
                        "num_sessions": str(len(sample.sessions)),
                    },
                )
            )
        except (KeyError, ValueError, TypeError) as exc:
            dropped += 1
            warnings.append(f"Sample {idx}: {exc}")
            logger.warning("Dropped LoCoMo sample %d: %s", idx, exc)

    return NormalizedDataset(
        source_dataset="locomo",
        source_split=split,
        records=records,
        total_contexts=len(records),
        total_updates=total_updates,
        total_cases=total_cases,
        dropped_records=dropped,
        warnings=warnings,
        benchmark_source_version=BENCHMARK_SOURCE_VERSION,
        annotation_version=ANNOTATION_VERSION,
        compiler_version=COMPILER_VERSION,
    )


def compile_longmemeval_records(
    records: Iterable[RawLongMemEvalRecord],
    *,
    split: str = "default",
) -> NormalizedDataset:
    normalized_records: list[NormalizedRecord] = []
    dropped = 0
    warnings: list[str] = []
    total_updates = 0
    total_cases = 0

    for idx, record in enumerate(records):
        try:
            context = _compile_longmemeval_context(record)
            cases = _compile_longmemeval_cases(record, context)
            total_updates += len(context.updates)
            total_cases += len(cases)
            normalized_records.append(
                NormalizedRecord(
                    schema_version=SCHEMA_VERSION,
                    source_dataset="longmemeval",
                    source_split=split,
                    source_record_id=record.question_id,
                    context=context,
                    cases=cases,
                    preprocessing_metadata={
                        "question_type": record.question_type,
                        "num_haystack_sessions": str(len(record.sessions)),
                    },
                )
            )
        except (KeyError, ValueError, TypeError) as exc:
            dropped += 1
            warnings.append(f"Record {idx}: {exc}")
            logger.warning("Dropped LongMemEval record %d: %s", idx, exc)

    return NormalizedDataset(
        source_dataset="longmemeval",
        source_split=split,
        records=normalized_records,
        total_contexts=len(normalized_records),
        total_updates=total_updates,
        total_cases=total_cases,
        dropped_records=dropped,
        warnings=warnings,
        benchmark_source_version=BENCHMARK_SOURCE_VERSION,
        annotation_version=ANNOTATION_VERSION,
        compiler_version=COMPILER_VERSION,
    )


def _compile_locomo_context(sample: RawLoCoMoSample) -> ExperimentContext:
    updates: list[MemoryRecord] = []
    ts_counter = 0

    for speaker, events in sample.event_summary.items():
        for event_text in events:
            event_confidence = estimate_confidence(event_text, speaker=speaker, attribute="event")
            event_importance = estimate_importance(event_text, speaker=speaker, attribute="event")
            source_record_id = f"{sample.sample_id}-evt-{ts_counter}"
            updates.append(
                MemoryRecord(
                    record_id=source_record_id,
                    entity=speaker,
                    attribute="event",
                    value=event_text,
                    timestamp=ts_counter,
                    session_id=sample.sample_id,
                    confidence=event_confidence,
                    importance=event_importance,
                    provenance="locomo_event_summary",
                    memory_kind="event",
                    speaker=speaker,
                )
            )
            for fact_idx, fact in enumerate(extract_structured_facts(event_text)):
                updates.append(
                    MemoryRecord(
                        record_id=f"{source_record_id}-fact-{fact_idx}",
                        entity=speaker,
                        attribute=fact.attribute,
                        value=fact.value,
                        timestamp=ts_counter,
                        session_id=sample.sample_id,
                        confidence=min(1.0, event_confidence + 0.08),
                        importance=min(1.8, event_importance + 0.15),
                        scope="default" if fact.is_stateful else f"event_{ts_counter}:fact_{fact_idx}",
                        provenance="locomo_event_summary_fact",
                        source_kind="structured_fact",
                        source_attribute="event",
                        memory_kind="state" if fact.is_stateful else "event",
                        source_entry_id=source_record_id,
                        support_text=compact_support_text(event_text),
                        speaker=speaker,
                    )
                )
            ts_counter += 1

    for session in sample.sessions:
        for turn in session.turns:
            if not turn.text.strip():
                continue
            confidence = estimate_confidence(turn.text, speaker=turn.speaker, attribute="dialogue")
            importance = estimate_importance(turn.text, speaker=turn.speaker, attribute="dialogue")
            source_record_id = f"{sample.sample_id}-{session.label}-{turn.turn_id or ts_counter}"
            updates.append(
                MemoryRecord(
                    record_id=source_record_id,
                    entity=turn.speaker,
                    attribute="dialogue",
                    value=turn.text,
                    timestamp=ts_counter,
                    session_id=f"{sample.sample_id}-{session.label}",
                    confidence=confidence,
                    importance=importance,
                    scope=session.label,
                    provenance="locomo_dialogue",
                    memory_kind="event",
                    speaker=turn.speaker,
                    source_date=session.date,
                    session_label=session.label,
                )
            )
            for fact_idx, fact in enumerate(extract_structured_facts(turn.text)):
                updates.append(
                    MemoryRecord(
                        record_id=f"{source_record_id}-fact-{fact_idx}",
                        entity=turn.speaker,
                        attribute=fact.attribute,
                        value=fact.value,
                        timestamp=ts_counter,
                        session_id=f"{sample.sample_id}-{session.label}",
                        confidence=min(1.0, confidence + 0.08),
                        importance=min(1.8, importance + 0.15),
                        scope=session.label if fact.is_stateful else f"{session.label}:turn_{ts_counter}:fact_{fact_idx}",
                        provenance="locomo_dialogue_fact",
                        source_kind="structured_fact",
                        source_attribute="dialogue",
                        memory_kind="state" if fact.is_stateful else "event",
                        source_entry_id=source_record_id,
                        support_text=compact_support_text(turn.text),
                        speaker=turn.speaker,
                        source_date=session.date,
                        session_label=session.label,
                    )
                )
            ts_counter += 1

    return ExperimentContext(
        context_id=sample.sample_id,
        session_id=sample.sample_id,
        updates=updates,
        metadata={
            "sample_id": sample.sample_id,
            "num_sessions": str(len(sample.sessions)),
        },
    )


def _compile_locomo_cases(sample: RawLoCoMoSample, context: ExperimentContext) -> list[ExperimentCase]:
    speakers = set(sample.event_summary.keys()) | {
        turn.speaker
        for session in sample.sessions
        for turn in session.turns
    }
    query_timestamp = max((update.timestamp for update in context.updates), default=0) + 1
    cases: list[ExperimentCase] = []
    for qa_idx, question in enumerate(sample.questions):
        if should_skip_locomo_question(question.category, question.answer):
            continue
        query_entity = infer_locomo_query_entity(question.question, speakers)
        query_attribute = choose_query_attribute(
            question.question,
            query_entity,
            context.updates,
            default_attribute="dialogue",
        )
        case_id = f"{sample.sample_id}-q{qa_idx}"
        runtime_query = RuntimeQuery(
            query_id=case_id,
            context_id=context.context_id,
            entity=query_entity,
            attribute=query_attribute,
            question=question.question,
            timestamp=query_timestamp,
            session_id=context.session_id,
            query_mode=locomo_query_mode(question.category),
            supports_abstention=str(question.category).strip().lower() in {"adversarial", "5"},
        )
        cases.append(
            ExperimentCase(
                case_id=case_id,
                context_id=context.context_id,
                runtime_query=runtime_query,
                eval_target=EvalTarget(
                    query_id=case_id,
                    gold_answer=question.answer,
                    benchmark_name="locomo",
                    benchmark_category=str(question.category),
                    supports_abstention=runtime_query.supports_abstention,
                ),
                metadata={
                    "sample_id": sample.sample_id,
                    "question_category": str(question.category),
                },
            )
        )
    return cases


def _compile_longmemeval_context(record: RawLongMemEvalRecord) -> ExperimentContext:
    updates: list[MemoryRecord] = []
    turn_counter = 0
    for session_idx, session in enumerate(record.sessions):
        scope = session.label or f"session_{session_idx}"
        for turn in session.turns:
            if not turn.text.strip():
                continue
            confidence = estimate_confidence(turn.text, speaker=turn.speaker, attribute="dialogue")
            importance = estimate_importance(turn.text, speaker=turn.speaker, attribute="dialogue")
            source_record_id = f"{record.question_id}-turn-{turn_counter}"
            updates.append(
                MemoryRecord(
                    record_id=source_record_id,
                    entity=turn.speaker,
                    attribute="dialogue",
                    value=turn.text,
                    timestamp=turn_counter,
                    session_id=record.question_id,
                    confidence=confidence,
                    importance=importance,
                    scope=scope,
                    provenance=f"longmemeval_raw_s{session_idx}",
                    memory_kind="event",
                    speaker=turn.speaker,
                    source_date=session.date,
                    session_label=session.label,
                )
            )
            for fact_idx, fact in enumerate(extract_structured_facts(turn.text)):
                updates.append(
                    MemoryRecord(
                        record_id=f"{record.question_id}-turn-{turn_counter}-fact-{fact_idx}",
                        entity=turn.speaker,
                        attribute=fact.attribute,
                        value=fact.value,
                        timestamp=turn_counter,
                        session_id=record.question_id,
                        confidence=min(1.0, confidence + 0.08),
                        importance=min(1.8, importance + 0.15),
                        scope=scope if fact.is_stateful else f"{scope}:turn_{turn_counter}:fact_{fact_idx}",
                        provenance=f"longmemeval_raw_s{session_idx}_fact",
                        source_kind="structured_fact",
                        source_attribute="dialogue",
                        memory_kind="state" if fact.is_stateful else "event",
                        source_entry_id=source_record_id,
                        support_text=compact_support_text(turn.text),
                        speaker=turn.speaker,
                        source_date=session.date,
                        session_label=session.label,
                    )
                )
            turn_counter += 1

    return ExperimentContext(
        context_id=record.question_id,
        session_id=record.question_id,
        updates=updates,
        metadata={
            "question_id": record.question_id,
            "question_type": record.question_type,
            "num_haystack_sessions": str(len(record.sessions)),
        },
    )


def _compile_longmemeval_cases(record: RawLongMemEvalRecord, context: ExperimentContext) -> list[ExperimentCase]:
    query_entity = infer_longmemeval_query_entity(record.question_type)
    query_attribute = choose_query_attribute(
        record.question,
        query_entity,
        context.updates,
        default_attribute="dialogue",
    )
    runtime_query = RuntimeQuery(
        query_id=record.question_id,
        context_id=context.context_id,
        entity=query_entity,
        attribute=query_attribute,
        question=record.question,
        timestamp=max((update.timestamp for update in context.updates), default=0) + 1,
        session_id=context.session_id,
        multi_attributes=record.multi_attributes,
        query_mode=longmemeval_query_mode(record.question_type),
        supports_abstention=record.question_id.endswith("_abs"),
    )
    return [
        ExperimentCase(
            case_id=record.question_id,
            context_id=context.context_id,
            runtime_query=runtime_query,
            eval_target=EvalTarget(
                query_id=record.question_id,
                gold_answer=record.answer,
                benchmark_name="longmemeval",
                benchmark_category=record.question_type,
                supports_abstention=runtime_query.supports_abstention,
            ),
            metadata={"question_type": record.question_type},
        )
    ]

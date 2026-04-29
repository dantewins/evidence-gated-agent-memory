from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Iterable, List

from memory_inference.domain.benchmark import ExperimentCase, ExperimentContext
from memory_inference.domain.memory import RetrievalBundle
from memory_inference.domain.query import RuntimeQuery
from memory_inference.domain.results import ExecutedCase
from memory_inference.llm.base import BaseReasoner
from memory_inference.memory.policies.interface import BaseMemoryPolicy
from memory_inference.orchestration.postprocess import format_multihop_prediction


@dataclass(slots=True)
class ContextCaseRunner:
    policy: BaseMemoryPolicy
    reasoner: BaseReasoner
    prepared_context_id: str | None = None

    def prepare_context(self, context: ExperimentContext) -> None:
        if self.prepared_context_id == context.context_id:
            return
        if self.prepared_context_id is not None and self.prepared_context_id != context.context_id:
            raise ValueError(
                f"Runner already prepared for context {self.prepared_context_id}; "
                f"create a new runner for context {context.context_id}"
            )
        self.policy.ingest(context.updates)
        self.policy.maybe_consolidate()
        self.prepared_context_id = context.context_id

    def run_case(self, case: ExperimentCase) -> ExecutedCase:
        if self.prepared_context_id != case.context_id:
            raise ValueError(
                f"Runner prepared for context {self.prepared_context_id}; "
                f"cannot execute case for context {case.context_id}"
            )
        runtime_query = self._query_with_case_metadata(case)
        retrieved_bundle = self._retrieve(runtime_query)
        trace = self.reasoner.answer_with_trace(case.runtime_query, retrieved_bundle.records)
        prediction = trace.answer
        if case.runtime_query.multi_attributes:
            prediction = format_multihop_prediction(prediction, case.runtime_query, retrieved_bundle.records)
        return ExecutedCase(
            case=case,
            retrieval_bundle=retrieved_bundle,
            reader_trace=trace,
            prediction=prediction,
            policy_name=self.policy.name,
        )

    def run_cases_for_context(
        self,
        context: ExperimentContext,
        cases: Iterable[ExperimentCase],
    ) -> List[ExecutedCase]:
        self.prepare_context(context)
        case_list = list(cases)
        runtime_queries = [self._query_with_case_metadata(case) for case in case_list]
        retrieved_bundles = [self._retrieve(query) for query in runtime_queries]
        traces = self.reasoner.answer_many_with_traces(
            [case.runtime_query for case in case_list],
            [bundle.records for bundle in retrieved_bundles],
        )
        executed: list[ExecutedCase] = []
        for case, retrieved_bundle, trace in zip(case_list, retrieved_bundles, traces):
            prediction = trace.answer
            if case.runtime_query.multi_attributes:
                prediction = format_multihop_prediction(
                    prediction,
                    case.runtime_query,
                    retrieved_bundle.records,
                )
            executed.append(
                ExecutedCase(
                    case=case,
                    retrieval_bundle=retrieved_bundle,
                    reader_trace=trace,
                    prediction=prediction,
                    policy_name=self.policy.name,
                )
            )
        return executed

    def _retrieve(self, query) -> RetrievalBundle:
        bundle = self._retrieve_for_query(query)
        records = list(bundle.records)
        for attr in query.multi_attributes:
            subquery = dataclasses.replace(query, attribute=attr, multi_attributes=())
            records.extend(self._retrieve_for_query(subquery).records)
        seen_ids: set[str] = set()
        deduped = []
        for record in records:
            if record.record_id in seen_ids:
                continue
            seen_ids.add(record.record_id)
            deduped.append(record)
        return RetrievalBundle(records=deduped, debug=dict(bundle.debug))

    def _retrieve_for_query(self, query) -> RetrievalBundle:
        return self.policy.retrieve_for_query(query)

    def _query_with_case_metadata(self, case: ExperimentCase) -> RuntimeQuery:
        category = (
            case.eval_target.benchmark_category
            or case.metadata.get("question_category", "")
            or case.metadata.get("question_type", "")
        )
        if not category:
            return case.runtime_query
        return dataclasses.replace(
            case.runtime_query,
            metadata={
                **case.metadata,
                "benchmark_category": str(category),
            },
        )

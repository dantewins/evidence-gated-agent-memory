"""Report generation: LaTeX tables, pgfplots data, and Markdown summaries."""
from __future__ import annotations

from typing import Sequence

from memory_inference.metrics import ExperimentMetrics
from memory_inference.statistics import ConfidenceInterval
from memory_inference.types import InferenceExample


def latex_main_table(
    metrics: Sequence[ExperimentMetrics],
    cis: dict[str, ConfidenceInterval] | None = None,
) -> str:
    """Generate LaTeX for the main results table (booktabs style)."""
    lines = [
        r"\begin{table}[t]",
        r"\caption{Synthetic benchmark results with a frozen Qwen2.5-7B-Instruct reader.}",
        r"\label{tab:main-results}",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Policy & Accuracy & State EM & Tokens & Snapshot \\",
        r"\midrule",
    ]
    best_acc = max(m.accuracy for m in metrics)
    for m in metrics:
        acc_str = f"{m.accuracy:.3f}"
        if cis and m.policy_name in cis:
            ci = cis[m.policy_name]
            acc_str = f"{ci.mean:.3f} [{ci.lower:.3f}, {ci.upper:.3f}]"
        if m.accuracy == best_acc:
            acc_str = r"\textbf{" + acc_str + "}"
        name = _latex_policy_name(m.policy_name)
        lines.append(
            f"{name} & {acc_str} & {m.current_state_exact_match:.3f} "
            f"& {m.amortized_end_to_end_tokens:.1f} & {m.avg_snapshot_size:.1f} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def latex_ablation_table(metrics: Sequence[ExperimentMetrics]) -> str:
    """Generate LaTeX for the ablation study table."""
    lines = [
        r"\begin{table}[t]",
        r"\caption{Ablation study: disabling individual ODV2 components (deterministic reader).}",
        r"\label{tab:ablation}",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"Variant & State EM & Sup.\ P & Sup.\ R & Conf.\ F1 & Scope \\",
        r"\midrule",
    ]
    for m in metrics:
        name = _latex_policy_name(m.policy_name)
        lines.append(
            f"{name} & {m.current_state_exact_match:.3f} "
            f"& {m.supersession_precision:.3f} & {m.supersession_recall:.3f} "
            f"& {m.conflict_detection_f1:.3f} & {m.scope_split_accuracy:.3f} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def latex_scenario_table(
    all_examples: dict[str, Sequence[InferenceExample]],
) -> str:
    """Generate per-scenario-family accuracy table.

    *all_examples* maps policy_name -> list of InferenceExample.
    """
    from memory_inference.statistics import scenario_family_breakdown

    families: set[str] = set()
    breakdowns: dict[str, dict[str, float]] = {}
    for policy_name, examples in all_examples.items():
        bd = scenario_family_breakdown(examples)
        breakdowns[policy_name] = {
            fam: (sum(vals) / len(vals) if vals else 0.0)
            for fam, vals in bd.items()
        }
        families.update(bd.keys())

    sorted_families = sorted(families)
    policy_names = list(all_examples.keys())
    col_spec = "l" + "c" * len(sorted_families)
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Per-scenario accuracy breakdown.}",
        r"\label{tab:scenario-breakdown}",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{@{}" + col_spec + r"@{}}",
        r"\toprule",
        "Policy & " + " & ".join(sorted_families) + r" \\",
        r"\midrule",
    ]
    for pn in policy_names:
        vals = " & ".join(
            f"{breakdowns[pn].get(f, 0.0):.3f}" for f in sorted_families
        )
        lines.append(f"{_latex_policy_name(pn)} & {vals} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def pgfplots_cost_accuracy(metrics: Sequence[ExperimentMetrics]) -> str:
    """Generate .dat file content for cost-accuracy scatter plot."""
    lines = ["policy tokens accuracy"]
    for m in metrics:
        name = m.policy_name.replace("_", "-")
        lines.append(f"{name} {m.amortized_end_to_end_tokens:.2f} {m.accuracy:.4f}")
    return "\n".join(lines)


def pgfplots_scenario_bars(
    all_examples: dict[str, Sequence[InferenceExample]],
) -> str:
    """Generate .dat file for grouped bar chart of per-scenario accuracy."""
    from memory_inference.statistics import scenario_family_breakdown

    families: set[str] = set()
    breakdowns: dict[str, dict[str, float]] = {}
    for policy_name, examples in all_examples.items():
        bd = scenario_family_breakdown(examples)
        breakdowns[policy_name] = {
            fam: (sum(vals) / len(vals) if vals else 0.0)
            for fam, vals in bd.items()
        }
        families.update(bd.keys())

    sorted_families = sorted(families)
    policy_names = list(all_examples.keys())
    header = "scenario " + " ".join(pn.replace("_", "-") for pn in policy_names)
    lines = [header]
    for fam in sorted_families:
        vals = " ".join(f"{breakdowns[pn].get(fam, 0.0):.4f}" for pn in policy_names)
        lines.append(f"{fam} {vals}")
    return "\n".join(lines)


def markdown_summary(metrics: Sequence[ExperimentMetrics]) -> str:
    """Generate a human-readable Markdown summary table."""
    lines = [
        "| Policy | Accuracy | State EM | Sup. P | Sup. R | Conf. F1 | Tokens |",
        "|--------|----------|----------|--------|--------|----------|--------|",
    ]
    for m in metrics:
        lines.append(
            f"| {m.policy_name} | {m.accuracy:.3f} | {m.current_state_exact_match:.3f} "
            f"| {m.supersession_precision:.3f} | {m.supersession_recall:.3f} "
            f"| {m.conflict_detection_f1:.3f} | {m.amortized_end_to_end_tokens:.1f} |"
        )
    return "\n".join(lines)


def _latex_policy_name(name: str) -> str:
    """Convert policy name to LaTeX-friendly display name."""
    mapping = {
        "append_only": "Append-only",
        "recency_salience": "Recency-sal.",
        "summary_only": "Summary-only",
        "exact_match": "Exact-match",
        "strong_retrieval": "Strong retr.",
        "offline_delta_v2": "ODV2 (full)",
        "odv2_no_revert": r"$-$Revert",
        "odv2_no_conflict": r"$-$Conflict",
        "odv2_no_scope": r"$-$Scope",
        "odv2_no_archive": r"$-$Archive",
    }
    return mapping.get(name, name.replace("_", r"\_"))

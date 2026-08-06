"""Measuring whether the answers are right, not just well formed."""

from nl2sql.evaluation.models import CaseOutcome, EvaluationCase, EvaluationReport
from nl2sql.evaluation.runner import evaluate, load_cases

__all__ = [
    "CaseOutcome",
    "EvaluationCase",
    "EvaluationReport",
    "evaluate",
    "load_cases",
]

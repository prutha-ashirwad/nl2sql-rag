"""The evaluation harness grades the system, so the grader itself needs pinning down.

An accuracy figure is only worth as much as the comparison behind it. These tests fix
what counts as a correct answer: row order must not matter, integer and float spellings
of the same number must not matter, extra descriptive columns are allowed, and a missing
reference column is always a failure.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nl2sql.evaluation.models import (
    CaseOutcome,
    EvaluationCase,
    EvaluationReport,
    Expectation,
    Verdict,
)
from nl2sql.evaluation.runner import _compare, load_cases


def case(case_id: str = "T-1", **kwargs) -> EvaluationCase:
    defaults = {
        "question": "q",
        "category": "test",
        "gold_sql": "SELECT 1",
    }
    return EvaluationCase(id=case_id, **{**defaults, **kwargs})


class TestComparingResultSets:
    def test_identical_results_are_exact(self) -> None:
        rows = [{"environment_name": "Production", "n": 3}]
        assert _compare(rows, list(rows))[0] is Verdict.EXACT

    def test_row_order_does_not_matter(self) -> None:
        gold = [{"name": "a"}, {"name": "b"}]
        got = [{"name": "b"}, {"name": "a"}]

        assert _compare(gold, got)[0] is Verdict.EXACT

    def test_an_integer_and_its_float_spelling_agree(self) -> None:
        # SQLite returns COUNT(*) as int; another engine may return 3.0.
        assert _compare([{"n": 3}], [{"n": 3.0}])[0] is Verdict.EXACT

    def test_extra_descriptive_columns_still_answer_the_question(self) -> None:
        gold = [{"interface_name": "eth0", "failure_count": 2}]
        got = [{"interface_name": "eth0", "device_name": "core-1", "failure_count": 2}]

        verdict, _ = _compare(gold, got)
        assert verdict is Verdict.ANSWER_SET

    def test_a_missing_reference_column_fails(self) -> None:
        gold = [{"interface_name": "eth0", "failure_count": 2}]
        got = [{"interface_name": "eth0"}]

        verdict, detail = _compare(gold, got)
        assert verdict is Verdict.WRONG_ROWS
        assert "failure_count" in detail

    def test_a_differently_aliased_column_fails_rather_than_guessing(self) -> None:
        # Deliberately conservative: matching by value would risk crediting a wrong
        # query that happens to line up, so the harness only ever under-reports.
        gold = [{"failure_count": 2}]
        got = [{"total": 2}]

        assert _compare(gold, got)[0] is Verdict.WRONG_ROWS

    def test_the_table_qualifier_is_ignored_when_matching(self) -> None:
        gold = [{"environment_name": "Production"}]
        got = [{"e.environment_name": "Production"}]

        assert _compare(gold, got)[0] is Verdict.EXACT

    def test_wrong_values_fail(self) -> None:
        assert _compare([{"n": 3}], [{"n": 4}])[0] is Verdict.WRONG_ROWS

    def test_a_missing_row_fails(self) -> None:
        gold = [{"name": "a"}, {"name": "b"}]

        verdict, detail = _compare(gold, [{"name": "a"}])
        assert verdict is Verdict.WRONG_ROWS
        assert "1 missing" in detail

    def test_duplicate_rows_are_counted_not_collapsed(self) -> None:
        # A GROUP BY that lost its grouping returns the right values the wrong number
        # of times; comparing sets rather than multisets would call that correct.
        gold = [{"name": "a"}, {"name": "a"}]

        assert _compare(gold, [{"name": "a"}])[0] is Verdict.WRONG_ROWS

    def test_two_empty_results_agree(self) -> None:
        assert _compare([], [])[0] is Verdict.EXACT

    def test_rows_where_none_were_expected_fail(self) -> None:
        assert _compare([], [{"n": 1}])[0] is Verdict.WRONG_ROWS

    def test_nulls_compare_equal_to_each_other(self) -> None:
        assert _compare([{"reason": None}], [{"reason": None}])[0] is Verdict.EXACT

    def test_null_and_empty_string_are_not_the_same(self) -> None:
        assert _compare([{"reason": None}], [{"reason": ""}])[0] is Verdict.WRONG_ROWS


class TestCaseDeclarations:
    def test_an_answerable_case_needs_a_reference_query(self) -> None:
        with pytest.raises(ValueError, match="needs gold_sql"):
            EvaluationCase(id="X", question="q", category="c", gold_sql=None)

    def test_a_refusal_case_must_not_carry_one(self) -> None:
        with pytest.raises(ValueError, match="must not carry gold_sql"):
            EvaluationCase(
                id="X",
                question="q",
                category="c",
                expect=Expectation.REFUSED,
                gold_sql="SELECT 1",
            )

    def test_duplicate_ids_are_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "cases.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "cases": [
                        {"id": "D", "question": "a", "category": "c", "gold_sql": "S"},
                        {"id": "D", "question": "b", "category": "c", "gold_sql": "S"},
                    ]
                }
            )
        )

        with pytest.raises(ValueError, match="Duplicate case ids"):
            load_cases(path)


class TestReportTotals:
    @staticmethod
    def _report(*verdicts: Verdict) -> EvaluationReport:
        report = EvaluationReport(engine="test")
        for index, verdict in enumerate(verdicts):
            report.outcomes.append(CaseOutcome(case=case(f"T-{index}"), verdict=verdict))
        return report

    def test_accuracy_counts_exact_answer_set_and_correct_refusals(self) -> None:
        report = self._report(
            Verdict.EXACT,
            Verdict.ANSWER_SET,
            Verdict.REFUSED_CORRECTLY,
            Verdict.WRONG_ROWS,
        )

        assert (report.passed, report.total) == (3, 4)
        assert report.accuracy == 0.75

    def test_a_broken_reference_query_is_excluded_from_the_denominator(self) -> None:
        # Otherwise a mistake in the case file would quietly depress the score.
        report = self._report(Verdict.EXACT, Verdict.GOLD_BROKEN)

        assert (report.passed, report.total) == (1, 1)
        assert report.accuracy == 1.0
        assert len(report.broken_cases) == 1

    def test_answering_a_question_that_should_be_refused_is_a_failure(self) -> None:
        report = self._report(Verdict.ANSWERED_WRONGLY)

        assert report.accuracy == 0.0


class TestTheShippedCaseFile:
    """The committed set has to stay loadable and honest, or the figure is stale."""

    def test_it_loads(self) -> None:
        assert load_cases()

    def test_every_case_declares_what_it_is_testing(self) -> None:
        for entry in load_cases():
            assert entry.category, f"{entry.id} has no category"
            assert entry.question.strip(), f"{entry.id} has no question"

    def test_no_case_reuses_a_knowledge_base_example(self) -> None:
        # The 12 curated examples are inside the retrieval corpus. Scoring against them
        # would measure recall of a few-shot prompt, not generalisation.
        from nl2sql.config import get_settings
        from nl2sql.knowledge_base.loader import load_knowledge_base

        knowledge_base = load_knowledge_base(get_settings().knowledge_base_path)
        seen = {
            example.question.strip().rstrip(".").lower()
            for example in knowledge_base.examples
        }

        for entry in load_cases():
            assert entry.question.strip().rstrip(".").lower() not in seen, (
                f"{entry.id} reuses a Knowledge Base example verbatim"
            )

"""Tests for SQL validation against the Knowledge Base."""

from __future__ import annotations

import pytest

from nl2sql.validation.models import IssueCode
from nl2sql.validation.validator import SQLValidator

VALID_QUERY = """
SELECT o.observation_id, i.interface_name
FROM observations o
INNER JOIN interfaces i ON o.interface_id = i.interface_id
WHERE o.status = 'FAILED'
LIMIT 10
"""


def codes(report) -> set[IssueCode]:  # noqa: ANN001 - ValidationReport
    """Collect the issue codes present on a report."""
    return {issue.code for issue in report.issues}


class TestValidQueries:
    def test_accepts_a_well_formed_query(self, validator: SQLValidator) -> None:
        report = validator.validate(VALID_QUERY)
        assert report.is_valid, report.error_messages()

    def test_records_referenced_tables(self, validator: SQLValidator) -> None:
        report = validator.validate(VALID_QUERY)
        assert set(report.referenced_tables) == {"observations", "interfaces"}

    def test_accepts_aggregates_with_aliased_ordering(
        self, validator: SQLValidator
    ) -> None:
        report = validator.validate(
            """
            SELECT e.environment_name, COUNT(*) AS observation_count
            FROM observations o
            INNER JOIN environments e ON o.environment_id = e.environment_id
            GROUP BY e.environment_name
            ORDER BY observation_count DESC
            """
        )
        assert report.is_valid, report.error_messages()

    def test_accepts_common_table_expressions(
        self, validator: SQLValidator
    ) -> None:
        report = validator.validate(
            """
            WITH failed AS (
                SELECT o.device_id FROM observations o WHERE o.status = 'FAILED'
            )
            SELECT COUNT(*) AS failure_count FROM failed
            """
        )
        assert report.is_valid, report.error_messages()

    def test_accepts_a_multi_hop_join(self, validator: SQLValidator) -> None:
        report = validator.validate(
            """
            SELECT s.site_name, COUNT(*) AS failure_count
            FROM observations o
            INNER JOIN devices d ON o.device_id = d.device_id
            INNER JOIN sites s ON d.site_id = s.site_id
            GROUP BY s.site_name
            """
        )
        assert report.is_valid, report.error_messages()


class TestSafety:
    @pytest.mark.parametrize(
        "sql",
        [
            "DELETE FROM observations",
            "UPDATE devices SET device_status = 'ACTIVE'",
            "DROP TABLE observations",
            "INSERT INTO teams (team_id) VALUES (1)",
        ],
    )
    def test_rejects_write_statements(
        self, validator: SQLValidator, sql: str
    ) -> None:
        report = validator.validate(sql)
        assert not report.is_valid
        assert IssueCode.NOT_READ_ONLY in codes(report)

    def test_rejects_multiple_statements(self, validator: SQLValidator) -> None:
        report = validator.validate(
            "SELECT 1 FROM observations; DROP TABLE observations"
        )
        assert not report.is_valid

    def test_rejects_an_empty_query(self, validator: SQLValidator) -> None:
        report = validator.validate("   ")
        assert IssueCode.EMPTY_QUERY in codes(report)

    def test_rejects_unparseable_sql(self, validator: SQLValidator) -> None:
        report = validator.validate("SELECT FROM WHERE ORDER GROUP")
        assert not report.is_valid

    def test_reports_an_unterminated_string_instead_of_raising(
        self, validator: SQLValidator
    ) -> None:
        # An apostrophe inside a value fails while tokenising, which raises a
        # TokenError — a sibling of ParseError, not a subclass. Catching only
        # ParseError let it escape the validator as an unhandled exception.
        report = validator.validate(
            "SELECT * FROM devices WHERE device_name = 'O'Brien'"
        )

        assert not report.is_valid
        assert IssueCode.SYNTAX_ERROR in codes(report)


class TestSchemaConformance:
    def test_rejects_an_unknown_table(self, validator: SQLValidator) -> None:
        report = validator.validate("SELECT * FROM invented_table")
        assert IssueCode.UNKNOWN_TABLE in codes(report)

    def test_rejects_an_unknown_column(self, validator: SQLValidator) -> None:
        report = validator.validate(
            "SELECT o.not_a_real_column FROM observations o"
        )
        assert IssueCode.UNKNOWN_COLUMN in codes(report)

    def test_rejects_an_unbound_alias(self, validator: SQLValidator) -> None:
        report = validator.validate("SELECT x.status FROM observations o")
        assert IssueCode.UNKNOWN_TABLE in codes(report)

    def test_rejects_an_undeclared_join(self, validator: SQLValidator) -> None:
        # devices.device_id = teams.team_id is not a declared relationship.
        report = validator.validate(
            """
            SELECT d.device_name FROM devices d
            INNER JOIN teams t ON d.device_id = t.team_id
            """
        )
        assert IssueCode.UNDECLARED_JOIN in codes(report)

    def test_rejects_a_join_without_a_condition(
        self, validator: SQLValidator
    ) -> None:
        report = validator.validate(
            "SELECT d.device_name FROM devices d CROSS JOIN teams t"
        )
        assert IssueCode.MISSING_JOIN_CONDITION in codes(report)

    def test_rejects_a_miscased_enum_literal(
        self, validator: SQLValidator
    ) -> None:
        # The database stores 'FAILED'; 'failed' silently returns zero rows.
        report = validator.validate(
            "SELECT o.observation_id FROM observations o WHERE o.status = 'failed'"
        )
        assert IssueCode.INVALID_ENUM_VALUE in codes(report)

    def test_rejects_an_invalid_value_in_an_in_list(
        self, validator: SQLValidator
    ) -> None:
        report = validator.validate(
            """
            SELECT o.observation_id FROM observations o
            WHERE o.status IN ('FAILED', 'BROKEN')
            """
        )
        assert IssueCode.INVALID_ENUM_VALUE in codes(report)

    def test_accepts_a_valid_enum_literal(self, validator: SQLValidator) -> None:
        report = validator.validate(
            "SELECT o.observation_id FROM observations o WHERE o.status = 'FAILED'"
        )
        assert report.is_valid, report.error_messages()


class TestWarnings:
    def test_select_star_is_rejected(self, validator: SQLValidator) -> None:
        # Every table is keyed on surrogate ids, so `SELECT *` returns columns that
        # mean nothing to a reader. It goes back through repair instead.
        report = validator.validate("SELECT * FROM observations o LIMIT 5")
        assert not report.is_valid
        assert IssueCode.SELECT_STAR in codes(report)

    def test_named_columns_are_accepted(self, validator: SQLValidator) -> None:
        report = validator.validate(
            "SELECT o.observation_id, o.status FROM observations o LIMIT 5"
        )
        assert report.is_valid, report.error_messages()

    def test_ambiguous_column_is_a_warning(self, validator: SQLValidator) -> None:
        # `created_at` exists on both tables, so an unqualified reference is unclear.
        report = validator.validate(
            """
            SELECT created_at FROM observations o
            INNER JOIN devices d ON o.device_id = d.device_id
            """
        )
        assert IssueCode.AMBIGUOUS_COLUMN in codes(report)


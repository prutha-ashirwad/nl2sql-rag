"""Tests for the Knowledge Base indexes: lexicon, enums, aliases and join paths."""

from __future__ import annotations

from nl2sql.knowledge_base.registry import KnowledgeBaseRegistry


class TestLookups:
    def test_finds_tables_case_insensitively(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        assert registry.get_table("OBSERVATIONS") is not None
        assert registry.has_table("observations")
        assert registry.get_table("no_such_table") is None

    def test_resolves_columns(self, registry: KnowledgeBaseRegistry) -> None:
        column = registry.get_column("observations", "observed_at")
        assert column is not None
        assert column.role.value == "timestamp"

    def test_prefers_event_time_over_load_time(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        primary = registry.primary_timestamp_column("observations")
        assert primary is not None
        assert primary.name == "observed_at"

    def test_assigns_unique_aliases(self, registry: KnowledgeBaseRegistry) -> None:
        aliases = [registry.alias_for(name) for name in registry.table_names]
        assert len(aliases) == len(set(aliases))

    def test_honours_preferred_aliases(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        assert registry.alias_for("observations") == "o"
        assert registry.alias_for("interfaces") == "i"
        assert registry.alias_for("devices") == "d"


class TestLexicon:
    def test_resolves_table_synonyms(self, registry: KnowledgeBaseRegistry) -> None:
        matches = registry.resolve_tables_for_phrase("port")
        assert "interfaces" in [match.table for match in matches]

    def test_scores_a_table_name_above_a_glossary_mention(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        direct = registry.resolve_tables_for_phrase("interfaces")[0]
        indirect = registry.resolve_tables_for_phrase("failure count")[0]
        assert direct.score > indirect.score

    def test_resolves_value_synonyms_to_stored_literals(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        matches = registry.resolve_enum_value("failure")
        statuses = [
            match for match in matches if match.table == "observations"
        ]
        assert statuses and statuses[0].value == "FAILED"

    def test_resolves_literal_values(self, registry: KnowledgeBaseRegistry) -> None:
        matches = registry.resolve_enum_value("PROD")
        assert any(match.column == "environment_code" for match in matches)

    def test_dimension_lexicon_excludes_measures(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        # "failure count" names a measure, so grouping by it would be meaningless.
        assert registry.resolve_dimension_for_phrase("failure count") == []
        assert registry.resolve_dimension_for_phrase("environment")

    def test_fact_tables_have_no_label_column(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        assert registry.label_column("observations") is None
        assert registry.label_column("environments") == "environment_name"

    def test_resolves_declared_metrics(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        metric = registry.resolve_metric_for_phrase("success rate")
        assert metric is not None
        assert metric.metric_alias == "success_rate_pct"


class TestJoinGraph:
    def test_direct_join_is_a_single_hop(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        path = registry.find_join_path("observations", "interfaces")
        assert path is not None
        assert len(path) == 1
        assert path[0].target_table == "interfaces"

    def test_multi_hop_join_walks_the_hierarchy(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        path = registry.find_join_path("observations", "regions")
        assert path is not None
        assert [step.target_table for step in path] == ["devices", "sites", "regions"]

    def test_prefers_the_business_path_over_an_incidental_one(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        # sites is reachable via devices or via collectors; the device path is the
        # meaningful one and carries a lower traversal cost.
        path = registry.find_join_path("observations", "sites")
        assert path is not None
        assert path[0].target_table == "devices"

    def test_same_table_needs_no_join(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        assert registry.find_join_path("observations", "observations") == []

    def test_reports_unreachable_tables(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        _, unreachable = registry.build_join_plan("observations", ["no_such_table"])
        assert unreachable == ["no_such_table"]

    def test_shared_hops_are_emitted_once(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        steps, unreachable = registry.build_join_plan(
            "observations", ["sites", "regions", "device_models"]
        )
        assert not unreachable
        targets = [step.target_table for step in steps]
        assert len(targets) == len(set(targets)), "a table was joined twice"
        assert targets.count("devices") == 1

    def test_grain_preserving_search_rejects_fan_out(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        # One observation has many metric samples, so this join multiplies rows.
        assert registry.find_join_path("observations", "observation_metrics") is not None
        assert (
            registry.find_join_path(
                "observations", "observation_metrics", preserve_grain=True
            )
            is None
        )

    def test_join_direction_marks_grain_preservation(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        forward = registry.find_join_path("observations", "devices")
        reverse = registry.find_join_path("devices", "observations")
        assert forward is not None and forward[0].preserves_grain
        assert reverse is not None and not reverse[0].preserves_grain

    def test_nullable_foreign_keys_use_a_left_join(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        path = registry.find_join_path("observations", "failure_reasons")
        assert path is not None
        assert path[0].join_type == "LEFT"

    def test_path_finding_is_deterministic(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        first = registry.find_join_path("observations", "regions")
        for _ in range(20):
            assert registry.find_join_path("observations", "regions") == first

    def test_every_table_is_reachable_from_the_core_fact_table(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        unreachable = [
            name
            for name in registry.table_names
            if registry.find_join_path("observations", name) is None
        ]
        assert unreachable == []


class TestRuleScoping:
    def test_global_rules_apply_everywhere(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        rules = registry.rules_for_tables({"regions"})
        assert any(rule.id == "RULE-001" for rule in rules)

    def test_scoped_rules_only_apply_to_their_tables(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        scoped = [rule for rule in registry.rules if rule.applies_to]
        assert scoped, "expected at least one table-scoped rule"

        rule = scoped[0]
        assert rule.applies_to_tables(set(rule.applies_to))
        assert not rule.applies_to_tables({"a_table_the_rule_does_not_cover"})

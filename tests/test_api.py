"""Tests for the HTTP API."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from nl2sql.api import app
from nl2sql.pipeline import NL2SQLPipeline


@pytest.fixture(scope="module")
def client(pipeline: NL2SQLPipeline) -> Iterator[TestClient]:
    """A test client with the shared pipeline injected.

    Injecting before start-up keeps the tests pointed at the seeded test database.
    The lifespan reuses whatever is already on ``app.state``, so it never falls back
    to building a pipeline from the developer's own configuration.
    """
    app.state.pipeline = pipeline
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.state.pipeline = None


def test_lifespan_does_not_replace_an_injected_pipeline(
    client: TestClient, pipeline: NL2SQLPipeline
) -> None:
    """Start-up must not discard a pipeline the host already supplied."""
    assert app.state.pipeline is pipeline


class TestHealth:
    def test_reports_knowledge_base_coverage(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["knowledge_base"]["tables"] >= 20
        assert payload["knowledge_base"]["relationships"] >= 5


class TestAsk:
    def test_answers_a_question(self, client: TestClient) -> None:
        response = client.post(
            "/ask", json={"question": "Display observations grouped by environment."}
        )
        assert response.status_code == 200

        payload = response.json()
        assert payload["succeeded"] is True
        assert "GROUP BY" in payload["sql"]
        assert payload["rows"]

    def test_omits_the_trace_unless_requested(self, client: TestClient) -> None:
        response = client.post("/ask", json={"question": "Show failed observations."})
        assert response.json()["trace"] is None

    def test_includes_the_trace_on_request(self, client: TestClient) -> None:
        response = client.post(
            "/ask",
            json={"question": "Show failed observations.", "include_trace": True},
        )
        trace = response.json()["trace"]
        assert trace
        assert trace[0]["node"] == "analyze_question"

    def test_unanswerable_question_is_a_normal_response(
        self, client: TestClient
    ) -> None:
        # Not a client error: the request was well formed, the answer is just "no".
        response = client.post("/ask", json={"question": "Delete all observations"})
        assert response.status_code == 200

        payload = response.json()
        assert payload["succeeded"] is False
        assert payload["sql"] is None

    def test_rejects_an_empty_question(self, client: TestClient) -> None:
        assert client.post("/ask", json={"question": ""}).status_code == 422

    def test_rejects_a_missing_question(self, client: TestClient) -> None:
        assert client.post("/ask", json={}).status_code == 422


class TestSchemaEndpoints:
    def test_lists_every_table(self, client: TestClient) -> None:
        payload = client.get("/schema/tables").json()
        assert payload["count"] >= 20
        assert any(table["name"] == "observations" for table in payload["tables"])

    def test_returns_full_metadata_for_one_table(self, client: TestClient) -> None:
        payload = client.get("/schema/tables/observations").json()
        assert payload["name"] == "observations"
        assert any(column["name"] == "observed_at" for column in payload["columns"])

    def test_unknown_table_is_a_404(self, client: TestClient) -> None:
        assert client.get("/schema/tables/no_such_table").status_code == 404

    def test_lists_relationships(self, client: TestClient) -> None:
        payload = client.get("/schema/relationships").json()
        assert payload["count"] >= 5
        assert all(
            {"from_table", "to_table", "cardinality"} <= set(relationship)
            for relationship in payload["relationships"]
        )

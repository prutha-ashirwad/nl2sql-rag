# Intelligent NL2SQL System

Turns natural-language questions into validated SQL, using a **LangGraph** workflow and a
**RAG** pipeline over a structured Knowledge Base.

The agent is never asked to recall the schema. Before any SQL is written it retrieves the
relevant tables, join paths, business definitions and SQL rules from the Knowledge Base,
generates against that context, then validates the result against the same source of
truth. That is what keeps invented table and column names out of the output.

```
"Show all failed observations in the last 24 hours."

  ▼

SELECT o.observation_id, o.observed_at, d.device_name, i.interface_name,
       e.environment_name, fr.reason_name AS failure_reason
FROM observations o
INNER JOIN interfaces i        ON o.interface_id = i.interface_id
INNER JOIN devices d           ON o.device_id = d.device_id
INNER JOIN environments e      ON o.environment_id = e.environment_id
LEFT  JOIN failure_reasons fr  ON o.failure_reason_id = fr.failure_reason_id
WHERE o.status = 'FAILED'
  AND o.observed_at >= datetime('now', '-24 hours')
ORDER BY o.observed_at DESC
```

**Contents** — [Architecture](#architecture) · [Quick start](#quick-start) ·
[Screenshots](#screenshots) · [How it works](#how-it-works) ·
[Knowledge Base](#the-knowledge-base) ·
[Adding a table](#adding-a-table-without-touching-code) · [Interfaces](#interfaces) ·
[Configuration](#configuration) · [Testing](#testing) ·
[Project structure](#project-structure) · [Design decisions](#design-decisions)

---

## Architecture

![NL2SQL system architecture](assets/architecture.svg)

Four entry points converge on a single facade, `NL2SQLPipeline`, which compiles one
LangGraph agent and reuses it. The agent's seven nodes are the whole control flow: there
is no free-running loop and no step that can be reached without a declared edge.

Four of the five subsystems read the **same** Knowledge Base — the analyzer resolves
phrases against it, retrieval ranks its documents, generation is prompted from it, and
validation checks the result against it. Only the executor touches the database. That
single source of truth is what makes the repair loop converge: an invented column fails
against the very file that produced the prompt.

| | |
| :-- | :-- |
| Entry points | 4 — web interface, CLI, HTTP API, Python import |
| Agent nodes | 7, with a repair loop bounded by `MAX_REPAIR_ATTEMPTS` |
| Retrieval corpus | 100 documents compiled from the Knowledge Base |
| Knowledge Base | 23 tables · 170 columns · 34 relationships · 15 rules · 16 glossary terms · 12 examples |
| Validation | 12 issue codes; all block execution except ambiguous-column, which warns |
| Swappable by config | model provider · embeddings · vector store · SQL dialect |
| Tests | 411 |

---

## Quick start

Python 3.11 or newer. **No API key is required** — with no model configured, the system
answers using its own Knowledge Base planner.

```bash
make venv
```

```bash
make install
```

```bash
make seed
```

```bash
make run
```

That serves the web interface on <http://127.0.0.1:8501>. It builds the pipeline
in-process, so this one command is the whole system.

The HTTP API is a separate, optional way in — nothing in the web interface depends on it:

```bash
make api
```

Both reload on change. To run them together in the background instead, use `make start`,
then `make status`, `make logs`, `make restart` and `make stop`. Each of those has a
`-frontend` and `-backend` variant for acting on one service (`make restart-backend`).

### Using a hosted model

Optional. Install the provider's SDK and set its key:

```bash
venv/bin/python -m pip install -e ".[openai]"
```

```bash
cp .env.example .env    # then set OPENAI_API_KEY
```

`".[anthropic]"` with `ANTHROPIC_API_KEY` works the same way. The model picker in the
sidebar lists every model and labels the ones whose key is missing, so what is configured
and what is not is always visible.

---

## Screenshots

**Ask** — a question, the answer, and the numbers behind it.

![The Ask page](assets/ask.png)

**Generated SQL** — the query that ran, followed by the workflow steps that produced it
and how long each took.

![Generated SQL and the workflow steps](assets/ask-sql.png)

**Agent Execution** — what every node decided: the parsed intent, what retrieval selected
and why, the joins it expanded, what validation checked.

![The Agent Execution page](assets/agent-execution.png)

**Knowledge Base** — browse the tables, relationships, rules and glossary the agent reads.
Tables can be added from the form or by editing the YAML directly, both validated.

![The Knowledge Base page](assets/knowledge-base.png)

**Compare Engines** — one question through every available engine, including the zero-cost
planner, so you can see whether a model earns its cost.

![The Compare Engines page](assets/compare-engines.png)

---

## How it works

### The LangGraph workflow

![The agent as a state machine](assets/workflow.svg)

The agent is a state machine, not a free-running loop. Every path is declared, every
router is a plain predicate over the state, and each transition is recorded in a trace the
interface renders back to you. Nothing about the control flow is decided by a model.

Four edges exit early straight to `finalize`, which is why a caller receives a formed
answer rather than an exception. `repair` returns to `validate`, not to `generate`, and the
router stops offering that edge once `MAX_REPAIR_ATTEMPTS` is spent — that is what makes
the one cycle in the graph terminate.

| Node | Responsibility |
| :-- | :-- |
| `analyze` | Classify intent; extract time windows, filters, groupings and the subject of the question. Write requests are rejected here. |
| `retrieve` | The RAG step: rank Knowledge Base documents, pick the anchor table, expand the join paths needed to reach every other table. |
| `generate` | Write SQL from the retrieved context, via a model or via the deterministic planner. |
| `validate` | Parse the SQL and check it against the Knowledge Base: real tables, real columns, bound aliases, declared joins, allowed literals, read-only, and a single statement — stacked queries are rejected outright. |
| `repair` | Feed validation errors back for another attempt, bounded by `MAX_REPAIR_ATTEMPTS`. |
| `execute` | Re-assert the statement is `SELECT`/`WITH`, run it, and apply the optional row cap. |
| `finalize` | Compose the answer. Every early exit lands here, so a caller always receives a formed response. |

### The RAG pipeline

![The RAG retrieval pipeline](assets/rag-pipeline.svg)

The Knowledge Base compiles into **100 retrieval documents** — one per table, plus
relationships, rules, glossary terms and worked examples. Retrieval combines three
signals:

- **Lexical** — BM25 over table names, column names and declared synonyms, so exact
  identifier matches win.
- **Dense** — cosine similarity over embedded documents.
- **Lexicon** — curated glossary synonyms, which catch business wording that appears
  nowhere in the schema text.

The three are fused with Reciprocal Rank Fusion, which merges *ranks* rather than raw
scores, so an unbounded BM25 score and a 0–1 cosine combine without calibration.

A fourth, structural step follows. Once the anchor table is chosen, every other candidate
is pulled in along **declared join paths**, so the model never has to invent a join. That
is the main reason invented joins do not appear in the output.

### Pluggable retrieval backends

Both halves of the dense signal are chosen by configuration, behind the protocols in
[`retrieval/base.py`](nl2sql/retrieval/base.py):

| Setting | Default | Alternative |
| :-- | :-- | :-- |
| `EMBEDDING_PROVIDER` | `openai` — `text-embedding-3-small` | `tfidf` — local, deterministic, no network |
| `VECTOR_STORE` | `memory` — exhaustive numpy scan | `faiss` — `IndexFlatIP` |

Set them in `.env` and restart; the sidebar shows which backend actually loaded.

**Embeddings are the only choice that changes answers.** `openai` is the default because
it reaches a table whose wording the question never uses — *"show me gear that keeps
breaking"* finds the device and observation documents, where TF-IDF scores zero. TF-IDF is
a lexical statistic, not a semantic one: two phrases sharing no words are unrelated to it.
With no key set, `openai` degrades to `tfidf`, so a clone with no credentials still runs.

**The index does not change answers at all.** Both are exact and return identical
rankings. At 100 documents the numpy scan is the faster of the two, so `memory` is the
default and `faiss` is there for a much larger Knowledge Base.

Every degradation path is covered — no key, a rejected key, a missing `faiss` install, a
mid-session outage. Each falls back to the local pair and logs it rather than failing
start-up.

---

## The Knowledge Base

![The 23-table data model](assets/data-model.svg)

Everything the agent knows about the database lives in YAML under
[`nl2sql/knowledge_base/data/`](nl2sql/knowledge_base/data/). None of it is compiled into
the code.

| Entity | Count |
| :-- | --: |
| Tables | 23 |
| Columns | 170 |
| Relationships | 34 |
| SQL generation rules | 15 |
| Business glossary terms | 16 |
| Worked example queries | 12 |

The 23 tables model a network observability platform across six subject areas:

| Domain | Tables |
| :-- | :-- |
| `observability` | `observations`, `observation_metrics`, `observation_types`, `failure_reasons`, `collectors` |
| `inventory` | `devices`, `interfaces`, `device_models`, `interface_types`, `vendors` |
| `organisation` | `sites`, `regions`, `environments`, `teams`, `users` |
| `alerting` | `alerts`, `alert_rules`, `incidents`, `incident_events` |
| `governance` | `slo_targets`, `maintenance_windows`, `audit_logs` |
| `procurement` | `purchase_orders` |

Each declaration carries the table's business definition, grain, primary key, foreign keys
and synonyms, plus per-column descriptions, roles, units and allowed values. Allowed
values are what turn the word "failed" into `status = 'FAILED'` with the exact casing the
database stores.

### Relationships

Foreign keys describe the *physical* link; relationships add the *intent* — why two tables
are joined and how the join should be written. Six of the major ones:

| Relationship | Join | Why it matters |
| :-- | :-- | :-- |
| `observations` → `interfaces` | INNER | The primary path from a measurement to the port it describes |
| `interfaces` → `devices` | INNER | Rolls port-level telemetry up to the hardware that owns it |
| `observations` → `environments` | INNER | Separates production from staging in every breakdown |
| `observations` → `failure_reasons` | LEFT | Left, because a successful observation has no reason |
| `alerts` → `incidents` | LEFT | Groups related alerts under one investigation |
| `devices` → `sites` → `regions` | INNER | Two hops from hardware to geography |

An invariant test asserts that **every declared foreign key has a matching relationship**.
Without one the join planner cannot traverse the link, so it silently routes around the
gap through a longer path — answering a different question.

---

## Adding a table without touching code

New tables and rules are data, not a release.

**From the browser** — the *Knowledge Base* page has a form for it. Fill in the table, its
columns and an optional link to an existing table; the YAML is generated, validated, and
written only if it is valid. The agent picks it up on the next question. A second tab
edits the YAML files directly, with the same validation.

**From an editor** — drop a file into `nl2sql/knowledge_base/data/tables/`:

```yaml
tables:
  - name: capacity_forecasts
    description: Projected utilisation per interface.
    business_definition: >-
      The projected utilisation of a port over a future window, used for capacity
      planning and upgrade scheduling.
    domain: capacity
    grain: one row per interface per forecast window
    primary_key: [forecast_id]
    synonyms: [capacity forecast, utilisation projection]
    foreign_keys:
      - column: interface_id
        references_table: interfaces
        references_column: interface_id
    columns:
      - name: forecast_id
        data_type: INTEGER
        description: Surrogate key for the forecast.
        role: identifier
        is_primary_key: true
      - name: interface_id
        data_type: INTEGER
        description: The interface the forecast applies to.
        role: identifier
      - name: projected_utilisation_pct
        data_type: REAL
        description: Projected peak utilisation as a percentage.
        role: measure
        unit: percent
```

Declare the matching relationship in `relationships.yaml` so the planner can join it, and
it is queryable. The same pattern applies to `sql_rules.yaml`, `business_glossary.yaml`
and `example_queries.yaml` — each is picked up by filename, with no registration step.

The other extension points need no agent change either: SQL dialects
([`dialects.py`](nl2sql/dialects.py)), model providers
([`llm/factory.py`](nl2sql/llm/factory.py)) and retrieval backends
([`retrieval/base.py`](nl2sql/retrieval/base.py)) are each resolved through a registry.

### Using a different database

SQLite backs the bundled demo, but **Postgres and MySQL** are supported too — for query
generation, for validation, and for materialising the schema.

```bash
venv/bin/python -m pip install -e ".[postgres]"
```

```bash
DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/observability
SQL_DIALECT=postgres
```

`make seed` builds and populates whichever database `DATABASE_URL` points at, so a fresh
Postgres instance is one command away from holding the same demo dataset.

Or switch without restarting, from the **Database** page. The connection is tested before
anything changes, and a failed switch leaves the current one untouched. Passwords are
masked wherever a URL is displayed or logged.

One Knowledge Base declaration renders per engine, so there is no second schema definition
to keep in step:

| Declared | SQLite | Postgres | MySQL |
| :-- | :-- | :-- | :-- |
| `VARCHAR(64)` | `TEXT` | `VARCHAR(64)` | `VARCHAR(64)` |
| `BOOLEAN` | `INTEGER` | `BOOLEAN` | `TINYINT(1)` |
| `TIMESTAMP` | `TEXT` | `TIMESTAMP` | `DATETIME` |
| `DECIMAL(5,2)` | `REAL` | `DECIMAL(5,2)` | `DECIMAL(5,2)` |

Adding an engine is one entry in [`dialects.py`](nl2sql/dialects.py) — a time-arithmetic
template and a type map. Nothing in the planner, the seeder or the interface changes.

---

## Interfaces

### Web interface

A Streamlit dashboard on <http://127.0.0.1:8501>, with the engine, retrieval settings and
environment in the sidebar. Five pages:

- **Ask** — the question box, worked examples grouped by answer type, and the answer with
  its SQL, row counts, timings, the tables that were read, and a CSV download.
- **Agent Execution** — the workflow run, step by step: what each node decided, where the
  time went, what retrieval selected, what validation checked.
- **Knowledge Base** — browse tables, relationships, rules and glossary; add a table from
  a form or edit the YAML directly.
- **Compare Engines** — one question through every available engine side by side.
- **Database** — connect to a different database, build the declared schema into it, and
  count what is there.

### Command line

```bash
venv/bin/python -m nl2sql.cli ask "List interfaces with the highest failure count."
```

`inspect` prints Knowledge Base coverage, `seed` rebuilds the demo database, and `demo`
runs the bundled examples end to end (`make demo`).

### HTTP API

| Route | Purpose |
| :-- | :-- |
| `GET /health` | Service status and Knowledge Base coverage |
| `POST /ask` | Question in; validated SQL and rows out |
| `GET /schema/tables` | Every declared table |
| `GET /schema/tables/{name}` | One table in full |
| `GET /schema/relationships` | Every declared join |

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "Display observations grouped by environment.", "include_trace": true}'
```

Interactive docs are at <http://127.0.0.1:8000/docs>.

An unanswerable question returns HTTP 200 with `succeeded: false` and an explanation —
that is a normal outcome, not a client error. A query the database rejects also reports
`succeeded: false`, with the reason in `errors`.

### Python

```python
from nl2sql.pipeline import NL2SQLPipeline

pipeline = NL2SQLPipeline.create()
answer = pipeline.answer("Show all failed observations in the last 24 hours.")

print(answer.sql, answer.row_count, answer.succeeded)
```

---

## Configuration

Every setting has a working default; the system runs with no `.env` at all. Copy
`.env.example` to `.env` to change any of it.

| Setting | Default | Purpose |
| :-- | :-- | :-- |
| `LLM_PROVIDER` | `auto` | `auto`, `anthropic`, `openai` or `deterministic` |
| `LLM_MODEL` | per provider | Rejected at start-up if it belongs to another provider |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | unset | Credentials; absent means the planner answers |
| `DATABASE_URL` | `sqlite:///data/observability.db` | Any SQLAlchemy URL |
| `SQL_DIALECT` | `sqlite` | Dialect the generated SQL targets |
| `EXECUTE_QUERIES` | `true` | Set false to generate SQL without running it |
| `MAX_RESULT_ROWS` | unset | Optional cap on rows returned; unset returns every row |
| `RETRIEVAL_TOP_K` | `8` | Documents retrieved per question |
| `LEXICAL_WEIGHT` | `0.5` | Keyword vs dense blend |
| `EMBEDDING_PROVIDER` | `openai` | `tfidf` or `openai`; falls back to tfidf without a key |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Used when the provider is `openai` |
| `VECTOR_STORE` | `memory` | `memory` or `faiss`; both exact, identical rankings |
| `MAX_REPAIR_ATTEMPTS` | `2` | Bound on the repair loop |
| `LANGSMITH_TRACING` | `false` | Record runs in LangSmith (see below) |
| `LANGSMITH_API_KEY` | unset | Required when tracing is on; without it tracing stays off |
| `LANGSMITH_PROJECT` | `nl2sql-rag` | Project runs are grouped under |
| `FRONTEND_HOST` / `FRONTEND_PORT` | `127.0.0.1` / `8501` | Web interface bind address |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

> `.env` is git-ignored. Keep real keys out of anything you share.

### Tracing runs with LangSmith

Off by default — the system runs with no accounts of any kind. Turn it on to record every
run:

```bash
LANGSMITH_TRACING=true LANGSMITH_API_KEY=lsv2_... make start
```

Each question then appears as a trace covering all seven nodes, with the retrieved
context, the generated SQL, any repair attempt and the timings. `/health` reports
`tracing_enabled` so you can confirm it is on without guessing.

Every trace carries the settings that could have changed the answer — provider, model,
`lexical_weight`, `retrieval_top_k`, `max_repair_attempts` — so runs can be filtered and
grouped on them.

Payloads are summarised, not raw. The full workflow state is ~400 KB per question across
eleven spans, which saturates the uploader; summarised it is ~20 KB. Result rows are
dropped outright — they are business data, and telemetry is the wrong place for them.

---

## Testing

```bash
make test
```

```bash
make coverage
```

**411 tests**, covering the Knowledge Base, retrieval, the workflow, both generators,
validation, the API, the CLI surface and the web interface callbacks.

Three suites are worth calling out, because they guard the claims this project makes:

- [`tests/test_semantic_correctness.py`](tests/test_semantic_correctness.py) — queries
  that parse, validate and run but answer the *wrong question*: grouping by another
  table's column, widening the grain with an unrequested dimension, ANDing mutually
  exclusive filters down to zero rows, or reporting a database error as an empty result.
  The validator cannot catch any of these, so they are pinned down explicitly.
- [`tests/test_retrieval_backends.py`](tests/test_retrieval_backends.py) — that the dense
  half of retrieval stays swappable: a substituted index is honoured, and one that cannot
  serve the corpus degrades the ranking instead of stopping the system.
- [`tests/test_extensibility_invariants.py`](tests/test_extensibility_invariants.py) —
  turns "a new table is a data-only change" into a failing build the moment it stops being
  true: every foreign key must have a relationship, every declared table must have demo
  data, and generated rows must only use declared columns.

---

## Project structure

```
nl2sql/
├── analysis/          Question analyzer: intent, filters, time windows, groupings
├── database/          Engine, schema builder, seeding, demo data, read-only executor
├── generation/        SQL generation
│   ├── deterministic/   Rule-based planner — the no-API-key path and the fallback
│   └── llm_generator.py Model-backed generation with the planner behind it
├── graph/             LangGraph state, nodes and workflow wiring
├── knowledge_base/    Models, loader, registry, browser-side authoring
│   └── data/            The Knowledge Base itself — YAML, no code
├── llm/               Provider clients and the factory that resolves them
├── prompts/           Prompt templates for generation and repair
├── retrieval/         Hybrid retriever, embeddings, keyword index, context builder
├── validation/        SQL validator and guardrails
├── ui/                Streamlit interface
│   ├── app.py           Entry point: sidebar and navigation
│   ├── state.py         Cached engines, Knowledge Base editor, last run
│   ├── components.py    Shared renderers
│   └── views/           One render(workspace) per page
├── api.py             FastAPI application
├── cli.py             Typer command line
├── config.py          Settings
├── dialects.py        Per-engine time arithmetic and column types
└── pipeline.py        Facade over the compiled workflow
```

---

## Design decisions

**The Knowledge Base is data, not code.** Tables, rules, joins and definitions are YAML
loaded at start-up, so adding a table is a file rather than a release.

**Two generators behind one interface.** The deterministic planner runs with no
credentials and stands behind the model as a fallback, so a provider outage degrades
answer quality instead of stopping the service.

**Validation reads the same Knowledge Base as generation.** An invented column fails
against the source of truth that produced the prompt, which is what makes the repair loop
converge.

**The workflow is a declared graph.** Bounded repair, explicit routing, recorded trace.

**Ambiguity is resolved against the question, not the file order.** `status` is declared by
several tables; ranking candidates against the question stops the answer depending on
which YAML file loaded first.

---

## Known limitations

- The deterministic planner covers list, aggregate and ranking shapes. Window functions,
  set operations and correlated subqueries are the model-backed path's job.
- Only values declared in the Knowledge Base become filters. The planner will not turn an
  arbitrary string — a hostname, a person's name — into a `WHERE` clause; the model-backed
  path handles those better.
- `faiss` is exact (`IndexFlatIP`). A Knowledge Base large enough to need an approximate
  index would want one more class behind the same protocol.
- Demo data is generated from a fixed seed for reproducibility, so row counts are stable
  but not meant to resemble real traffic volumes.

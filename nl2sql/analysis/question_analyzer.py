"""Structured analysis of a natural-language question."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from nl2sql.knowledge_base.registry import (
    NAMED_TABLE_EVIDENCE,
    DimensionMatch,
    EnumValueMatch,
    KnowledgeBaseRegistry,
)
from nl2sql.logging_config import get_logger
from nl2sql.retrieval.text import extract_ngrams, normalise

logger = get_logger(__name__)


class QueryIntent(str, Enum):
    """The shape of result the question is asking for."""

    LIST = "list"
    """Return individual rows, e.g. "show all failed observations"."""

    AGGREGATE = "aggregate"
    """Return grouped totals, e.g. "observations grouped by environment"."""

    RANKING = "ranking"
    """Return an ordered leaderboard, e.g. "interfaces with the most failures"."""

    UNSUPPORTED = "unsupported"
    """The request cannot be served, e.g. it asks to modify data."""


class TimeUnit(str, Enum):
    """Units a relative time window can be expressed in."""

    MINUTE = "minutes"
    HOUR = "hours"
    DAY = "days"
    WEEK = "weeks"
    MONTH = "months"
    YEAR = "years"


@dataclass(frozen=True, slots=True)
class TimeWindow:
    """A relative time window, expressed independently of any SQL dialect."""

    quantity: int
    unit: TimeUnit
    phrase: str
    include_upper_bound: bool = False
    """True for closed windows such as "yesterday", which exclude today."""

    def describe(self) -> str:
        """Human readable summary used in logs and explanations."""
        return f"last {self.quantity} {self.unit.value}"


@dataclass(frozen=True, slots=True)
class ValueFilter:
    """An equality filter on an enumerated column, resolved from the KB."""

    table: str
    column: str
    value: str
    phrase: str


@dataclass(frozen=True, slots=True)
class GroupingDimension:
    """A dimension the caller asked to group or break results down by."""

    table: str
    column: str
    phrase: str


@dataclass(frozen=True, slots=True)
class MetricRequest:
    """A named metric the question asked for, resolved from the glossary."""

    name: str
    expression: str
    alias: str
    phrase: str


@dataclass(slots=True)
class QuestionAnalysis:
    """Everything the analyzer could determine about a question."""

    question: str
    intent: QueryIntent = QueryIntent.LIST
    mentioned_tables: list[str] = field(default_factory=list)
    named_tables: list[str] = field(default_factory=list)
    """The subset of :attr:`mentioned_tables` the question named outright."""

    value_filters: list[ValueFilter] = field(default_factory=list)
    groupings: list[GroupingDimension] = field(default_factory=list)
    metric: MetricRequest | None = None
    time_window: TimeWindow | None = None
    row_limit: int | None = None
    descending: bool = True
    wants_count: bool = False
    rejection_reason: str | None = None

    @property
    def is_supported(self) -> bool:
        """True when the question can be turned into a read-only query."""
        return self.intent is not QueryIntent.UNSUPPORTED

    @property
    def is_grounded(self) -> bool:
        """True when some part of the question resolved to the Knowledge Base."""
        return bool(
            self.named_tables
            or self.value_filters
            or self.groupings
            or self.metric is not None
        )

    def subject_tables(self) -> set[str]:
        """The tables the question is actually about.

        Narrower than :attr:`mentioned_tables`: a table qualifies by carrying a
        filter or grouping the question asked for, or by being the strongest name.
        """
        tables = set(self.named_tables)
        tables.update(item.table for item in self.value_filters)
        tables.update(item.table for item in self.groupings)
        if not tables and self.mentioned_tables:
            tables.add(self.mentioned_tables[0])
        return tables

    def to_prompt_hints(self) -> str:
        """Render the analysis as explicit hints for the generation prompt."""
        hints: list[str] = [f"- Detected intent: {self.intent.value}"]

        if self.time_window:
            hints.append(
                f"- Time window requested: {self.time_window.describe()} "
                f"(from the phrase {self.time_window.phrase!r})"
            )
        for value_filter in self.value_filters:
            hints.append(
                f"- Filter on {value_filter.table}.{value_filter.column} = "
                f"'{value_filter.value}' (from {value_filter.phrase!r})"
            )
        for grouping in self.groupings:
            hints.append(f"- Group by {grouping.table}.{grouping.column}")
        if self.metric:
            hints.append(
                f"- Requested metric: {self.metric.name} -> "
                f"{self.metric.expression} AS {self.metric.alias}"
            )
        if self.row_limit:
            hints.append(f"- Explicit row limit requested: {self.row_limit}")
        if self.wants_count:
            hints.append("- The question asks for a count")

        return "\n".join(hints)


# --- Vocabulary ---------------------------------------------------------------

_MUTATION_TERMS = frozenset(
    {
        "insert", "update", "delete", "drop", "truncate", "alter", "create",
        "grant", "revoke", "rename", "modify", "remove", "wipe", "purge",
    }
)

_AGGREGATION_TERMS = frozenset(
    {
        "count", "total", "sum", "average", "avg", "mean", "number",
        "how many", "breakdown", "grouped", "group", "per", "distribution",
        "aggregate", "rate", "percentage", "percent",
    }
)

_RANKING_TERMS = frozenset(
    {
        "top", "highest", "most", "largest", "biggest", "worst", "greatest",
        "lowest", "least", "smallest", "best", "ranked", "ranking", "leaderboard",
    }
)

_ASCENDING_TERMS = frozenset({"lowest", "least", "smallest", "fewest", "best"})

_COUNT_TERMS = frozenset({"count", "how many", "number of", "how much"})

_UNIT_LOOKUP: dict[str, TimeUnit] = {
    "minute": TimeUnit.MINUTE,
    "minutes": TimeUnit.MINUTE,
    "min": TimeUnit.MINUTE,
    "mins": TimeUnit.MINUTE,
    "hour": TimeUnit.HOUR,
    "hours": TimeUnit.HOUR,
    "hr": TimeUnit.HOUR,
    "hrs": TimeUnit.HOUR,
    "day": TimeUnit.DAY,
    "days": TimeUnit.DAY,
    "week": TimeUnit.WEEK,
    "weeks": TimeUnit.WEEK,
    "month": TimeUnit.MONTH,
    "months": TimeUnit.MONTH,
    "year": TimeUnit.YEAR,
    "years": TimeUnit.YEAR,
}

_NUMBER_WORDS: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12,
    "twenty": 20, "thirty": 30, "sixty": 60,
}

# "in the last 24 hours", "over the past 7 days", "within the previous 2 weeks"
_RELATIVE_WINDOW_PATTERN = re.compile(
    r"\b(?:last|past|previous|latest|recent)\s+"
    r"(?:(\d+)|(" + "|".join(_NUMBER_WORDS) + r"))?\s*"
    r"(" + "|".join(_UNIT_LOOKUP) + r")\b"
)

# "top 5", "first 20 rows", "limit 50"
_ROW_LIMIT_PATTERN = re.compile(r"\b(?:top|first|limit)\s+(\d+)\b")


# "grouped by environment", "per device", "for each site", "broken down by team"
_GROUPING_PATTERN = re.compile(
    r"\b(?:grouped\s+by|group\s+by|broken\s+down\s+by|broken\s+out\s+by|"
    r"split\s+by|by|per|for\s+each|across\s+each|across|each)\s+"
    r"([a-z][a-z0-9_ ]{2,40})"
)

# Words that only ever introduce a sort order, never a grouping dimension.
_GROUPING_BLOCKLIST = frozenset(
    {"the", "their", "its", "which", "what", "descending", "ascending"}
)

# Only in these phrases is "order" a sort instruction rather than a column name.
_SORT_ORDER_PHRASES = ("descending order", "ascending order", "order by")

# Words that are an enum value in isolation but a verb particle after these verbs.
_PARTICLE_VERBS: dict[str, frozenset[str]] = {
    "down": frozenset({"break", "breaks", "breaking", "broke", "broken", "drill"}),
    "up": frozenset({"sum", "sums", "summing", "add", "adds", "adding", "roll"}),
}

_NAMED_WINDOWS: dict[str, TimeWindow] = {
    "yesterday": TimeWindow(1, TimeUnit.DAY, "yesterday", include_upper_bound=True),
    "today": TimeWindow(1, TimeUnit.DAY, "today"),
    "this week": TimeWindow(7, TimeUnit.DAY, "this week"),
    "this month": TimeWindow(1, TimeUnit.MONTH, "this month"),
    "last night": TimeWindow(12, TimeUnit.HOUR, "last night"),
    "right now": TimeWindow(1, TimeUnit.HOUR, "right now"),
    "currently": TimeWindow(1, TimeUnit.HOUR, "currently"),
}


class QuestionAnalyzer:
    """Extracts structured query intent from a natural-language question."""

    def __init__(self, registry: KnowledgeBaseRegistry) -> None:
        self._registry = registry

    def analyze(self, question: str) -> QuestionAnalysis:
        """Analyse ``question`` and return everything that could be determined."""
        text = normalise(question)
        ngrams = extract_ngrams(question, max_size=3)

        rejection = self._detect_unsupported_request(text)
        if rejection is not None:
            return QuestionAnalysis(
                question=question,
                intent=QueryIntent.UNSUPPORTED,
                rejection_reason=rejection,
            )

        analysis = QuestionAnalysis(question=question)
        analysis.mentioned_tables, analysis.named_tables = self._resolve_tables(ngrams)
        analysis.metric = self._resolve_metric(ngrams)
        analysis.value_filters = self._resolve_value_filters(
            ngrams, analysis.mentioned_tables, analysis.metric, text
        )
        analysis.time_window = self._parse_time_window(text)
        analysis.row_limit = self._parse_row_limit(text)
        analysis.groupings = self._parse_groupings(question, analysis.mentioned_tables)
        analysis.wants_count = any(term in text for term in _COUNT_TERMS)

        if not analysis.is_grounded:
            # Retrieval has no similarity floor, so an off-topic question still
            # reaches a table and would otherwise be answered confidently.
            analysis.intent = QueryIntent.UNSUPPORTED
            analysis.rejection_reason = (
                "This question does not refer to anything in the Knowledge Base. "
                "Try naming a subject from the schema, such as observations, "
                "devices or alerts."
            )
            logger.info("Declining ungrounded question: %r", question)
            return analysis

        analysis.descending = not any(term in text.split() for term in _ASCENDING_TERMS)
        analysis.intent = self._classify_intent(text, analysis)

        logger.debug(
            "Analysed %r -> intent=%s tables=%s filters=%d groupings=%d window=%s",
            question,
            analysis.intent.value,
            analysis.mentioned_tables,
            len(analysis.value_filters),
            len(analysis.groupings),
            analysis.time_window.describe() if analysis.time_window else None,
        )
        return analysis

    # -- Individual extractors ------------------------------------------------

    @staticmethod
    def _detect_unsupported_request(text: str) -> str | None:
        """Return a rejection reason when the question asks to modify data."""
        words = set(text.split())
        mutations = words & _MUTATION_TERMS
        if mutations:
            return (
                f"This system answers read-only questions. The request appears to ask "
                f"for a data modification ({', '.join(sorted(mutations))})."
            )
        return None

    def _naming_phrases(self, ngrams: set[str]) -> list[set[str]]:
        """Return the word sets of phrases that named a table outright."""
        return [
            set(ngram.split())
            for ngram in ngrams
            if any(
                match.score >= NAMED_TABLE_EVIDENCE
                for match in self._registry.resolve_tables_for_phrase(ngram)
            )
            and " " in ngram
        ]

    def _resolve_tables(self, ngrams: set[str]) -> tuple[list[str], list[str]]:
        """Map phrases in the question onto declared table names, best first.

        Returns:
            Every matched table ranked best first, and the subset that some phrase
            named outright.
        """
        scores: dict[str, float] = {}
        named: set[str] = set()

        # Sorted for determinism: ranking must not depend on set iteration order.
        for ngram in sorted(ngrams):
            specificity = len(ngram.split())
            for match in self._registry.resolve_tables_for_phrase(ngram):
                scores[match.table] = max(
                    scores.get(match.table, 0.0), match.score * specificity
                )
                if match.score >= NAMED_TABLE_EVIDENCE:
                    named.add(match.table)

        ranked = [
            name
            for name, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        ]
        return ranked, [name for name in ranked if name in named]

    def _resolve_metric(self, ngrams: set[str]) -> MetricRequest | None:
        """Find the glossary metric the question asks for, most specific phrase first."""
        best: MetricRequest | None = None
        best_specificity = 0

        for ngram in sorted(ngrams):
            term = self._registry.resolve_metric_for_phrase(ngram)
            if term is None:
                continue

            specificity = len(ngram.split())
            if specificity > best_specificity:
                best_specificity = specificity
                best = MetricRequest(
                    name=term.term,
                    expression=" ".join(term.metric_expression.split()),
                    alias=term.metric_alias,
                    phrase=ngram,
                )

        return best

    def _resolve_value_filters(
        self,
        ngrams: set[str],
        ranked_tables: list[str],
        metric: MetricRequest | None,
        text: str,
    ) -> list[ValueFilter]:
        """Map phrases onto enumerated column values declared in the KB.

        A declared synonym is trusted on its own; a match on the bare stored literal
        counts only when the question also names the owning table, since enum values
        are ordinary English words. One phrase yields at most one filter, so two
        mutually exclusive predicates are never ANDed together.
        """
        filters: list[ValueFilter] = []
        seen: set[tuple[str, str, str]] = set()
        mentioned = set(ranked_tables)
        metric_words = set(metric.phrase.split()) if metric else set()
        naming_phrases = self._naming_phrases(ngrams)

        for ngram in sorted(ngrams):
            words = set(ngram.split())
            if metric_words and words <= metric_words:
                continue
            # A word already spent naming a table is not also a value to filter on.
            if any(words < phrase for phrase in naming_phrases):
                continue
            if self._is_verb_particle(ngram, text):
                continue

            candidates = []
            for match in self._registry.resolve_enum_value(ngram):
                if not match.from_synonym and match.table not in mentioned:
                    logger.debug(
                        "Ignoring %r as a value of %s.%s: it matches only the stored "
                        "literal and the question does not refer to that table",
                        ngram,
                        match.table,
                        match.column,
                    )
                    continue
                candidates.append(match)

            if not candidates:
                continue

            best = min(
                candidates,
                key=lambda match: self._filter_rank(match, ranked_tables, text),
            )
            if len(candidates) > 1:
                logger.debug(
                    "Phrase %r is an allowed value of %d columns; filtering on %s.%s",
                    ngram,
                    len(candidates),
                    best.table,
                    best.column,
                )

            key = (best.table, best.column, best.value)
            if key in seen:
                continue
            seen.add(key)
            filters.append(
                ValueFilter(
                    table=best.table,
                    column=best.column,
                    value=best.value,
                    phrase=ngram,
                )
            )

        return filters

    @staticmethod
    def _is_verb_particle(ngram: str, text: str) -> bool:
        """True when ``ngram`` completes a phrasal verb rather than naming a value."""
        verbs = _PARTICLE_VERBS.get(ngram)
        if verbs is None:
            return False

        return any(f"{verb} {ngram}" in text for verb in verbs)

    @staticmethod
    def _filter_rank(
        match: EnumValueMatch, ranked_tables: list[str], text: str
    ) -> tuple[int, int, int, str, str]:
        """Score a candidate value filter; lower sorts first.

        Ranked by whether the question names the column, then whether the match came
        from curated wording, then how close the table is to the question's subject.
        """
        named_rank = 0 if match.column.lower() in text else 1
        curated_rank = 0 if match.from_synonym else 1

        try:
            subject_rank = ranked_tables.index(match.table)
        except ValueError:
            subject_rank = len(ranked_tables)

        return (named_rank, curated_rank, subject_rank, match.table, match.column)

    @staticmethod
    def _parse_time_window(text: str) -> TimeWindow | None:
        """Extract a relative time window, if the question contains one."""
        for phrase, window in _NAMED_WINDOWS.items():
            if phrase in text:
                return window

        match = _RELATIVE_WINDOW_PATTERN.search(text)
        if match is None:
            return None

        digits, number_word, unit_token = match.groups()
        if digits:
            quantity = int(digits)
        elif number_word:
            quantity = _NUMBER_WORDS[number_word]
        else:
            quantity = 1

        return TimeWindow(
            quantity=quantity,
            unit=_UNIT_LOOKUP[unit_token],
            phrase=match.group(0),
        )

    @staticmethod
    def _parse_row_limit(text: str) -> int | None:
        """Extract an explicit row limit such as "top 5"."""
        match = _ROW_LIMIT_PATTERN.search(text)
        return int(match.group(1)) if match else None

    def _parse_groupings(
        self, question: str, ranked_tables: list[str]
    ) -> list[GroupingDimension]:
        """Extract the dimensions the question asks to break results down by."""
        groupings: list[GroupingDimension] = []
        seen: set[tuple[str, str]] = set()

        for match in _GROUPING_PATTERN.finditer(normalise(question)):
            phrase = match.group(1).strip()

            # In "top 5 sites by failure count", "by <measure>" names the sort key.
            if set(phrase.split()) & _AGGREGATION_TERMS:
                continue

            dimension = self._resolve_dimension(phrase, ranked_tables)
            if dimension is None:
                continue
            key = (dimension.table, dimension.column)
            if key not in seen:
                seen.add(key)
                groupings.append(dimension)

        return groupings

    def _resolve_dimension(
        self, phrase: str, ranked_tables: list[str]
    ) -> GroupingDimension | None:
        """Resolve a phrase such as "environment" to a concrete groupable column.

        The phrase is tried longest-first so "failure reason" resolves before the bare
        word "reason". Candidates are ranked against the subject of the question rather
        than taken in KB declaration order.
        """
        blocked = set(_GROUPING_BLOCKLIST)
        if any(marker in phrase for marker in _SORT_ORDER_PHRASES):
            blocked.add("order")

        words = [word for word in phrase.split() if word not in blocked]
        if not words:
            return None

        for length in range(min(len(words), 3), 0, -1):
            candidate = " ".join(words[:length])
            matches = self._registry.resolve_dimension_for_phrase(candidate)

            # The lexicon misses prefixed columns (window_status against "status"),
            # so try the subject table before settling for another table's column.
            subject_match = self._subject_column(candidate, ranked_tables, matches)
            if subject_match is not None:
                return GroupingDimension(
                    table=subject_match[0], column=subject_match[1], phrase=candidate
                )

            if not matches:
                continue

            best = min(
                matches,
                key=lambda match: self._dimension_rank(match, candidate, ranked_tables),
            )
            if len(matches) > 1:
                logger.debug(
                    "Grouping phrase %r matches %d columns; chose %s.%s",
                    candidate,
                    len(matches),
                    best.table,
                    best.column,
                )
            return GroupingDimension(
                table=best.table, column=best.column, phrase=candidate
            )

        return None

    def _subject_column(
        self,
        candidate: str,
        ranked_tables: list[str],
        matches: list[DimensionMatch],
    ) -> tuple[str, str] | None:
        """Find ``candidate`` as a suffixed column on the table being asked about.

        Returns nothing when ``matches`` already covers the subject table, so a column
        the lexicon knows about always wins.
        """
        subject = ranked_tables[0] if ranked_tables else None
        if subject is None or any(match.table == subject for match in matches):
            return None

        table = self._registry.get_table(subject)
        if table is None:
            return None

        wanted = candidate.replace(" ", "_")
        for column in table.columns:
            if column.name.lower().endswith(f"_{wanted}"):
                return (subject, column.name)

        return None

    @staticmethod
    def _dimension_rank(
        match: DimensionMatch, candidate: str, ranked_tables: list[str]
    ) -> tuple[int, int, str, str]:
        """Score a grouping candidate; lower sorts first.

        Ordered by subject table, then how literally the column is named; the table and
        column names close the tuple so ties resolve deterministically.
        """
        try:
            subject_rank = ranked_tables.index(match.table)
        except ValueError:
            subject_rank = len(ranked_tables)

        column = match.column.lower()
        wanted = candidate.replace(" ", "_")
        if column == wanted:
            name_rank = 0
        elif column.endswith(f"_{wanted}"):
            name_rank = 1
        else:
            name_rank = 2

        return (subject_rank, name_rank, match.table, match.column)

    @staticmethod
    def _classify_intent(text: str, analysis: QuestionAnalysis) -> QueryIntent:
        """Decide whether the answer is a row list, an aggregate or a ranking."""
        words = set(text.split())

        if words & _RANKING_TERMS:
            return QueryIntent.RANKING

        has_aggregation_word = bool(words & _AGGREGATION_TERMS) or any(
            term in text for term in _AGGREGATION_TERMS if " " in term
        )
        if analysis.groupings or has_aggregation_word:
            return QueryIntent.AGGREGATE

        return QueryIntent.LIST

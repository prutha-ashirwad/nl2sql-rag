"""Text normalisation shared by the lexical and vector retrievers."""

from __future__ import annotations

import re

_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")

# Function words plus terms appearing in nearly every schema document.
STOP_WORDS = frozenset(
    {
        "a", "an", "and", "any", "are", "as", "at", "be", "by", "can", "do", "does",
        "each", "for", "from", "get", "give", "has", "have", "how", "i", "in", "into",
        "is", "it", "its", "list", "me", "of", "on", "or", "our", "please", "show",
        "that", "the", "their", "them", "there", "these", "this", "those", "to", "us",
        "was", "we", "were", "what", "when", "where", "which", "who", "will", "with",
        "column", "table", "value",
    }
)

# Lightweight rules: a full stemmer would collapse distinct schema identifiers.
_SUFFIX_RULES: tuple[tuple[str, str], ...] = (
    ("ies", "y"),
    ("sses", "ss"),
    ("ches", "ch"),
    ("shes", "sh"),
    ("xes", "x"),
    ("s", ""),
)

_MIN_STEM_LENGTH = 4


def normalise(text: str) -> str:
    """Lowercase ``text`` and collapse punctuation into single spaces."""
    return re.sub(r"[^a-z0-9_]+", " ", text.lower()).strip()


def stem(token: str) -> str:
    """Apply conservative plural stripping so 'interfaces' matches 'interface'."""
    if len(token) <= _MIN_STEM_LENGTH:
        return token
    for suffix, replacement in _SUFFIX_RULES:
        if token.endswith(suffix):
            stemmed = token[: -len(suffix)] + replacement
            if len(stemmed) >= _MIN_STEM_LENGTH - 1:
                return stemmed
    return token


def tokenize(text: str, *, drop_stop_words: bool = True) -> list[str]:
    """Split ``text`` into normalised, stemmed tokens.

    Underscored identifiers are emitted whole and split, so ``observed_at`` matches
    a query mentioning either "observed" or "observed_at".
    """
    tokens: list[str] = []

    for raw in _TOKEN_PATTERN.findall(text.lower()):
        if drop_stop_words and raw in STOP_WORDS:
            continue

        tokens.append(stem(raw))

        if "_" in raw:
            for part in raw.split("_"):
                if part and not (drop_stop_words and part in STOP_WORDS):
                    tokens.append(stem(part))

    return tokens


def extract_ngrams(text: str, max_size: int = 3) -> set[str]:
    """Return every contiguous word n-gram of length 1..``max_size``."""
    words = normalise(text).split()
    ngrams: set[str] = set()

    for size in range(1, max_size + 1):
        for start in range(len(words) - size + 1):
            ngrams.add(" ".join(words[start : start + size]))

    return ngrams

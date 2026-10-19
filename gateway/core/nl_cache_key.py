"""
Literal signature for the semantic NL->SQL cache.

Embedding similarity is the wrong test for "same SQL": "revenue in 2024" and
"revenue in 2023", "top customers" and "bottom customers", "orders from
Germany" and "orders from France" all embed almost identically but need
different SQL. So a cache hit additionally requires the two questions to
carry the same *literals* — numbers, quoted strings, capitalized proper
nouns, sort direction and negation. Paraphrases ("max" vs "maximum",
"customer with most orders" vs "which customer ordered the most") still
normalize to the same signature and keep hitting.
"""
from __future__ import annotations

import hashlib
import re

_DIRECTION = {
    "max": {"max", "maximum", "highest", "most", "top", "largest", "biggest", "greatest", "best"},
    "min": {"min", "minimum", "lowest", "least", "bottom", "smallest", "fewest", "worst"},
}
_NEGATION = {"not", "no", "without", "excluding", "except", "never", "non"}
_COMPARATORS = {
    "gt": {"more", "greater", "above", "over", "after", "exceeding"},
    "lt": {"less", "fewer", "below", "under", "before"},
}
# Words that start questions or are capitalized for reasons other than being a
# value the SQL filters on.
_NOT_PROPER = {
    "what", "which", "who", "whom", "how", "when", "where", "why", "show", "list", "give", "find",
    "get", "tell", "count", "the", "a", "an", "i", "is", "are", "do", "does", "for", "in", "of",
}
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")
_QUOTED_RE = re.compile(r"""["']([^"']+)["']""")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]*")


def literal_signature(question: str) -> str:
    """Short stable hash of the question's filter-relevant literals."""
    parts: set[str] = set()
    parts.update(f"n:{n.replace(',', '')}" for n in _NUMBER_RE.findall(question))
    parts.update(f"q:{q.strip().lower()}" for q in _QUOTED_RE.findall(question))

    words = _WORD_RE.findall(question)
    for i, w in enumerate(words):
        lw = w.lower()
        for label, vocab in _DIRECTION.items():
            if lw in vocab:
                parts.add(f"d:{label}")
        for label, vocab in _COMPARATORS.items():
            if lw in vocab:
                parts.add(f"c:{label}")
        if lw in _NEGATION or lw.endswith("n't"):
            parts.add("neg")
        if w[0].isupper() and lw not in _NOT_PROPER and not (i == 0):
            parts.add(f"p:{lw}")

    if not parts:
        return "none"
    return hashlib.sha1("|".join(sorted(parts)).encode()).hexdigest()[:12]

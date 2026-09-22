from __future__ import annotations

import re
from typing import Any

STOPWORDS = {
    "the", "a", "an", "is", "my", "i", "to", "for", "of", "on", "do", "you", "it",
    "can", "what", "when", "how", "and", "in", "me", "this", "that", "with", "please",
}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]+", text.lower()) if t not in STOPWORDS and len(t) > 2}


def search(kb: dict[str, Any], query: str, limit: int = 2) -> list[dict[str, Any]]:
    """Keyword-overlap retrieval over the mock knowledge base."""
    query_tokens = _tokens(query)
    lowered = query.lower()
    scored: list[tuple[float, dict[str, Any]]] = []
    for article in kb["articles"]:
        score = 0.0
        for keyword in article["keywords"]:
            if keyword in lowered:
                score += 2.0
        score += len(query_tokens & _tokens(article["title"])) * 1.0
        score += len(query_tokens & _tokens(article["body"])) * 0.25
        if score > 0:
            scored.append((score, article))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [article for _, article in scored[:limit]]

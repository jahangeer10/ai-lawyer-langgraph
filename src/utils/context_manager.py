"""Utility functions for managing conversation context."""
from typing import List, Any


def count_tokens(messages: List[Any]) -> int:
    """Naive token count based on whitespace splitting."""
    return sum(len(getattr(m, "content", "").split()) for m in messages)


def trim_messages(messages: List[Any], max_tokens: int) -> List[Any]:
    """Trim messages so that total tokens do not exceed max_tokens."""
    trimmed: List[Any] = []
    total = 0
    for msg in reversed(messages):
        tokens = len(getattr(msg, "content", "").split())
        if total + tokens > max_tokens:
            break
        trimmed.append(msg)
        total += tokens
    return list(reversed(trimmed))


def format_legal_response(text: str, citations: List[str], confidence: float) -> str:
    """Format the response for API consumers."""
    citation_text = "; ".join(citations) if citations else "No citations"
    return f"{text}\n\nCitations: {citation_text}\nConfidence: {confidence:.2f}"

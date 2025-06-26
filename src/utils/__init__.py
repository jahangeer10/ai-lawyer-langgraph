"""Utility modules for AI Lawyer Agent."""

from .legal_knowledge import LegalKnowledgeBase
from .context_manager import count_tokens, trim_messages, format_legal_response
from .document_processor import DocumentProcessor

__all__ = [
    "LegalKnowledgeBase",
    "count_tokens",
    "trim_messages",
    "format_legal_response",
    "DocumentProcessor",
]

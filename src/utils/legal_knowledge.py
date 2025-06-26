"""Simple knowledge base for legal references."""
from typing import Optional

class LegalKnowledgeBase:
    """A very small in-memory knowledge base for demo purposes."""

    def __init__(self) -> None:
        # Example constitutional articles and IPC sections
        self.articles = {
            "article 19": "Protection of certain rights regarding freedom of speech and expression",
            "article 21": "Protection of life and personal liberty",
        }
        self.sections = {
            "ipc 420": "Cheating and dishonestly inducing delivery of property",
            "ipc 376": "Punishment for rape",
        }

    def search(self, term: str) -> Optional[str]:
        """Return a short description for a matching article or section."""
        key = term.lower()
        if key in self.articles:
            return f"{term.upper()}: {self.articles[key]}"
        if key in self.sections:
            return f"{term.upper()}: {self.sections[key]}"
        return None

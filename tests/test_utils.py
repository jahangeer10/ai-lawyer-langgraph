from pathlib import Path
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(Path(__file__).resolve().parents[1], "")))

from src.utils import count_tokens, trim_messages, DocumentProcessor, LegalKnowledgeBase


@dataclass
class Message:
    content: str

def test_count_tokens():
    messages = [Message(content="hello world"), Message(content="test message")]
    assert count_tokens(messages) == 4

def test_trim_messages():
    messages = [Message(content="a " * 50), Message(content="b " * 50)]
    trimmed = trim_messages(messages, max_tokens=60)
    assert len(trimmed) == 1


def test_document_processor(tmp_path: Path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("sample text")
    processor = DocumentProcessor(str(tmp_path))
    loaded = processor.load(str(file_path))
    assert "sample text" in loaded


def test_legal_knowledge_search():
    kb = LegalKnowledgeBase()
    assert kb.search("article 19") is not None


"""Simple document processing utilities."""
import os
from typing import Optional


class DocumentProcessor:
    """Processes uploaded documents into text."""

    def __init__(self, upload_dir: Optional[str] = None) -> None:
        self.upload_dir = upload_dir or os.path.join(os.getcwd(), "uploads")
        os.makedirs(self.upload_dir, exist_ok=True)

    def save_file(self, filename: str, data: bytes) -> str:
        path = os.path.join(self.upload_dir, filename)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def parse_pdf(self, path: str) -> str:
        text = ""
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            raise RuntimeError("PyPDF2 is required for PDF parsing")

        reader = PdfReader(path)
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    def load(self, path: str) -> str:
        if path.lower().endswith(".pdf"):
            return self.parse_pdf(path)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

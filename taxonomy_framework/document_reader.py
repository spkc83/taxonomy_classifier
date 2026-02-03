"""
DocumentReader - Extract plain text from various input sources.
"""
import json
import os
from typing import Any, List


class DocumentReader:
    """Extract plain text from a variety of input sources.

    Instances of this class expose a ``read_text`` method which
    accepts a string, file path or JSON object and returns a single
    string containing all textual content.  If the input is already a
    string it is returned unchanged.  If the input is a dictionary or
    list the method concatenates all string values it finds.  File
    paths are processed according to their extension: PDFs and Word
    documents are handled if optional dependencies are present, JSON
    files are parsed and other files are read as UTF‑8 text.
    """

    def read_text(self, source: Any) -> str:
        """Return the textual content of the given source.

        Args:
            source: Either a plain string, a file path, or a JSON
                object (dict or list).  If ``source`` is a dictionary or
                list the method will attempt to collect all string
                values recursively.  If it is a file path the method
                will determine the format from the extension and call
                the appropriate helper function.

        Returns:
            The extracted text as a single string.  Non–string
            content (e.g. numbers or booleans) is ignored unless
            converted to strings.
        """
        if isinstance(source, str):
            # Determine if string is a path to an existing file
            if os.path.exists(source) and os.path.isfile(source):
                return self._read_file(source)
            # Otherwise treat it as raw text
            return source
        elif isinstance(source, dict) or isinstance(source, list):
            return self._extract_text_from_json(source)
        else:
            raise TypeError(
                "Unsupported source type. Expected str, dict or list, got "
                f"{type(source).__name__}."
            )

    def _read_file(self, file_path: str) -> str:
        """Read text from a file based on its extension."""
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        if ext == ".pdf":
            return self._read_pdf(file_path)
        elif ext in {".doc", ".docx"}:
            return self._read_docx(file_path)
        elif ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return self._extract_text_from_json(data)
        else:
            # Fallback: read file as UTF‑8 text
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

    def _read_pdf(self, file_path: str) -> str:
        """Extract text from a PDF using pdfplumber if available."""
        try:
            import pdfplumber  # type: ignore
        except ImportError:
            raise ImportError(
                "pdfplumber is required to extract text from PDFs. "
                "Install it via 'pip install pdfplumber'."
            )
        text_parts: List[str] = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts)

    def _read_docx(self, file_path: str) -> str:
        """Extract text from a Word document using python‑docx."""
        try:
            import docx  # type: ignore
        except ImportError:
            raise ImportError(
                "python‑docx is required to extract text from DOCX files. "
                "Install it via 'pip install python-docx'."
            )
        doc = docx.Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs]
        return "\n".join(paragraphs)

    def _extract_text_from_json(self, data: Any) -> str:
        """Recursively extract all string values from a JSON object."""
        texts: List[str] = []
        if isinstance(data, dict):
            for value in data.values():
                texts.append(self._extract_text_from_json(value))
        elif isinstance(data, list):
            for item in data:
                texts.append(self._extract_text_from_json(item))
        elif isinstance(data, str):
            texts.append(data)
        else:
            # Other types (int, float, bool, None) are ignored
            pass
        return " ".join([t for t in texts if t])

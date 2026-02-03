"""Tests for DocumentReader class."""
import json
import os
import pytest

from taxonomy_framework import DocumentReader


class TestDocumentReaderBasic:
    """Basic tests for DocumentReader."""

    def test_read_text_returns_plain_string(self):
        """read_text() returns input string unchanged when not a file path."""
        reader = DocumentReader()
        text = "Hello, this is plain text."
        result = reader.read_text(text)
        assert result == text

    def test_read_text_returns_multiline_string(self):
        """read_text() handles multiline strings."""
        reader = DocumentReader()
        text = "Line 1\nLine 2\nLine 3"
        result = reader.read_text(text)
        assert result == text

    def test_read_text_extracts_text_from_dict(self):
        """read_text() extracts text from dict/JSON objects."""
        reader = DocumentReader()
        data = {"title": "Test", "body": "Content here"}
        result = reader.read_text(data)
        assert "Test" in result
        assert "Content here" in result

    def test_read_text_extracts_text_from_list(self):
        """read_text() extracts text from list/JSON arrays."""
        reader = DocumentReader()
        data = ["first", "second", "third"]
        result = reader.read_text(data)
        assert "first" in result
        assert "second" in result
        assert "third" in result

    def test_read_text_raises_type_error_for_unsupported_types(self):
        """read_text() raises TypeError for unsupported input types."""
        reader = DocumentReader()
        with pytest.raises(TypeError, match="Unsupported source type"):
            reader.read_text(12345)
        with pytest.raises(TypeError, match="Unsupported source type"):
            reader.read_text(None)


class TestExtractTextFromJson:
    """Tests for _extract_text_from_json method."""

    def test_extract_text_handles_nested_dict(self):
        """_extract_text_from_json handles nested dicts."""
        reader = DocumentReader()
        data = {
            "level1": {
                "level2": {
                    "text": "deep value"
                }
            }
        }
        result = reader._extract_text_from_json(data)
        assert "deep value" in result

    def test_extract_text_handles_nested_list(self):
        """_extract_text_from_json handles nested lists."""
        reader = DocumentReader()
        data = [["a", "b"], ["c", "d"]]
        result = reader._extract_text_from_json(data)
        assert "a" in result
        assert "b" in result
        assert "c" in result
        assert "d" in result

    def test_extract_text_handles_mixed_types(self):
        """_extract_text_from_json ignores non-string types."""
        reader = DocumentReader()
        data = {
            "text": "hello",
            "number": 42,
            "boolean": True,
            "null": None,
            "nested": {"value": "world"}
        }
        result = reader._extract_text_from_json(data)
        assert "hello" in result
        assert "world" in result
        assert "42" not in result
        assert "True" not in result

    def test_extract_text_handles_empty_structures(self):
        """_extract_text_from_json handles empty dicts/lists."""
        reader = DocumentReader()
        assert reader._extract_text_from_json({}) == ""
        assert reader._extract_text_from_json([]) == ""


class TestFileReading:
    """Tests for file reading functionality."""

    def test_read_plain_text_file(self, tmpdir):
        """read_text() reads plain text files."""
        reader = DocumentReader()
        file_path = tmpdir.join("test.txt")
        file_path.write("This is test content.")
        result = reader.read_text(str(file_path))
        assert result == "This is test content."

    def test_read_json_file(self, tmpdir):
        """read_text() reads and extracts text from JSON files."""
        reader = DocumentReader()
        file_path = tmpdir.join("test.json")
        data = {"key1": "value1", "key2": "value2"}
        file_path.write(json.dumps(data))
        result = reader.read_text(str(file_path))
        assert "value1" in result
        assert "value2" in result

    def test_read_text_file_with_unknown_extension(self, tmpdir):
        """read_text() reads unknown extensions as plain text."""
        reader = DocumentReader()
        file_path = tmpdir.join("test.xyz")
        file_path.write("Custom format content")
        result = reader.read_text(str(file_path))
        assert result == "Custom format content"


class TestOptionalDependencies:
    """Tests for optional PDF/DOCX dependencies."""

    def test_pdf_import_error_message(self, tmpdir):
        """_read_pdf raises ImportError with helpful message if pdfplumber missing."""
        reader = DocumentReader()
        file_path = tmpdir.join("test.pdf")
        file_path.write_binary(b"%PDF-1.4 fake pdf content")
        
        # Only test if pdfplumber is NOT installed
        try:
            import pdfplumber
            pytest.skip("pdfplumber is installed, cannot test import error")
        except ImportError:
            with pytest.raises(ImportError, match="pdfplumber is required"):
                reader.read_text(str(file_path))

    def test_docx_import_error_message(self, tmpdir):
        """_read_docx raises ImportError with helpful message if python-docx missing."""
        reader = DocumentReader()
        file_path = tmpdir.join("test.docx")
        file_path.write_binary(b"fake docx content")
        
        # Only test if python-docx is NOT installed
        try:
            import docx
            pytest.skip("python-docx is installed, cannot test import error")
        except ImportError:
            with pytest.raises(ImportError, match="python.docx is required"):
                reader.read_text(str(file_path))


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_string_input(self):
        """read_text() handles empty string input."""
        reader = DocumentReader()
        result = reader.read_text("")
        assert result == ""

    def test_string_that_looks_like_path_but_isnt(self):
        """read_text() returns string if path doesn't exist."""
        reader = DocumentReader()
        fake_path = "/nonexistent/path/to/file.txt"
        result = reader.read_text(fake_path)
        assert result == fake_path

    def test_unicode_content(self, tmpdir):
        """read_text() handles unicode content."""
        reader = DocumentReader()
        file_path = tmpdir.join("unicode.txt")
        content = "Unicode: αβγδ εζηθ 日本語 🎉"
        file_path.write_text(content, encoding="utf-8")
        result = reader.read_text(str(file_path))
        assert result == content

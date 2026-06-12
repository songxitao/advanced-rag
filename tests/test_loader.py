import pytest
import docx
import fitz
from src.loader import DocumentLoader

def test_document_loader_txt(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello 早稻田 IPS 和东南大学", encoding="utf-8")
    loader = DocumentLoader()
    text = loader.load(str(test_file))
    assert text == "Hello 早稻田 IPS 和东南大学"

def test_document_loader_md(tmp_path):
    test_file = tmp_path / "test.md"
    test_file.write_text("# Hello Markdown\n- item 1\n- item 2", encoding="utf-8")
    loader = DocumentLoader()
    text = loader.load(str(test_file))
    assert text == "# Hello Markdown\n- item 1\n- item 2"

def test_document_loader_file_not_found():
    loader = DocumentLoader()
    with pytest.raises(FileNotFoundError):
        loader.load("non_existent_file.txt")

def test_document_loader_unsupported_format(tmp_path):
    test_file = tmp_path / "test.jpg"
    test_file.write_text("dummy", encoding="utf-8")
    loader = DocumentLoader()
    with pytest.raises(ValueError):
        loader.load(str(test_file))

def test_document_loader_pdf(tmp_path):
    test_file = tmp_path / "test.pdf"
    # 创建一个临时的 PDF 文件并写入文本
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Hello PDF Document")
    doc.save(str(test_file))
    doc.close()
    
    loader = DocumentLoader()
    text = loader.load(str(test_file))
    assert "Hello PDF Document" in text

def test_document_loader_docx(tmp_path):
    test_file = tmp_path / "test.docx"
    # 创建一个临时的 Word 文件并写入文本
    doc = docx.Document()
    doc.add_paragraph("Hello DOCX Document")
    doc.save(str(test_file))
    
    loader = DocumentLoader()
    text = loader.load(str(test_file))
    assert text == "Hello DOCX Document"

def test_document_loader_srt(tmp_path):
    test_file = tmp_path / "test.srt"
    srt_content = (
        "1\n"
        "00:00:01,000 --> 00:00:04,000\n"
        "Hello SRT Line 1\n\n"
        "2\n"
        "00:00:05,000 --> 00:00:08,000\n"
        "Hello SRT Line 2\n"
    )
    test_file.write_text(srt_content, encoding="utf-8")
    
    loader = DocumentLoader()
    text = loader.load(str(test_file))
    # 过滤后应该只剩下文本
    assert "Hello SRT Line 1" in text
    assert "Hello SRT Line 2" in text
    assert "00:00:01" not in text
    assert "-->" not in text

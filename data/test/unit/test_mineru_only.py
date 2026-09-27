"""Regression guard: QA consumes prepared Markdown; MinerU is the only ingester."""
import ast
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
from hrm_finance_qa.document_sources import discover_documents, read_document, DocumentSourceError
class MineruOnlyTests(TestCase):
 def test_runtime_has_no_other_ocr_import(self):
  forbidden={'fitz','pymupdf','pdfplumber','pytesseract','easyocr','paddleocr','pdf2image','pypdf','PyPDF2','rapidocr','ocrmypdf'}
  roots=[Path('src'),Path('tham_khao/thamkhao_int/backend/app')]
  for root in roots:
   for path in root.rglob('*.py'):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    for n in ast.walk(tree):
     imported=[a.name for a in n.names] if isinstance(n,ast.Import) else [n.module or ''] if isinstance(n,ast.ImportFrom) else []
     self.assertFalse({x.split('.')[0] for x in imported}&forbidden,str(path))
 def test_pdf_rejected_in_query_path(self):
  with TemporaryDirectory() as d:
   p=Path(d)/'report.pdf';p.write_bytes(b'%PDF-1.7\n')
   with patch('hrm_finance_qa.document_sources.subprocess.run',side_effect=AssertionError('QA must not start extraction')):
    for fn in [read_document,discover_documents]:
     with self.assertRaises(DocumentSourceError):fn(p)
 def test_markdown_never_starts_ocr(self):
  with TemporaryDirectory() as d:
   p=Path(d)/'report.md';p.write_text('## Page 1\nDoanh thu: 123 đồng\n',encoding='utf8')
   with patch('hrm_finance_qa.document_sources.subprocess.run',side_effect=AssertionError('QA must not start OCR')):
    self.assertEqual(discover_documents(p),[p.resolve()]);self.assertTrue(any('123' in row.text for row in read_document(p)))
 def test_reference_requirements_have_no_ocr_fallback(self):
  dependencies=Path('tham_khao/thamkhao_int/backend/requirements.txt').read_text().lower().splitlines()
  self.assertFalse({'pdfplumber','pymupdf','pytesseract','pdf2image'}&set(dependencies))

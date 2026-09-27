from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest

SUITE = Path(__file__).resolve().parents[3] / "data/test/suites/hrm_value_eval_20260925"
sys.path.insert(0, str(SUITE))
from mineru_provenance_gate import verify_mineru_sources


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def fixture(root, *, with_index=False):
    pdf = root / "report.pdf"
    md = root / "report.md"
    sidecar = root / "report.provenance.json"
    pdf.write_bytes(b"%PDF-1.4\n")
    md.write_text(
        "# report\n\n<!-- source: report.pdf; extractor: MinerU OCR; batch-pages: 4 -->\n"
        "## Page 1\n\n| metric | 10 |\n", encoding="utf8")
    provenance = {
        "artifact": "report.md", "artifact_sha256": digest(md),
        "pdf_path": "report.pdf", "pdf_sha256": digest(pdf),
        "pdf_pages": 1, "extractor": "MinerU API /file_parse",
    }
    sidecar.write_text(json.dumps(provenance), encoding="utf8")
    source = {
        "id": "XYZ", "artifact": "report.md", "sha256": digest(md),
        "pdf_path": "report.pdf", "pdf_sha256": digest(pdf),
        "pdf_pages": 1, "provenance": "report.provenance.json",
        "provenance_sha256": digest(sidecar),
    }
    manifest = {"evaluation_scope": "independent_holdout", "sources": [source]}
    if with_index:
        structured = root / "report_content_list.json"
        structured.write_text("[]", encoding="utf8")
        index = root / "index.json"
        index.write_text(json.dumps({"sources": [{
            "company_code": "XYZ", "artifact": "report.md",
            "artifact_sha256": digest(md), "pdf_path": "report.pdf",
            "pdf_sha256": digest(pdf), "pdf_pages": 1,
            "provenance_sha256": digest(sidecar),
            "content_list_files": [{
                "path": "report_content_list.json", "sha256": digest(structured)
            }],
        }]}), encoding="utf8")
        manifest["source_artifact_index"] = {
            "path": "index.json", "sha256": digest(index)}
    return manifest


class MinerUProvenanceGateTests(unittest.TestCase):
    def test_accepts_bound_mineru_markdown_and_content_list(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = verify_mineru_sources(fixture(root, with_index=True),
                                           base_directory=root)
            self.assertEqual(result["status"], "verified_mineru")
            self.assertEqual(result["content_list_files_verified"], 1)

    def test_rejects_another_extractor_even_with_markdown_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture(root)
            sidecar = root / "report.provenance.json"
            record = json.loads(sidecar.read_text(encoding="utf8"))
            record["extractor"] = "pdftotext"
            sidecar.write_text(json.dumps(record), encoding="utf8")
            manifest["sources"][0]["provenance_sha256"] = digest(sidecar)
            with self.assertRaisesRegex(ValueError, "not MinerU"):
                verify_mineru_sources(manifest, base_directory=root)

    def test_rejects_missing_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture(root)
            (root / "report.provenance.json").unlink()
            with self.assertRaisesRegex(ValueError, "sidecar missing"):
                verify_mineru_sources(manifest, base_directory=root)

    def test_rejects_changed_structured_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture(root, with_index=True)
            (root / "report_content_list.json").write_text("[{}]", encoding="utf8")
            with self.assertRaisesRegex(ValueError, "content_list hash changed"):
                verify_mineru_sources(manifest, base_directory=root)


if __name__ == "__main__":
    unittest.main()

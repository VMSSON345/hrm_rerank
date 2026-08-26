from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "generate_text_statement_programs",
    ROOT / "scripts" / "generate_text_statement_programs.py",
)
assert SPEC is not None and SPEC.loader is not None
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


class TextStatementProgramGenerationTests(unittest.TestCase):
    def test_extracts_two_period_rows_and_company_group(self) -> None:
        text = """CÔNG TY CỔ PHẦN DỊCH VỤ MẪU
BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH
Mã CHỈ TIÊU Thuyết minh Năm 2024 Năm 2023
01 1. Doanh thu bán hàng và cung cấp dịch vụ 21 625.175.196.590 627.781.708.902
10 2. Doanh thu thuần 622.912.195.780 625.980.584.262
20 3. Lợi nhuận gộp 37.984.975.564 13.759.538.130
30 4. Lợi nhuận thuần 76.255.310.590 98.301.146.785
"""

        company_key = GENERATOR._company_key(text, "source.txt")
        blocks = GENERATOR._statement_blocks(text)

        self.assertEqual(company_key, "cong ty co phan dich vu mau")
        self.assertEqual(len(blocks), 1)
        self.assertIn("| Chỉ tiêu | Năm 2024 (VND) | Năm 2023 (VND) |", blocks[0])
        self.assertIn(
            "| Doanh thu bán hàng và cung cấp dịch vụ | 625.175.196.590 | 627.781.708.902 |",
            blocks[0],
        )

    def test_rejects_rows_with_ambiguous_or_corrupt_values(self) -> None:
        text = """CÔNG TY CỔ PHẦN DỊCH VỤ MẪU
Mã CHỈ TIÊU Thuyết minh Năm 2024 Năm 2023
01 Doanh thu 625.175.196.590 627.781.708.902
02 Dòng OCR hỏng 208.01X703358 123.000.000
03 Chỉ có một giá trị 100.000.000
04 Nợ phải trả 113.945.389.064 113.374.028.230
"""

        blocks = GENERATOR._statement_blocks(text)

        self.assertEqual(len(blocks), 1)
        self.assertIn("Doanh thu", blocks[0])
        self.assertIn("Nợ phải trả", blocks[0])
        self.assertNotIn("OCR hỏng", blocks[0])
        self.assertNotIn("Chỉ có một", blocks[0])

    def test_company_group_joins_wrapped_name_and_ignores_auditor(self) -> None:
        text = """CONG TY CO PHAN
MOI TRUONG DO
THI NHA TRANG
BAO CAO TAI CHINH NAM 2024
CONG TY TNHH KIEM TOAN CPA VIETNAM
"""

        company_key = GENERATOR._company_key(text, "source.txt")

        self.assertEqual(company_key, "cong ty co phan moi truong do thi nha trang")

    def test_company_group_truncates_disclosure_boilerplate(self) -> None:
        text = (
            "CONG TY CO PHAN DICH VU SONADEZI "
            "GIAI TRINH TINH HINH BIEN DONG LOI NHUAN SAU THUE\n"
        )

        company_key = GENERATOR._company_key(text, "source.txt")

        self.assertEqual(company_key, "cong ty co phan dich vu sonadezi")

    def test_company_group_prefers_complete_name_over_repeated_fragment(self) -> None:
        text = """CONG TY CO PHAN MOI
CONG TY CO PHAN MOI
CONG TY CO PHAN MOI TRUONG DO THI NHA TRANG
"""

        company_key = GENERATOR._company_key(text, "source.txt")

        self.assertEqual(company_key, "cong ty co phan moi truong do thi nha trang")

    def test_company_group_ignores_historical_legal_form_sentence(self) -> None:
        text = """CONG TY CO PHAN NUOC GIAI KHAT SANNA KHANH HOA
Cong ty la doanh nghiep duoc chuyen tu Cong ty TNHH mot thanh vien sang cong ty co phan
"""

        company_key = GENERATOR._company_key(text, "source.txt")

        self.assertEqual(company_key, "cong ty co phan nuoc giai khat sanna khanh hoa")

    def test_company_manifest_requires_exact_source_hash(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "report.txt"
            source.write_text("financial statement", encoding="utf-8")
            manifest = root / "companies.json"
            manifest.write_text(
                json.dumps(
                    {
                        "protocol": "text-statement-company-manifest-v1",
                        "documents": {
                            source.name: {
                                "company_key": "cong ty co phan mau",
                                "source_sha256": GENERATOR._sha256(source),
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            companies = GENERATOR._load_company_manifest(manifest, [source])

            self.assertEqual(companies, {source.name: "cong ty co phan mau"})
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["documents"][source.name]["source_sha256"] = "0" * 64
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                GENERATOR._load_company_manifest(manifest, [source])


if __name__ == "__main__":
    unittest.main()

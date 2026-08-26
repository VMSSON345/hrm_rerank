from __future__ import annotations

import unittest

from scripts.discover_v12c_negative_ambiguity_pdf_pages import near_miss_type, role_mentions


class V12CNegativeAmbiguityDiscoveryTests(unittest.TestCase):
    def test_missing_long_term_role_is_a_near_miss(self) -> None:
        mentions = role_mentions("Vay ngắn hạn 100 Tổng cộng nguồn vốn 900")
        self.assertEqual(mentions["short_term_borrowings"], 1)
        self.assertEqual(mentions["long_term_borrowings"], 0)
        self.assertEqual(mentions["total_capital"], 1)
        self.assertEqual(near_miss_type(mentions), "missing_long_term_borrowings")

    def test_duplicate_complete_page_requires_ambiguity_review(self) -> None:
        mentions = role_mentions(
            "Vay ngắn hạn 10 Vay ngắn hạn 11 Vay dài hạn 20 Tổng nguồn vốn 100"
        )
        self.assertEqual(
            near_miss_type(mentions),
            "complete_lexical_with_duplicate_role_mentions",
        )

    def test_single_role_is_not_shortlisted(self) -> None:
        self.assertIsNone(near_miss_type(role_mentions("Vay dài hạn 100")))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web_demo"


class WebDemoPackagingTests(unittest.TestCase):
    def test_demo_assets_are_self_contained(self) -> None:
        html = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        self.assertIn('lang="vi"', html)
        self.assertIn('href="/styles.css"', html)
        self.assertIn('src="/app.js"', html)
        self.assertNotRegex(html, r"https?://")

    def test_html_ids_are_unique(self) -> None:
        html = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        identifiers = re.findall(r'\bid="([^"]+)"', html)
        self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_design_tokens_remain_achromatic(self) -> None:
        css = (WEB_ROOT / "styles.css").read_text(encoding="utf-8")
        for token in (
            "--color-sidebar-mist: #f9f9f9",
            "--color-pure-white: #ffffff",
            "--color-graphite-ink: #0d0d0d",
            "--color-hairline: #0000001a",
        ):
            self.assertIn(token, css)
        self.assertNotIn("box-shadow", css)

    def test_checkpoint_and_upload_routes_are_wired(self) -> None:
        script = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
        server = (ROOT / "scripts" / "serve_document_hrm.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("/v1/upload", script)
        self.assertIn("/v1/ingest", script)
        self.assertIn("/v1/answer", script)
        self.assertIn("document_hrm_V8_candidate.pt", server)

    def test_conversation_scroll_is_user_controlled(self) -> None:
        html = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        css = (WEB_ROOT / "styles.css").read_text(encoding="utf-8")
        script = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="jump-to-latest"', html)
        self.assertIn("overscroll-behavior-y: contain", css)
        self.assertIn("scrollbar-gutter: stable", css)
        self.assertIn("isNearConversationBottom", script)
        self.assertIn("shouldFollowAnswer", script)
        self.assertIn('addEventListener(\n      "scroll"', script)

    def test_abstained_answer_is_withheld_from_primary_summary(self) -> None:
        script = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

        self.assertIn('answer.status === "ABSTAIN"', script)
        self.assertIn("candidate_answer", script)
        self.assertIn("Không đủ bằng chứng để kết luận", script)


if __name__ == "__main__":
    unittest.main()

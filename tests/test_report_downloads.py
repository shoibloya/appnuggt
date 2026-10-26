"""Prepared report downloads must carry their bytes, not a temporary server URL."""

import base64
from html.parser import HTMLParser
from pathlib import Path
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

from streamlit.testing.v1 import AppTest

from app_ui import download_link_html
from design_thinking import build_pdf, to_markdown
from tests.fixtures import report


class DownloadParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.anchors = []
        self.tags = []
        self.text = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        if tag == "a":
            self.anchors.append(dict(attrs))

    def handle_data(self, data):
        self.text.append(data)


def parse_download(html):
    parser = DownloadParser()
    parser.feed(html)
    if len(parser.anchors) != 1:
        raise AssertionError("Expected one download anchor")
    return parser, parser.anchors[0]


class ReportDownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_report = report()
        cls.pdf = build_pdf(cls.sample_report)
        cls.markdown = to_markdown(cls.sample_report).encode("utf-8")

    def test_pdf_link_decodes_to_the_complete_original_pdf(self):
        html = download_link_html(self.pdf, "nuggt-design-thinking-report.pdf", "application/pdf", "Download PDF")
        parser, anchor = parse_download(html)
        self.assertEqual(parser.tags, ["a"])
        self.assertEqual(anchor["download"], "nuggt-design-thinking-report.pdf")
        self.assertEqual(anchor["target"], "_self")
        self.assertEqual(urlsplit(anchor["href"]).scheme, "data")
        prefix, encoded = anchor["href"].split(",", 1)
        self.assertEqual(prefix, "data:application/pdf;base64")
        recovered = base64.b64decode(encoded, validate=True)
        self.assertEqual(recovered, self.pdf)
        self.assertTrue(recovered.startswith(b"%PDF-"))
        self.assertTrue(recovered.rstrip().endswith(b"%%EOF"))
        self.assertGreater(len(recovered), 10000)
        self.assertNotIn("/media/", prefix)

    def test_markdown_link_carries_exact_utf8_report_and_correct_name(self):
        _, anchor = parse_download(download_link_html(
            self.markdown, "nuggt-design-thinking-report.md", "text/markdown", "Markdown"
        ))
        prefix, encoded = anchor["href"].split(",", 1)
        self.assertEqual(prefix, "data:text/markdown;base64")
        self.assertEqual(anchor["download"], "nuggt-design-thinking-report.md")
        self.assertEqual(base64.b64decode(encoded, validate=True), self.markdown)
        self.assertIn("## Desirability", base64.b64decode(encoded).decode("utf-8"))

    def test_filename_and_label_cannot_create_html_or_event_attributes(self):
        filename = 'report" onmouseover="alert(1)<&>.pdf'
        label = '<script>alert("test")</script> & Download'
        html = download_link_html(self.pdf, filename, "application/pdf", label)
        parser, anchor = parse_download(html)
        self.assertEqual(anchor["download"], filename)
        self.assertEqual(set(anchor), {"class", "target", "download", "href"})
        self.assertEqual(parser.tags, ["a"])
        self.assertEqual("".join(parser.text), label)
        self.assertNotIn("<script>", html)
        self.assertIn("&quot;", html)

    def test_empty_nonbinary_and_nonpdf_payloads_are_rejected(self):
        for data in (b"", None, "PDF text", bytearray(b"%PDF-1.7"), b"not a PDF", b"<html>404: missing file</html>"):
            with self.subTest(payload_type=type(data).__name__), self.assertRaises(ValueError):
                download_link_html(data, "report.pdf", "application/pdf", "Download PDF")
        with self.assertRaisesRegex(ValueError, "Unsupported report format"):
            download_link_html(self.pdf, "report.exe", "application/octet-stream", "Download")

    def test_completed_report_renders_embedded_downloads_without_media_widgets(self):
        page = Path(__file__).resolve().parents[1] / "pages" / "Design Thinking.py"
        app = AppTest.from_file(str(page), default_timeout=20)
        app.session_state["dt_report"] = self.sample_report
        with patch("design_thinking.evaluate_idea", side_effect=AssertionError("Download rendering must not run research")) as evaluate:
            app.run()
        self.assertEqual([item.message for item in app.exception], [])
        evaluate.assert_not_called()
        self.assertEqual(len(app.get("download_button")), 0)
        downloads = []
        for markdown in app.markdown:
            if 'class="dt-download-link"' in markdown.value:
                _, anchor = parse_download(markdown.value)
                downloads.append(anchor)
        self.assertEqual({anchor["download"] for anchor in downloads}, {
            "nuggt-design-thinking-report.pdf", "nuggt-design-thinking-report.md",
        })
        self.assertTrue(all(urlsplit(anchor["href"]).scheme == "data" for anchor in downloads))


if __name__ == "__main__":
    unittest.main()

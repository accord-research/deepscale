#!/usr/bin/env python3
"""Inject actual PDF page numbers into the generated validation-report HTML."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


def _headings(markdown_path: Path) -> list[str]:
    headings: list[str] = []
    for line in markdown_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            headings.append(line[3:].strip())
    return headings


def _page_count(pdf_path: Path) -> int:
    info = subprocess.check_output(
        ["pdfinfo", str(pdf_path)],
        text=True,
    )
    match = re.search(r"^Pages:\s+(\d+)$", info, flags=re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not read page count from {pdf_path}")
    return int(match.group(1))


def _pdf_page(pdf_path: Path, page_number: int) -> str:
    return subprocess.check_output(
        ["pdftotext", "-layout", "-f", str(page_number), "-l", str(page_number), str(pdf_path), "-"],
        text=True,
    )


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _page_map(pdf_path: Path, headings: list[str]) -> dict[str, int]:
    page_map: dict[str, int] = {}
    page_count = _page_count(pdf_path)
    for heading in headings:
        normalized_heading = _normalize(heading)
        for index in range(2, page_count + 1):
            page = _pdf_page(pdf_path, index)
            lines = [_normalize(line) for line in page.splitlines()]
            visible_lines = [line for line in lines if line]
            if not visible_lines:
                continue
            if normalized_heading in visible_lines[:4]:
                page_map[heading] = index
                break
        if heading not in page_map:
            raise RuntimeError(f"Could not find heading in draft PDF: {heading}")
    return page_map


def _inject(html_path: Path, page_map: dict[str, int]) -> None:
    html = html_path.read_text(encoding="utf-8")
    payload = (
        "<script>\n"
        "window.REPORT_TOC_PAGES = "
        + json.dumps(page_map, ensure_ascii=False, sort_keys=True)
        + ";\n"
        "</script>\n"
    )
    marker = "<script>\n(function () {"
    if marker not in html:
        raise RuntimeError("Could not find TOC script marker in generated HTML")
    html = html.replace(marker, payload + marker, 1)
    html_path.write_text(html, encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 4:
        print(
            "Usage: fill_report_toc_pages.py REPORT.md DRAFT.pdf REPORT.html",
            file=sys.stderr,
        )
        return 2

    markdown_path = Path(sys.argv[1])
    draft_pdf_path = Path(sys.argv[2])
    html_path = Path(sys.argv[3])

    page_map = _page_map(draft_pdf_path, _headings(markdown_path))
    _inject(html_path, page_map)
    print(json.dumps(page_map, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

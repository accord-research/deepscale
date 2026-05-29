"""Integration test for scripts.s2s.publish_pages (gh-pages tree assembly)."""
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _make_site(tmp_path: Path) -> Path:
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text(
        "<!doctype html><html><head><link rel='stylesheet' href='theme.css'></head>"
        "<body><a class='card' href='s2s/'>Sub-seasonal</a>"
        "<div class='card disabled'>Seasonal coming soon</div></body></html>"
    )
    (site / "theme.css").write_text(":root{--bg:#1a1a1a}")
    return site


def _make_s2s_build(tmp_path: Path) -> Path:
    build = tmp_path / "build" / "s2s"
    (build / "kenya" / "2026-05-21").mkdir(parents=True)
    (build / "kenya" / "metrics.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (build / "kenya" / "2026-05-21" / "comparison.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (build / "index.html").write_text(
        "<link rel='stylesheet' href='../theme.css'>"
        "<a href='../'>All forecasts</a>"
        "<img src='kenya/metrics.png'><img src='kenya/2026-05-21/comparison.png'>"
    )
    return build


def test_publish_assembles_tree_and_preserves_siblings(tmp_path):
    from scripts.s2s.publish_pages import publish_pages
    site = _make_site(tmp_path)
    build = _make_s2s_build(tmp_path)
    pages = tmp_path / "gh-pages"
    (pages / "seasonal").mkdir(parents=True)          # sibling owned by another pipeline
    (pages / "seasonal" / "sentinel").write_text("keep me")

    publish_pages(site_src=site, s2s_build=build, pages_root=pages)

    assert (pages / "index.html").exists()            # hub shell at root
    assert (pages / "theme.css").exists()
    hub = (pages / "index.html").read_text()
    assert "theme.css" in hub
    assert "href='s2s/'" in hub                        # S2S tile is an active link
    assert "disabled" in hub                           # seasonal placeholder

    s2s_index = pages / "s2s" / "index.html"           # s2s subtree rebuilt
    assert s2s_index.exists()
    assert "../theme.css" in s2s_index.read_text()
    assert (pages / "s2s" / "kenya" / "metrics.png").exists()
    assert (pages / "s2s" / "kenya" / "2026-05-21" / "comparison.png").exists()

    assert (pages / "seasonal" / "sentinel").read_text() == "keep me"  # untouched


def test_publish_rebuild_drops_stale_s2s_files(tmp_path):
    from scripts.s2s.publish_pages import publish_pages
    site = _make_site(tmp_path)
    build = _make_s2s_build(tmp_path)
    pages = tmp_path / "gh-pages"
    (pages / "s2s" / "ethiopia").mkdir(parents=True)   # stale from a previous publish
    (pages / "s2s" / "ethiopia" / "old.png").write_bytes(b"stale")

    publish_pages(site_src=site, s2s_build=build, pages_root=pages)

    assert not (pages / "s2s" / "ethiopia" / "old.png").exists()  # rebuilt clean
    assert (pages / "s2s" / "kenya" / "metrics.png").exists()


def test_publish_without_build_preserves_existing_s2s(tmp_path):
    from scripts.s2s.publish_pages import publish_pages
    site = _make_site(tmp_path)
    pages = tmp_path / "gh-pages"
    (pages / "s2s").mkdir(parents=True)
    (pages / "s2s" / "index.html").write_text("previous dashboard")

    publish_pages(site_src=site, s2s_build=None, pages_root=pages)

    assert (pages / "index.html").exists()             # hub + theme still installed
    assert (pages / "theme.css").exists()
    assert (pages / "s2s" / "index.html").read_text() == "previous dashboard"  # intact

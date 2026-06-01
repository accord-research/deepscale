"""End-to-end integration test for scripts.s2s.render_dashboard."""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

pytestmark = pytest.mark.integration


def _make_method_ds(seed: int):
    rng = np.random.default_rng(seed)
    lat = np.linspace(-5, 5, 24)
    lon = np.linspace(33, 42, 36)
    return xr.Dataset(
        {"mean": (("lat", "lon"), rng.gamma(2, 1.5, (24, 36)).astype("float32"))},
        coords={"lat": lat, "lon": lon},
    )


def _seed_store(store_root: Path, country: str, issuance: date, targets: list[date]):
    from scripts.s2s.issuance_store import write_issuance
    seed = 0
    for method in ["raw", "climatology", "bcsd"]:
        for tgt in targets:
            write_issuance(store_root, country, issuance, method, tgt, _make_method_ds(seed))
            seed += 1


def _seed_verification(verif_root: Path, country: str, records: list[dict]):
    p = verif_root / country / "scores.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")


@pytest.fixture
def cfg_path(tmp_path):
    cfg_dict = {
        "countries": {
            "kenya": {
                "bbox": {"min_lat": -5.0, "max_lat": 5.0, "min_lon": 33.0, "max_lon": 42.0},
                "methods": ["raw", "climatology", "bcsd"],
                "obs": "obs/chirps-dekadal",
                "forecast": "c3s/ecmwf-s2s",
                "variable": "precip",
            },
        },
        "lead_days": {"min": 0, "max": 46},
        "climatology_years": [1991, 2020],
        "store_root": str(tmp_path / "issuances"),
    }
    p = tmp_path / "s2s.yml"
    p.write_text(yaml.safe_dump(cfg_dict))
    return p


def test_render_dashboard_produces_pngs(cfg_path, tmp_path):
    from scripts.s2s.render_dashboard import render_dashboard
    store = tmp_path / "issuances"
    verif = tmp_path / "verification"
    dashboard = tmp_path / "dashboard"

    issuance = date(2026, 5, 15)
    targets = [date(2026, 5, 21), date(2026, 6, 1)]
    _seed_store(store, "kenya", issuance, targets)
    _seed_verification(verif, "kenya", [
        {"country": "kenya", "issuance": "2026-05-15", "method": "raw",
         "target_dekad": "2026-05-21", "acc": 0.1, "rmse": 1.0, "bias": 0.0, "rpss": 0.0},
        {"country": "kenya", "issuance": "2026-05-15", "method": "bcsd",
         "target_dekad": "2026-05-21", "acc": 0.3, "rmse": 0.8, "bias": -0.1, "rpss": 0.05},
    ])

    render_dashboard(store_root=store, verification_root=verif, dashboard_root=dashboard, config_path=cfg_path)

    # Comparison grid per (country, issuance)
    assert (dashboard / "kenya" / "2026-05-15" / "comparison.png").exists()
    # Metrics panel per country
    assert (dashboard / "kenya" / "metrics.png").exists()
    # Top-level index
    assert (dashboard / "index.html").exists()
    html = (dashboard / "index.html").read_text()
    assert "kenya" in html
    assert "2026-05-15" in html
    assert "../theme.css" in html        # links the shared dark theme
    assert "All forecasts" in html       # hub back-link present
    # Skill-over-time plots are still rendered + deployed, but intentionally NOT
    # surfaced on the public index page.
    assert (dashboard / "kenya" / "metrics.png").exists()   # still created
    assert "metrics.png" not in html                         # but not linked
    assert "Skill over time" not in html                     # section removed


def _seed_store_methods(store_root: Path, country: str, issuance: date, targets: list[date], methods: list[str]):
    from scripts.s2s.issuance_store import write_issuance
    seed = 100
    for method in methods:
        for tgt in targets:
            write_issuance(store_root, country, issuance, method, tgt, _make_method_ds(seed))
            seed += 1


def test_render_dashboard_flags_full_vs_degraded_and_adds_filter(cfg_path, tmp_path):
    """Issuances with a downscaling method are flagged full; raw+climatology-only
    ones are flagged not-full, and the index offers a 'full weeks only' checkbox
    (default on) plus a 'baselines only' label for the degraded ones."""
    import re
    from scripts.s2s.render_dashboard import render_dashboard
    store = tmp_path / "issuances"
    verif = tmp_path / "verification"
    dashboard = tmp_path / "dashboard"

    tgt = [date(2026, 5, 21)]
    _seed_store_methods(store, "kenya", date(2026, 5, 15), tgt, ["raw", "climatology", "bcsd"])  # full
    _seed_store_methods(store, "kenya", date(2026, 5, 11), tgt, ["raw", "climatology"])          # degraded

    render_dashboard(store_root=store, verification_root=verif, dashboard_root=dashboard, config_path=cfg_path)
    html = (dashboard / "index.html").read_text()

    # Checkbox present and defaulted on.
    assert "id='full-only'" in html
    assert "checkbox" in html and "checked" in html
    # Embedded data distinguishes full from degraded.
    m = re.search(r"const COMPARISONS = (.+);", html)
    assert m, "COMPARISONS object not found in index.html"
    by_date = {e["d"]: e["full"] for e in json.loads(m.group(1))["kenya"]}
    assert by_date["2026-05-15"] is True    # has bcsd → full week
    assert by_date["2026-05-11"] is False   # raw+climatology only → degraded
    # Degraded issuances get a label so they don't read as broken.
    assert "baselines only" in html


def test_render_dashboard_skips_missing_data(cfg_path, tmp_path):
    """Empty store → index.html still produced, no crash."""
    from scripts.s2s.render_dashboard import render_dashboard
    dashboard = tmp_path / "dashboard"
    render_dashboard(
        store_root=tmp_path / "empty_store",
        verification_root=tmp_path / "empty_verif",
        dashboard_root=dashboard,
        config_path=cfg_path,
    )
    assert (dashboard / "index.html").exists()


def test_render_dashboard_windows_comparison_maps(cfg_path, tmp_path):
    """Only issuances within comparison_window_days of the latest get a comparison
    map; the selector lists exactly those. Skill metrics keep full history."""
    from scripts.s2s.render_dashboard import render_dashboard
    store = tmp_path / "issuances"
    verif = tmp_path / "verification"
    dashboard = tmp_path / "dashboard"

    old = date(2026, 1, 1)        # > 90 days before the latest issuance
    recent = date(2026, 5, 15)
    _seed_store(store, "kenya", old, [date(2026, 1, 5)])
    _seed_store(store, "kenya", recent, [date(2026, 5, 21)])

    render_dashboard(store_root=store, verification_root=verif,
                     dashboard_root=dashboard, config_path=cfg_path)

    # In-window issuance has a comparison map; out-of-window one does not.
    assert (dashboard / "kenya" / "2026-05-15" / "comparison.png").exists()
    assert not (dashboard / "kenya" / "2026-01-01").exists()

    html_txt = (dashboard / "index.html").read_text()
    assert "sel-country" in html_txt and "sel-issuance" in html_txt   # selector markup
    assert "COMPARISONS" in html_txt                                  # embedded data
    assert "2026-05-15" in html_txt                                   # recent offered
    assert "2026-01-01" not in html_txt                               # windowed out
    assert (dashboard / "kenya" / "metrics.png").exists()             # full-history summary


def test_comparison_grid_obs_pending_stays_three_columns():
    """With no obs yet the grid keeps observed | forecast | difference (3 cols)
    with 'pending' placeholders, instead of collapsing to one narrow column —
    and the method row labels read horizontally (rotation 0)."""
    import matplotlib.pyplot as plt
    from scripts.s2s.plotting import comparison_grid

    methods = {m: _make_method_ds(i) for i, m in enumerate(["raw", "climatology", "bcsd"])}
    fig = comparison_grid(None, methods, dekad_label="2026-05-21")
    try:
        titles = [ax.get_title() for ax in fig.axes]
        assert any("observed" in t for t in titles)      # all three column headers
        assert any(t.strip() == "forecast" for t in titles)
        assert any("difference" in t for t in titles)
        # Empty obs/difference cells are placeholders, not blank.
        texts = [t.get_text().lower() for ax in fig.axes for t in ax.texts]
        assert any("pending" in t for t in texts)
        # Row labels read horizontally (vertical labels were half the "rotated" look).
        rotations = [ax.yaxis.label.get_rotation() for ax in fig.axes
                     if ax.yaxis.label.get_text() in {"raw", "climatology", "bcsd"}]
        assert rotations and all(r == 0 for r in rotations)
    finally:
        plt.close(fig)

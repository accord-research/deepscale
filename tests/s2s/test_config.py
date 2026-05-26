"""Unit tests for scripts.s2s.config.load_config."""

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def sample_config_path(tmp_path):
    cfg = {
        "countries": {
            "kenya": {
                "bbox": {"min_lat": -5.0, "max_lat": 5.5, "min_lon": 33.5, "max_lon": 42.0},
                "methods": ["raw", "climatology", "bcsd", "cca", "rank-analog"],
                "obs": "obs/chirps-dekadal",
                "forecast": "c3s/ecmwf-s2s",
                "variable": "precip",
            },
        },
        "lead_days": {"min": 0, "max": 46},
        "climatology_years": [1991, 2020],
        "store_root": "issuances",
    }
    path = tmp_path / "s2s.yml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_load_config_returns_object_with_countries(sample_config_path):
    from scripts.s2s.config import load_config
    cfg = load_config(sample_config_path)
    assert "kenya" in cfg.countries
    kenya = cfg.countries["kenya"]
    assert kenya.bbox == {"min_lat": -5.0, "max_lat": 5.5, "min_lon": 33.5, "max_lon": 42.0}
    assert kenya.methods == ["raw", "climatology", "bcsd", "cca", "rank-analog"]
    assert kenya.obs == "obs/chirps-dekadal"
    assert kenya.forecast == "c3s/ecmwf-s2s"


def test_load_config_global_fields(sample_config_path):
    from scripts.s2s.config import load_config
    cfg = load_config(sample_config_path)
    assert cfg.lead_days == (0, 46)
    assert cfg.climatology_years == (1991, 2020)
    assert cfg.store_root == "issuances"


def test_load_config_rejects_unknown_method_in_country(tmp_path):
    """Unknown method in config raises rather than failing silently mid-run."""
    cfg = {
        "countries": {
            "kenya": {
                "bbox": {"min_lat": -5.0, "max_lat": 5.5, "min_lon": 33.5, "max_lon": 42.0},
                "methods": ["nonexistent-method"],
                "obs": "obs/chirps-dekadal",
                "forecast": "c3s/ecmwf-s2s",
                "variable": "precip",
            },
        },
        "lead_days": {"min": 0, "max": 46},
        "climatology_years": [1991, 2020],
        "store_root": "issuances",
    }
    path = tmp_path / "bad.yml"
    path.write_text(yaml.safe_dump(cfg))
    from scripts.s2s.config import load_config
    with pytest.raises(ValueError, match="unknown method"):
        load_config(path)


def test_load_config_respects_env_override(sample_config_path, monkeypatch):
    from scripts.s2s.config import load_config
    monkeypatch.setenv("S2S_STORE_ROOT", "/tmp/override-store")
    cfg = load_config(sample_config_path)
    assert cfg.store_root == "/tmp/override-store"

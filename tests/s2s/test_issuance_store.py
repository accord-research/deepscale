"""Unit tests for scripts.s2s.issuance_store."""

from datetime import date

import numpy as np
import pytest
import xarray as xr


def _make_field(prng=None):
    prng = prng or np.random.default_rng(0)
    return xr.Dataset(
        {
            "mean": (("lat", "lon"), prng.standard_normal((4, 6)).astype("float32")),
            "tercile_probs": (
                ("category", "lat", "lon"),
                prng.dirichlet([1, 1, 1], size=(4, 6)).transpose(2, 0, 1).astype("float32"),
            ),
        },
        coords={
            "lat": np.linspace(-2, 2, 4),
            "lon": np.linspace(30, 35, 6),
            "category": ["below", "normal", "above"],
        },
    )


def test_issuance_path_layout(tmp_path):
    from scripts.s2s.issuance_store import issuance_path
    p = issuance_path(tmp_path, "kenya", date(2026, 5, 15), "bcsd", date(2026, 5, 21))
    assert p == tmp_path / "kenya" / "2026-05-15" / "bcsd" / "dekad_2026-05-21.nc"


def test_write_and_read_round_trip(tmp_path):
    from scripts.s2s.issuance_store import write_issuance, read_issuance
    ds = _make_field()
    write_issuance(tmp_path, "kenya", date(2026, 5, 15), "bcsd", date(2026, 5, 21), ds)
    read = read_issuance(tmp_path, "kenya", date(2026, 5, 15), "bcsd", date(2026, 5, 21))
    xr.testing.assert_allclose(ds, read)


def test_write_creates_parent_dirs(tmp_path):
    from scripts.s2s.issuance_store import write_issuance
    write_issuance(tmp_path, "kenya", date(2026, 5, 15), "bcsd", date(2026, 5, 21), _make_field())
    assert (tmp_path / "kenya" / "2026-05-15" / "bcsd" / "dekad_2026-05-21.nc").exists()


def test_list_pending_pairs_returns_unscored(tmp_path):
    """list_pending_pairs returns (country, issuance, method, dekad) tuples
    present in the store but not yet present in verification/."""
    from scripts.s2s.issuance_store import write_issuance, list_pending_pairs
    write_issuance(tmp_path / "issuances", "kenya", date(2026, 5, 15), "bcsd", date(2026, 5, 21), _make_field())
    write_issuance(tmp_path / "issuances", "kenya", date(2026, 5, 15), "raw",  date(2026, 5, 21), _make_field())
    scored = {("kenya", date(2026, 5, 15), "bcsd", date(2026, 5, 21))}
    pending = list_pending_pairs(tmp_path / "issuances", already_scored=scored)
    assert pending == [("kenya", date(2026, 5, 15), "raw", date(2026, 5, 21))]

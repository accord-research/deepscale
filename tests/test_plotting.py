import numpy as np
import pytest
import xarray as xr


# ===================================================================
# 15. Plotting subpackage
# ===================================================================

def test_plotting_package_imports():
    """Package must import cleanly even when matplotlib/cartopy aren't installed."""
    import deepscale.plotting  # noqa: F401


def test_plot_skill_maps_smoke():
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt
    from deepscale.skill import SkillReport
    from deepscale.plotting.skill import plot_skill_maps

    lat = np.linspace(-5, 5, 6)
    lon = np.linspace(30, 45, 8)
    rpss = xr.DataArray(
        np.random.RandomState(0).uniform(-1, 1, (6, 8)),
        dims=["lat", "lon"], coords={"lat": lat, "lon": lon},
    )
    rmse = xr.DataArray(
        np.random.RandomState(1).uniform(0, 2, (6, 8)),
        dims=["lat", "lon"], coords={"lat": lat, "lon": lon},
    )
    report = SkillReport(scores={"rpss": float(rpss.mean()), "rmse": float(rmse.mean())},
                         spatial={"rpss": rpss, "rmse": rmse})

    fig = plot_skill_maps(report, ["rpss", "rmse"], ncols=2)

    assert fig is not None
    assert len(fig.axes) >= 2
    plt.close(fig)


def test_plot_domains_smoke():
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt
    from deepscale.plotting.domains import plot_domains

    # predictand: East Africa, predictor: tropical Pacific (antimeridian-spanning)
    fig = plot_domains(
        predictor_extent=(-20, 20, 120, -60),     # lon_w > lon_e — crosses dateline
        predictand_extent=(-12, 15, 22, 52),
    )

    assert fig is not None
    plt.close(fig)


def test_plot_tercile_forecast_smoke():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from deepscale.plotting.forecasts import plot_tercile_forecast

    n_lat, n_lon = 4, 5
    probs = np.zeros((3, n_lat, n_lon))
    probs[0, :, :] = 0.15
    probs[1, :, :] = 0.25
    probs[2, :, :] = 0.60
    pr_fcst = xr.DataArray(
        probs,
        dims=["tercile", "lat", "lon"],
        coords={
            "tercile": [0, 1, 2],
            "lat": np.linspace(-5, 5, n_lat),
            "lon": np.linspace(30, 45, n_lon),
        },
    )
    fig = plot_tercile_forecast(pr_fcst)
    assert fig is not None
    plt.close(fig)


def test_plot_deterministic_forecast_smoke():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from deepscale.plotting.forecasts import plot_deterministic_forecast

    n_lat, n_lon = 4, 5
    da = xr.DataArray(
        np.random.RandomState(2).randn(n_lat, n_lon),
        dims=["lat", "lon"],
        coords={"lat": np.linspace(-5, 5, n_lat), "lon": np.linspace(30, 45, n_lon)},
    )
    fig = plot_deterministic_forecast(da, title="test")
    assert fig is not None
    plt.close(fig)


def test_plot_exceedance_probability_smoke():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from deepscale.plotting.forecasts import plot_exceedance_probability

    n_lat, n_lon = 4, 5
    da = xr.DataArray(
        np.random.RandomState(3).uniform(0, 1, (n_lat, n_lon)),
        dims=["lat", "lon"],
        coords={"lat": np.linspace(-5, 5, n_lat), "lon": np.linspace(30, 45, n_lon)},
    )
    fig = plot_exceedance_probability(da, threshold=100.0)
    assert fig is not None
    plt.close(fig)


def test_plot_flex_pdf_smoke():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from deepscale.plotting.forecasts import plot_flex_pdf

    fig = plot_flex_pdf(
        fcst_mu=2.5, fcst_scale=1.2,
        climo_mu=2.0, climo_scale=1.5,
        location=(35.0, 0.0),
    )
    assert fig is not None
    plt.close(fig)


# ===================================================================
# Reliability diagram (paired with metrics/reliability tests in test_metrics)
# ===================================================================

def test_plot_reliability_diagram_smoke(synthetic_obs):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from deepscale.plotting.reliability import plot_reliability_diagram

    n_year, n_lat, n_lon = synthetic_obs.shape
    fcst = np.ones((n_year, 3, n_lat, n_lon)) / 3.0
    forecast = xr.DataArray(
        fcst, dims=["year", "tercile", "lat", "lon"],
        coords={
            "year": synthetic_obs.year,
            "tercile": [0, 1, 2],
            "lat": synthetic_obs.lat,
            "lon": synthetic_obs.lon,
        },
    )
    fig = plot_reliability_diagram(forecast, synthetic_obs)
    assert fig is not None
    plt.close(fig)


# ===================================================================
# 18b. EOF / CCA mode plots (§3.2)
# ===================================================================

def _build_dual_grid_fixture(seed=0, n_years=25, signal_amp=2.0, noise_amp=0.3):
    """Synthetic SST→precip dual-grid fixture (duplicated from test_methods.py).

    Predictor: 'tropical Pacific' SST on a coarse 6x8 grid (lat ±10°, lon 180-240°).
    Predictand: 'East Africa' precip on a fine 12x12 grid (lat -5 to 15°, lon 30-50°).
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2000, 2000 + n_years)
    members = np.arange(3)

    p_lat = np.linspace(-10, 10, 6)
    p_lon = np.linspace(180, 240, 8)
    o_lat = np.linspace(-5, 15, 12)
    o_lon = np.linspace(30, 50, 12)

    t = np.arange(n_years)
    time_signal = np.sin(2 * np.pi * t / 5.0)

    p_pattern = np.outer(np.sin(np.deg2rad(p_lat) * 3), np.cos(np.deg2rad(p_lon) * 2))
    o_pattern = np.outer(np.cos(np.deg2rad(o_lat) * 2), np.sin(np.deg2rad(o_lon) * 4))

    p_signal = signal_amp * time_signal[:, None, None] * p_pattern[None, :, :]
    o_signal = signal_amp * time_signal[:, None, None] * o_pattern[None, :, :]

    p_noise = rng.standard_normal((n_years, len(members), len(p_lat), len(p_lon))) * noise_amp
    o_noise = rng.standard_normal((n_years, len(o_lat), len(o_lon))) * noise_amp

    predictor = xr.DataArray(
        p_signal[:, None, :, :] + p_noise + 290.0,
        dims=["year", "member", "lat", "lon"],
        coords={"year": years, "member": members, "lat": p_lat, "lon": p_lon},
    )
    predictand = xr.DataArray(
        o_signal + o_noise + 5.0,
        dims=["year", "lat", "lon"],
        coords={"year": years, "lat": o_lat, "lon": o_lon},
    )
    return predictor, predictand, o_pattern


def test_apply_sign_convention_flips_negative_dominant_lobe():
    from deepscale.plotting.modes import _apply_sign_convention
    arr = np.array([[-3.0, 1.0], [0.5, -0.5]])
    flipped, sign = _apply_sign_convention(arr)
    assert sign == -1.0
    np.testing.assert_array_equal(flipped, -arr)
    # After flip, the dominant lobe is positive.
    assert flipped.flat[int(np.nanargmax(np.abs(flipped)))] > 0


def test_apply_sign_convention_keeps_positive_dominant_lobe():
    from deepscale.plotting.modes import _apply_sign_convention
    arr = np.array([[3.0, -1.0], [0.5, -0.5]])
    out, sign = _apply_sign_convention(arr)
    assert sign == 1.0
    np.testing.assert_array_equal(out, arr)


def test_apply_sign_convention_handles_all_nan():
    from deepscale.plotting.modes import _apply_sign_convention
    arr = np.full((2, 2), np.nan)
    out, sign = _apply_sign_convention(arr)
    assert sign == 1.0
    assert np.all(np.isnan(out))


def _fit_cca_for_mode_plots():
    """Helper: fit CCAMethod on the dual-grid fixture for plotting tests."""
    from deepscale.methods.cca import CCAMethod
    predictor, predictand, _ = _build_dual_grid_fixture()
    m = CCAMethod(n_modes=3, x_eof_modes=4, y_eof_modes=4)
    m.fit(predictor, predictand)
    return m, predictor, predictand


def test_plot_eof_modes_predictor_returns_figure():
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt
    from deepscale.plotting.modes import plot_eof_modes
    m, _, _ = _fit_cca_for_mode_plots()
    fig = plot_eof_modes(m, kind="predictor", n_modes=3)
    assert fig is not None
    # 3 mode panels (plus colorbars are extra axes)
    map_axes = [ax for ax in fig.axes if hasattr(ax, "coastlines") and ax.get_visible()]
    assert len(map_axes) == 3
    plt.close(fig)


def test_plot_eof_modes_predictand_returns_figure():
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt
    from deepscale.plotting.modes import plot_eof_modes
    m, _, _ = _fit_cca_for_mode_plots()
    fig = plot_eof_modes(m, kind="predictand", n_modes=2)
    map_axes = [ax for ax in fig.axes if hasattr(ax, "coastlines") and ax.get_visible()]
    assert len(map_axes) == 2
    plt.close(fig)


def test_plot_eof_modes_invalid_kind_raises():
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    from deepscale.plotting.modes import plot_eof_modes
    m, _, _ = _fit_cca_for_mode_plots()
    with pytest.raises(ValueError, match="kind"):
        plot_eof_modes(m, kind="bogus")


def test_plot_eof_modes_caps_n_modes_at_available():
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt
    from deepscale.plotting.modes import plot_eof_modes
    m, _, _ = _fit_cca_for_mode_plots()
    # Ask for more modes than were fitted; should silently cap.
    fig = plot_eof_modes(m, kind="predictor", n_modes=99)
    map_axes = [ax for ax in fig.axes if hasattr(ax, "coastlines") and ax.get_visible()]
    assert len(map_axes) == m.eofx_.shape[1]
    plt.close(fig)


def test_plot_eof_modes_title_includes_variance_fraction():
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt
    from deepscale.plotting.modes import plot_eof_modes
    m, _, _ = _fit_cca_for_mode_plots()
    fig = plot_eof_modes(m, kind="predictor", n_modes=2)
    titles = [
        ax.get_title() for ax in fig.axes
        if hasattr(ax, "coastlines") and ax.get_visible()
    ]
    assert all("EOF" in t for t in titles)
    assert all("var" in t for t in titles)
    plt.close(fig)


def test_plot_cca_modes_returns_paired_grid():
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt
    from deepscale.plotting.modes import plot_cca_modes
    m, _, _ = _fit_cca_for_mode_plots()
    fig = plot_cca_modes(m, n_modes=2)
    map_axes = [ax for ax in fig.axes if hasattr(ax, "coastlines") and ax.get_visible()]
    # 2 modes x (predictor + predictand) = 4 map panels
    assert len(map_axes) == 4
    plt.close(fig)


def test_plot_cca_modes_title_includes_canonical_correlation():
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt
    from deepscale.plotting.modes import plot_cca_modes
    m, _, _ = _fit_cca_for_mode_plots()
    fig = plot_cca_modes(m, n_modes=1)
    titles = [
        ax.get_title() for ax in fig.axes
        if hasattr(ax, "coastlines") and ax.get_visible()
    ]
    assert any("predictor" in t for t in titles)
    assert any("predictand" in t for t in titles)
    assert all("r=" in t for t in titles)
    plt.close(fig)


def test_plot_cca_modes_caps_at_available_modes():
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt
    from deepscale.plotting.modes import plot_cca_modes
    m, _, _ = _fit_cca_for_mode_plots()
    fig = plot_cca_modes(m, n_modes=99)
    map_axes = [ax for ax in fig.axes if hasattr(ax, "coastlines") and ax.get_visible()]
    assert len(map_axes) == 2 * m.ncc_
    plt.close(fig)


def test_mode_plots_dual_grid_integration(tmp_path):
    """Integration: fit CCA on the dual-grid fixture and render both mode plots to disk."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt
    from deepscale.methods.cca import CCAMethod
    from deepscale.plotting.modes import plot_eof_modes, plot_cca_modes

    predictor, predictand, _ = _build_dual_grid_fixture()
    m = CCAMethod(n_modes=3, x_eof_modes=4, y_eof_modes=4)
    m.fit(predictor, predictand)

    eof_path = tmp_path / "eof_predictor.png"
    cca_path = tmp_path / "cca_modes.png"
    fig_eof = plot_eof_modes(m, kind="predictor", n_modes=3)
    fig_eof.savefig(eof_path, dpi=80)
    plt.close(fig_eof)
    fig_cca = plot_cca_modes(m, n_modes=2)
    fig_cca.savefig(cca_path, dpi=80)
    plt.close(fig_cca)

    # Both files exist and are non-trivially sized (a blank figure is much smaller).
    assert eof_path.exists() and eof_path.stat().st_size > 5000
    assert cca_path.exists() and cca_path.stat().st_size > 5000


def test_plot_cca_modes_pair_shares_sign_convention():
    """Predictor and predictand of a CCA pair should be flipped together."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    from deepscale.plotting.modes import (
        plot_cca_modes, _apply_sign_convention, _reconstruct_spatial,
    )
    import matplotlib.pyplot as plt
    m, _, _ = _fit_cca_for_mode_plots()

    # Manually compute what the locked-sign predictor / predictand patterns should be
    # for mode 0, then check the rendered colour-meshes' raw arrays match.
    p_raw = _reconstruct_spatial(
        (m.eofx_ @ m.s_.T)[:, 0], m.x_valid_, m.predictor_shape_
    )
    o_raw = _reconstruct_spatial(
        (m.eofy_ @ m.r_)[:, 0], m.y_valid_, m.predictand_shape_
    )
    p_signed, sign = _apply_sign_convention(p_raw)
    o_signed = o_raw * sign

    fig = plot_cca_modes(m, n_modes=1)
    map_axes = [ax for ax in fig.axes if hasattr(ax, "coastlines") and ax.get_visible()]
    p_mesh = map_axes[0].collections[0].get_array().reshape(m.predictor_shape_)
    o_mesh = map_axes[1].collections[0].get_array().reshape(m.predictand_shape_)
    np.testing.assert_allclose(np.asarray(p_mesh), p_signed, equal_nan=True)
    np.testing.assert_allclose(np.asarray(o_mesh), o_signed, equal_nan=True)
    plt.close(fig)

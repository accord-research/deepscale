"""deepscale.plotting TercileStyle + styled tercile / comparison plots.

Added on the `acmad` branch. Rendering runs under the Agg backend; country/lakes
overlays are best-effort and absent here, which the renderer tolerates.
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
import xarray as xr

import deepscale
from deepscale.plotting import TercileStyle
from deepscale.plotting.styled import _binned_rgb


LAT = np.array([-2.0, 0.0, 2.0])
LON = np.array([10.0, 12.0])

STYLE = TercileStyle(
    below_colors=["#fcf3c8", "#f8d808", "#d49e00"],
    normal_colors=["#eefcff", "#d6efef", "#c7e8e8"],
    above_colors=["#c9f5c2", "#34c818", "#188c08"],
    prob_bins=[33.3, 45, 60, 100.01],          # 3 intervals -> 3 colours
)


def _probs(fill):
    a = np.empty((3, len(LAT), len(LON)))
    for k in range(3):
        a[k] = fill[k]
    return xr.DataArray(a, dims=("tercile", "lat", "lon"),
                        coords={"tercile": [0, 1, 2], "lat": LAT, "lon": LON})


def test_style_validates_ramp_and_bin_lengths():
    with pytest.raises(ValueError):
        TercileStyle(below_colors=["#000"], normal_colors=["#111", "#222"],
                     above_colors=["#333"], prob_bins=[33.3, 100])
    with pytest.raises(ValueError):
        TercileStyle(below_colors=["#000"], normal_colors=["#111"],
                     above_colors=["#222"], prob_bins=[33.3, 50, 100])  # too many bins


def test_binned_rgb_picks_category_and_band():
    import matplotlib.colors as mcolors
    # Dominant = above (cat 2) at 65% -> third band -> above_colors[2].
    probs = _probs((0.15, 0.20, 0.65))
    rgb, _ = _binned_rgb(probs.values, LAT, LON, STYLE)
    assert np.allclose(rgb[0, 0], mcolors.to_rgb("#188c08"))


def test_dry_mask_paints_dry_color():
    import matplotlib.colors as mcolors
    dry = xr.DataArray(np.array([[True, False], [False, False], [False, False]]),
                       dims=("lat", "lon"), coords={"lat": LAT, "lon": LON})
    style = TercileStyle(STYLE.below_colors, STYLE.normal_colors, STYLE.above_colors,
                         STYLE.prob_bins, dry_mask=dry, dry_color="#bebebe")
    probs = _probs((0.6, 0.2, 0.2))
    rgb, _ = _binned_rgb(probs.values, LAT, LON, style)
    # lat is ascending [-2,0,2]; imshow flips to descending, so lat=-2 is last row.
    assert np.allclose(rgb[-1, 0], mcolors.to_rgb("#bebebe"))


def test_plot_terciles_with_style_returns_fig():
    fig = deepscale.plot_terciles(_probs((0.6, 0.25, 0.15)), style=STYLE, title="t")
    assert fig is not None


def test_plot_tercile_comparison_returns_fig_and_image():
    import matplotlib.pyplot as plt
    ours = _probs((0.6, 0.25, 0.15))
    ref = _probs((0.2, 0.3, 0.5))
    fig, axes = plt.subplots(1, 3)
    out_fig, diff_im = deepscale.plot_tercile_comparison(
        ours, ref, style=STYLE, axes=axes, diff_limit=40,
        labels=("ours", "ref", "diff"))
    assert out_fig is not None and diff_im is not None

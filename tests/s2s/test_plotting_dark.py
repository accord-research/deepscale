"""Unit tests for the dashboard dark matplotlib theme."""
import matplotlib
import matplotlib.pyplot as plt


def test_apply_dark_theme_sets_dark_rcparams():
    import scripts.s2s.plotting as plotting  # import applies the theme
    plotting.apply_dark_theme()
    assert matplotlib.rcParams["figure.facecolor"] == plotting._DARK_BG
    assert matplotlib.rcParams["savefig.facecolor"] == plotting._DARK_BG
    assert matplotlib.rcParams["axes.facecolor"] == plotting._DARK_BG
    assert matplotlib.rcParams["text.color"] == plotting._DARK_FG


def test_metrics_panel_figure_is_dark_and_saves(tmp_path):
    from scripts.s2s.plotting import metrics_panel
    fig = metrics_panel([], country="kenya")  # empty -> single-axes placeholder fig
    r, g, b, _ = fig.get_facecolor()
    assert (round(r, 2), round(g, 2), round(b, 2)) == (0.1, 0.1, 0.1)  # #1a1a1a
    out = tmp_path / "metrics.png"
    fig.savefig(out)
    assert out.stat().st_size > 0
    # Top-left pixel of the saved PNG is the dark facecolor, not white.
    px = plt.imread(out)[0, 0]
    assert px[0] < 0.2 and px[1] < 0.2 and px[2] < 0.2


def test_metrics_panel_uses_single_figure_legend():
    from scripts.s2s.plotting import metrics_panel
    scores = [
        {"method": "raw",  "target_dekad": "2026-05-21", "acc": 0.10, "rmse": 1.0, "bias": 0.00, "rpss": 0.00},
        {"method": "bcsd", "target_dekad": "2026-05-21", "acc": 0.30, "rmse": 0.8, "bias": -0.10, "rpss": 0.05},
        {"method": "raw",  "target_dekad": "2026-06-01", "acc": 0.15, "rmse": 0.9, "bias": 0.02, "rpss": 0.02},
        {"method": "bcsd", "target_dekad": "2026-06-01", "acc": 0.35, "rmse": 0.7, "bias": -0.05, "rpss": 0.08},
    ]
    fig = metrics_panel(scores, country="kenya")
    # Exactly one legend, attached to the figure (not to any individual axes).
    assert fig.legends, "expected a figure-level legend"
    assert all(ax.get_legend() is None for ax in fig.axes), "no per-axes legends expected"

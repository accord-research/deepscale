"""WMO-SVSLRF PDF composition.

Composes the SkillReport into a PDF following WMO-No. 1246 layout
conventions:

  1. Title page (metadata).
  2. Mandatory triplet summary (RPSS, ROC areas, reliability scalar).
  3. Spatial maps grid (one page).
  4. ROC curves page (if report.diagrams["roc"]).
  5. Reliability diagram page (if report.diagrams["reliability"]).
  6. Secondary metrics table (everything not in the mandatory set).
"""

from .._optional import require_optional


_MANDATORY = ("rpss", "roc_bn", "roc_nn", "roc_an", "reliability")


def render(report, path):
    """Render a SkillReport to `path` as a WMO-SVSLRF-formatted PDF."""
    require_optional("matplotlib", "pip install deepscale[plotting]")
    from matplotlib.backends.backend_pdf import PdfPages

    from . import _pages

    mandatory_scores = {k: v for k, v in report.scores.items() if k in _MANDATORY}
    secondary_scores = {k: v for k, v in report.scores.items() if k not in _MANDATORY}

    # Filter spatial to DataArrays only — roc/reliability store scalars via
    # scores.update(result) when spatial=True, which would confuse map_grid_page.
    import xarray as xr
    spatial_maps = {k: v for k, v in report.spatial.items()
                    if isinstance(v, xr.DataArray)}

    with PdfPages(str(path)) as pdf:
        _pages.title_page(
            pdf,
            title="Verification report",
            subtitle="WMO-SVSLRF format",
            metadata=report.metadata,
        )
        _pages.scalar_table_page(pdf, mandatory_scores, title="Mandatory triplet")

        if spatial_maps:
            _pages.map_grid_page(pdf, spatial_maps)

        roc = report.diagrams.get("roc")
        if roc:
            _pages.roc_page(pdf, roc)

        rel = report.diagrams.get("reliability")
        if rel:
            _pages.reliability_page(pdf, rel)

        if secondary_scores:
            _pages.scalar_table_page(pdf, secondary_scores, title="Secondary metrics")

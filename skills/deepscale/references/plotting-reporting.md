# Plotting and reporting

Plotting/reporting live in `deepscale.plotting` and `deepscale.reporting`. Importing the subpackage does **not** load matplotlib/cartopy; each function gates on its optional deps and raises a clear `ImportError` with a `pip install accord-deepscale[plotting]` hint. Basemaps use cartopy when available, else geopandas with cached Natural Earth shapefiles (`~/.local/share/cartopy/shapefiles/natural_earth/...`), else plain axes.

## Top-level re-exports

```python
ds.plot_terciles(...)            # = plotting.forecasts.plot_tercile_forecast
ds.plot_field(...)
ds.plot_tercile_comparison(...)
```

## Forecast maps

```python
plot_tercile_forecast(pr_fcst, *, style=None, ax=None, title=None,
                      variable_kind="precip", legend=True)
```
IRI-style dominant-tercile map from a `(tercile, lat, lon)` array. `variable_kind` ∈ `{"precip", "temp"}` selects the palette; color intensity scales with `(max_prob − 1/3)`, saturating at 0.37.

```python
plot_field(field, *, style=None, ax=None, cmap="RdBu_r", vmin=None, vmax=None,
           center=None, title=None, grey_dry=True) -> mappable
plot_tercile_comparison(forecast, reference, *, style=None, axes=None,
                        labels=("forecast", "reference", "difference"),
                        diff_cmap="BrBG", diff_limit=40.0, title=None) -> (axes, diff_mappable)
    # "tilt" difference = P(above) − P(below), percentage points
plot_deterministic_forecast(det_fcst, *, ax=None, title=None, cmap="RdBu_r", center=None)
plot_exceedance_probability(exceedance_prob, threshold, *, ax=None)
plot_flex_pdf(fcst_mu, fcst_scale, climo_mu, climo_scale, *, location, ax=None)  # location=(lon, lat)
```

## Diagnostics

```python
plot_domains(predictor_extent, predictand_extent, *, ax=None, title=None)  # extents (lat_s, lat_n, lon_w, lon_e); cartopy
plot_skill_maps(skill_report, metric_names, *, ncols=3)   # grid from SkillReport.spatial; fixed cmap/range per metric; cartopy
plot_reliability_diagram(forecast, obs, *, n_bins=5, ax=None, title=None)  # BN/NN/AN curves
plot_eof_modes(cca_fit, kind="predictor", n_modes=3, *, ncols=3)
plot_cca_modes(cca_fit, n_modes=3)
```

## `TercileStyle`

```python
@dataclass
class TercileStyle:
    below_colors: list[str]; normal_colors: list[str]; above_colors: list[str]
    prob_bins: list[float]        # percent edges; len == n_colors + 1
    dry_mask = None; dry_color = "#bebebe"
    clip_to = None                # list of country NAMEs or a shapely geometry
    lakes = False; lake_color = "#78b8f8"
    nodata_color = "#ffffff"
    extent = None                 # (lon_w, lon_e, lat_s, lat_n)
```

Pass as `style=` to the tercile plotting functions to control palette, probability binning, dry masking, country clipping, lakes, and extent.

## Reports

```python
report = ds.skill(cv_terc, obs, metrics="svslrf", spatial=True)
report.to_pdf("verification.pdf", style="svslrf")
```

`SkillReport.to_pdf` renders a WMO-SVSLRF verification PDF via `reporting.svslrf.render(report, path)`: cover + mandatory triplet (RPSS, ROC BN/NN/AN, reliability), ROC + reliability diagram page, spatial-map grid, member-contributions page, and a secondary-metrics table.

```python
cmp = ds.skill_compare({"bcsd": fc1, "cca": fc2}, obs, metrics=["rpss"], spatial=True)
cmp.to_table()                    # DataFrame: methods x metrics
cmp.to_heatmap("heatmap.png")     # RdBu, vmin=-1, vmax=1
cmp.to_pdf("comparison.pdf", spatial_maps=True)
```

## Export

```python
ds.write_terciles(tercile_fc, "forecast.nc", title="MAM precip", method="cca")
    # NetCDF with below/normal/above percent (0-100) variables, float32, _FillValue=-9999.0
report.to_geotiff("rpss.tif", "rpss")   # one spatial metric as EPSG:4326 GeoTIFF (rioxarray)
ds.tercile_mae(candidate_probs, "reference.nc")  # MAE in percentage points vs a reference
```

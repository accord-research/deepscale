# Downscaling Method Validation Report

Date: 2026-06-12

## Overview

This validation study evaluates whether the current DeepScale statistical
downscaling methods behave correctly on a controlled precipitation downscaling
problem. The focus for this version of the report is algorithm verification,
not operational forecast-skill ranking. To keep the interpretation clean, the
report uses CHIRPS-derived coarse/fine experiments and independent method
references where they are available.

Here, coarse/fine means paired inputs made from the same CHIRPS fields: the
fine grid is treated as the answer, and a blurred lower-resolution version is
treated as the pseudo-model predictor.

The practical goal is to give the local development team a clear read on which
methods are already stable, which methods need targeted follow-up, and what
kind of evidence supports each conclusion.

The downscaling methods evaluated were:

- `bcsd`: bias correction and spatial disaggregation
- `cca`: canonical correlation analysis
- `delta`: applies the climatological difference, or change signal, to the
  observed grid
- `dqm`: detrended quantile mapping
- `qm`: quantile mapping

The report also includes `climatology` as a historical observed baseline.

## Executive Summary

The findings below are scoped to the controlled CHIRPS coarse/fine fixtures and
the external reference checks in this report.

- Every active DeepScale method beats the historical-mean climatology baseline
  on both controlled fixtures, which is the first implementation sanity check.
- Delta has the cleanest reference result: DeepScale matches both the direct
  formula and `python-cmethods` implementations on the same held-out folds.
- BCSD has strong support from two directions: it lands close to the
  scikit-downscale adapter on the CHIRPS fixtures, and it reproduces the
  behavior of a published SEAS5-BCSD research product over the Nile D03 basin.
- CCA runs end to end and lands near PyCPT, but it is weaker on this
  reconstruction task and should remain under targeted parity review.
- QM and DQM are stable, but the `xsdba` references score better on these
  fixtures because they use an additive-adjustment convention that the
  CHIRPS-derived setup favors. The recommended follow-up is an explicit
  additive-factor option rather than a silent default change.
- The bilinear interpolation baseline is an important caution: on this
  synthetic coarse/fine setup, simple regridding is competitive with several
  methods. Passing this test confirms sensible behavior; it does not establish
  operational forecast skill.

## Methodology

The validation starts with CHIRPS rainfall data and treats it as the answer we
want the methods to recover. We then make a lower-resolution version of the
same data to stand in for a coarse climate-model forecast. This gives us a
controlled test: because the coarse input was made from CHIRPS, we know what a
good downscaled result should look like, and we can judge each method against
the original higher-resolution precipitation grid.

The primary CHIRPS data are fetched through Rosetta product
`obs/chirps-v3-monthly`, variable `precip`. In the Rosetta catalog, this is the
native UCSB CHIRPS v3 monthly product at roughly `0.05° x 0.05°` spatial
resolution. The harness averages those monthly fields to one gridded
precipitation layer per year or season.

The report focuses on two fixtures:

| Fixture | Rosetta product | Years | Months | Fine grid | Coarse pseudo-model grid |
|---|---|---|---|---|---|
| Texas | `obs/chirps-v3-monthly` | 1991-2020 | all months | `100 x 100` | `10 x 10`, 3 members |
| Ethiopia FMA | `obs/chirps-v3-monthly` | 1991-2020 | February-April | `240 x 300` | `24 x 30`, 3 members |

<div class="region-grid">
  <figure>
    <img src="figures/region_texas_locator_report.png" alt="Texas validation domain locator map" />
    <figcaption>Texas validation domain</figcaption>
  </figure>
  <figure>
    <img src="figures/region_ethiopia_locator_report.png" alt="Ethiopia validation domain locator map" />
    <figcaption>Ethiopia FMA validation domain</figcaption>
  </figure>
</div>

From each fixture, the harness builds the controlled test in three steps. In
this report, a gridded precipitation layer means one map-like data object: one
rainfall value at each latitude/longitude cell for a specific year and season.
A 1991-2020 fixture therefore contains 30 gridded layers, one for each year.

1. Collapse to one gridded precipitation layer per year. Texas averages all
   twelve months into a single annual precipitation grid for each year. The
   Ethiopia fixtures average only their season's months, such as February
   through April for FMA, so Ethiopia FMA 1991 is one grid representing mean
   FMA precipitation in 1991.
2. Coarsen each yearly grid into a stand-in model forecast. Averaging it down
   by a factor of ten in each direction turns native `0.05°` CHIRPS into a
   roughly `0.5°` pseudo-model grid. Edges that do not divide evenly by ten are
   trimmed first. To exercise ensemble-aware code paths without treating this
   as a real ensemble forecast, the harness creates three deterministic members
   from the same coarse grid: one copy scaled by `0.99`, one unchanged copy,
   and one copy scaled by `1.01`. These members should not be interpreted as
   independent forecasts.
3. Score it leave-one-year-out. Each method is trained on every year but one
   and asked to predict the year that was held out, repeated for all years (30
   folds for Texas and Ethiopia FMA). Holding out the target year
   keeps a method from looking good simply because it trained on the answer. The
   held-out predictions are then combined and summarized with bias, mean
   absolute error (MAE), root mean squared error (RMSE), and spatial
   correlation.

The harness also records extra sanity checks beyond the headline scores. For
each method, it compares the prediction with the CHIRPS answer using simple
summary statistics: minimum, mean, maximum, standard deviation, and several
percentiles. For example, `p90` is the 90th percentile: 90% of values are below
that number and 10% are above it. These checks help identify cases where a
method has the right broad pattern but produces rainfall values that are too
flat, too extreme, or shifted too wet or dry. A no-training bilinear
interpolation baseline is always scored, so every method can be compared with a
simple "just regrid the coarse data" approach. The validation also checks
whether downscaling preserves the original coarse-grid signal; that is important
for bias-correction methods because a correction should not erase the forecast
information that was already present in the coarse input.

The compact diagnostics below surface the most useful of those checks. `P90
bias` is the prediction's 90th percentile minus CHIRPS' 90th percentile, in
mm/day; negative values mean the method underrepresents high-rainfall values.
For scale, CHIRPS mean rainfall is about `2.79` mm/day in the Texas fixture and
`1.41` mm/day in Ethiopia FMA, so the climatology rows' P90 biases are a clear
heavy-rainfall flattening signal. `Coarse-signal change` is downscaled
correlation minus coarse-input correlation, so values closer to zero preserve
more of the original pseudo-model signal.

| Fixture | Method | Implementation | RMSE | Corr | P90 bias | Coarse-signal change |
|---|---|---|---:|---:|---:|---:|
| Texas | `bilinear` | `interpolation` | 0.1333 | 0.9870 | -0.0312 | -0.0076 |
| Texas | `delta` | `deepscale` | 0.1080 | 0.9915 | -0.0229 | -0.0031 |
| Texas | `bcsd` | `deepscale` | 0.1839 | 0.9783 | -0.0792 | -0.0163 |
| Texas | `cca` | `deepscale` | 0.2857 | 0.9437 | -0.0650 | -0.0509 |
| Texas | `qm` | `deepscale` | 0.1897 | 0.9747 | -0.0552 | -0.0199 |
| Texas | `dqm` | `deepscale` | 0.2019 | 0.9714 | -0.0597 | -0.0232 |
| Texas | `qm` | `xsdba` | 0.1414 | 0.9853 | -0.0364 | -0.0093 |
| Texas | `dqm` | `xsdba` | 0.1418 | 0.9855 | -0.0583 | -0.0090 |
| Texas | `climatology` | `deepscale` | 0.6607 | 0.6442 | -0.3907 | -0.3503 |
| Ethiopia FMA | `bilinear` | `interpolation` | 0.2989 | 0.9747 | -0.0758 | -0.0142 |
| Ethiopia FMA | `delta` | `deepscale` | 0.2047 | 0.9881 | -0.0262 | -0.0008 |
| Ethiopia FMA | `bcsd` | `deepscale` | 0.2646 | 0.9810 | -0.1023 | -0.0079 |
| Ethiopia FMA | `cca` | `deepscale` | 0.5105 | 0.9211 | -0.1218 | -0.0679 |
| Ethiopia FMA | `qm` | `deepscale` | 0.3405 | 0.9671 | -0.0718 | -0.0218 |
| Ethiopia FMA | `dqm` | `deepscale` | 0.3454 | 0.9662 | -0.0749 | -0.0228 |
| Ethiopia FMA | `qm` | `xsdba` | 0.3170 | 0.9715 | -0.0333 | -0.0175 |
| Ethiopia FMA | `dqm` | `xsdba` | 0.3177 | 0.9714 | -0.0429 | -0.0176 |
| Ethiopia FMA | `climatology` | `deepscale` | 0.7633 | 0.8131 | -0.2962 | -0.1758 |

By design this is an easy test. The predictor is just a blurred copy of the
CHIRPS truth, so it strongly favors methods that rebuild local detail from a
field already related to the answer. That is the point: the goal here is to
confirm the algorithms behave sensibly under known conditions, not to rank
operational skill. Passing is necessary but not sufficient. A method that fails
here is unlikely to be trusted anywhere, while a method that passes still has to
be re-checked on production-like inputs before anyone calls it skillful.

Public method and reference sources used in this validation:

- IRI CPT: https://iri.columbia.edu/our-expertise/climate/tools/cpt/
- PyCPT installation: https://iri-pycpt.github.io/installation/
- scikit-downscale API: https://scikit-downscale.readthedocs.io/en/latest/api.html
- xsdba documentation: https://xsdba.readthedocs.io/en/stable/
- python-cmethods documentation: https://python-cmethods.readthedocs.io/en/latest/methods.html
- Wood et al. (2004), `Hydrologic Implications of Dynamical and Statistical Approaches to Downscaling Climate Model Outputs`, Climatic Change.
- Lorenz et al. (2021), `Bias-corrected and spatially disaggregated seasonal forecasts: a long-term reference forecast product for the water sector in semi-arid regions`, Earth System Science Data, https://essd.copernicus.org/articles/13/2701/2021/
- WDCC SaWaM D03 SEAS5-BCSD data publication: https://doi.org/10.26050/WDCC/SaWaM_D03_SEAS5_BCSD
- ECMWF-S2S4AFRICA: https://github.com/alecjong-lab/ECMWF-S2S4AFRICA

## BCSD

### Method Overview

BCSD, or bias correction and spatial disaggregation, first corrects the coarse
predictor distribution and then reconstructs fine-grid spatial detail from the
observed climatology. The validation harness also keeps a small direct
implementation of the classic Wood et al. (2004) BCSD workflow: bias-correct
the coarse field, interpolate it to the fine grid, then restore fine-scale
spatial detail from the observed climatology. BCSD is widely used in climate
downscaling and hydrologic applications because it combines distributional bias
correction with a simple spatial refinement step.

The external comparison uses scikit-downscale's
`skdownscale.pointwise_models.BcsdPrecipitation`, which the package describes
as a classic BCSD model for precipitation. Its API is pointwise and
time-series oriented: it fits a one-dimensional coarse predictor series against
a one-dimensional reference series, then predicts a corrected series. To compare
it with DeepScale's gridded leave-one-year-out workflow, the validation harness
fits `BcsdPrecipitation(return_anoms=False)` at each coarse grid cell, predicts
the full pseudo-model time series, selects the held-out year, and then applies
the same fine-grid detail reconstruction as the direct BCSD implementation.
The table below focuses on DeepScale and this scikit-downscale adapter. The
direct implementation is shared validation logic behind that comparison, not a
third truth dataset or an external forecast product.

### Validation Results

| Fixture | Implementation | RMSE | Corr |
|---|---|---:|---:|
| Texas | DeepScale | 0.1839 | 0.9783 |
| Texas | scikit | 0.2167 | 0.9722 |
| Ethiopia FMA | DeepScale | 0.2646 | 0.9810 |
| Ethiopia FMA | scikit | 0.2715 | 0.9800 |

The scikit-downscale comparison is most useful as an implementation check. It
lands very close to DeepScale on both fixtures, with slightly higher RMSE
against CHIRPS. That pattern is what we want to see: an independently
maintained BCSD implementation follows the same behavior, while the small
differences are consistent with the adapter needed to translate
scikit-downscale's pointwise series API into this gridded leave-one-year-out
validation.

The maps below show one example held-out year, 2020. Each map uses the same
three-panel format: CHIRPS observed, prediction, and prediction minus CHIRPS.
The table above summarizes performance across all validation years.

![BCSD Texas DeepScale](figures/bcsd_deepscale_maps_long_texas_1991_2020.png)

![BCSD Texas scikit](figures/bcsd_scikit_maps_long_texas_1991_2020.png)

![BCSD Ethiopia DeepScale](figures/bcsd_deepscale_maps_ethiopia_fma_1991_2020.png)

![BCSD Ethiopia scikit](figures/bcsd_scikit_maps_ethiopia_fma_1991_2020.png)

### Interpretation

BCSD passes the current functional-reference check and behaves strongly on the
CHIRPS coarse/fine fixtures. It consistently beats climatology. Because BCSD
depends on empirical distributional relationships, it should benefit from
longer training records, though this report's fixed 30-year leave-one-year-out
design does not isolate that sample-size effect directly.

### Research-Paper Replication: Lorenz et al. SEAS5-BCSD

Lorenz et al. (2021) published a ready-made BCSD forecast dataset for part of
the Nile basin. The starting forecast data were ECMWF SEAS5 seasonal rainfall
forecasts. The reference data were ERA5-Land rainfall estimates. The authors
used BCSD to turn the coarse SEAS5 forecasts into finer monthly rainfall maps,
then published those maps through WDCC as the SaWaM D03 SEAS5-BCSD product.

Our replication asks whether DeepScale can reproduce that published product's
behavior. We downloaded the WDCC BCSD files, fetched matching raw SEAS5 and
ERA5-Land data through Rosetta, ran DeepScale BCSD on the same basin and
months, and compared all three products against ERA5-Land. This is a stronger
external check than the CHIRPS coarse/fine fixtures because the benchmark is an
independently published research dataset rather than a synthetic coarse/fine
test made only for this report. It is still not an exact source-code
replication: we compare against the published WDCC output, not the authors'
private processing code.

The replication has two layers:

1. `validation/lorenz_bcsd_benchmark.py` validates the downloaded WDCC
   `D03_BCSD_monthly_pr` files. It records the paper and dataset metadata,
   opens the local WDCC NetCDF issue files, confirms the variable/dimensions,
   plots the mean field and lead behavior, and reproduces the paper-style
   monthly bias/RMSE diagnostics against ERA5-Land.
2. `validation/lorenz_deepscale_bcsd_compare.py` runs a direct common-period
   comparison between raw Rosetta SEAS5, DeepScale BCSD, and the WDCC
   SEAS5-BCSD product. For each issue month and lead month, it aligns Rosetta
   `lead_time` 1-6 to WDCC time index 0-5, interpolates ERA5-Land to the WDCC
   0.1-degree grid, predicts DeepScale BCSD from the SEAS5 hindcast, and scores
   all fields against ERA5-Land.

The main common-period comparison uses 1993-2016, all 12 issue months, and
lead months 1-6. The table reports means across the 72 issue-month/lead cases.

| Product | Bias | MAE | RMSE | Corr |
|---|---:|---:|---:|---:|
| Raw SEAS5 | -0.0599 | 1.0632 | 2.1986 | 0.6855 |
| DeepScale BCSD | 0.0232 | 0.6238 | 1.1756 | 0.8624 |
| WDCC SEAS5-BCSD | 0.0800 | 0.6000 | 1.1516 | 0.8778 |
| DeepScale BCSD vs WDCC BCSD | -0.0568 | 0.2163 | 0.3955 | 0.9817 |

The result verifies three important points. First, both BCSD products strongly
improve on raw SEAS5 against ERA5-Land: DeepScale reduces mean RMSE from
`2.1986` to `1.1756`, while the WDCC product scores `1.1516`. Second,
DeepScale lands close to the published paper product across all issue months
and leads, with a DeepScale-vs-WDCC correlation of `0.9817`. Third, the
remaining gap is plausible for an implementation-level comparison: the WDCC
product is the paper's full production archive, while DeepScale is trained
through the local validation adapter on Rosetta inputs and aligned to the WDCC
grid for scoring.

The earlier 1981-2016 comparison tells the same story on the longer locally
available span: raw SEAS5 RMSE `2.2294`, DeepScale BCSD RMSE `1.2368`, WDCC
BCSD RMSE `1.1828`, and DeepScale-vs-WDCC correlation `0.9840`.

![Lorenz SEAS5-BCSD benchmark maps](figures/lorenz_bcsd_benchmark_maps_d03_monthly_pr_era5_land.png)

![Lorenz paper-style lead diagnostics](figures/lorenz_bcsd_benchmark_paper_metrics_d03_monthly_pr_era5_land.png)

![DeepScale vs WDCC BCSD comparison](figures/lorenz_deepscale_bcsd_compare_d03_1993_2016.png)

![DeepScale vs WDCC BCSD lead heatmaps](figures/lorenz_deepscale_bcsd_compare_heatmaps_d03_1993_2016.png)

Interpretation: the Lorenz/WDCC experiment is the strongest BCSD evidence in
this report so far. It shows DeepScale BCSD is not only internally sensible and
close to scikit-downscale on controlled CHIRPS coarse/fine fixtures; it also
reproduces the behavior of an independently published SEAS5-BCSD research
product on the same basin, reference dataset, issue months, and lead-month
structure.

## CCA

### Method Overview

Canonical correlation analysis learns coupled spatial patterns between a
predictor field and an observed target field. In climate prediction it is often
used for model output statistics, especially when the goal is to extract a
small number of large-scale modes that explain predictable regional variation.
CCA can be powerful when the predictor-target relationship is well represented
by a low-dimensional linear structure, but it is sensitive to mode selection,
crossvalidation choices, and output conventions.

The external comparison uses IRI CPT through PyCPT/CPT diagnostic workflows.
CPT is the most relevant reference because it is the long-standing operational
tool for CCA-based climate prediction workflows. The validation here compares
DeepScale's leave-one-year-out CCA against PyCPT/CPT on CHIRPS-derived Texas
and Ethiopia fixtures. As with the BCSD reference, CPT is not a separate truth
dataset; it is a functional reference for expected CCA behavior and output
conventions.

### Validation Results

The PyCPT rows should be read as method-reference checks rather than separate
independent benchmarks.

| Fixture | Implementation | RMSE | Corr |
|---|---|---:|---:|
| Texas | DeepScale | 0.2857 | 0.9437 |
| Texas | PyCPT | 0.3066 | 0.9353 |
| Ethiopia FMA | DeepScale | 0.5105 | 0.9211 |
| Ethiopia FMA | PyCPT | 0.5542 | 0.9067 |

The PyCPT comparison is most useful as a parity diagnostic. PyCPT and
DeepScale land in the same performance neighborhood on both fixtures. That
supports the broad implementation shape, but it does not close the question,
because CCA has several valid sign, scaling, and anomaly conventions.

The maps below show one example held-out year for each implementation. Each map
uses the same three-panel format: CHIRPS observed, prediction, and prediction
minus CHIRPS. The table above summarizes performance across all validation
years.

![CCA Texas DeepScale](figures/pycpt_cca_maps_latest_year.png)

![CCA Texas PyCPT](figures/pycpt_cca_pycpt_maps_latest_year.png)

![CCA Ethiopia DeepScale](figures/pycpt_cca_maps_ethiopia_fma_1991_2020.png)

![CCA Ethiopia PyCPT](figures/pycpt_cca_pycpt_maps_ethiopia_fma_1991_2020.png)

### Interpretation

CCA passes the current functional smoke test: it runs end to end, produces
plausible held-out fields, and lands near PyCPT on the CHIRPS parity fixtures.
It is weaker than Delta and BCSD on the controlled CHIRPS coarse/fine
reconstruction task, which is expected because this fixture strongly favors
methods that restore blurred local detail from an already-related field.

## Delta Baseline

### Method Overview

Delta baseline applies a simple climatological correction: it estimates the
difference between the coarse predictor climatology and the observed
climatology, then transfers the anomaly or change signal onto the observed
grid. This family of methods is commonly used in climate-impact and hydrologic
workflows because it is transparent, stable with limited training data, and
easy to interpret. Its limitation is also clear: it mainly adjusts mean
structure and does not learn complex nonlinear or spatial relationships.

The external package comparison uses `python-cmethods`, a climate bias
correction package that exposes a general `adjust(...)` function. The validation
harness calls that function with `method="delta_method"` and `kind="+"`, meaning
an additive delta adjustment. For each held-out year, it gives
`python-cmethods` the training-period observed climatology, the training-period
coarse climatology interpolated to the CHIRPS grid, and the held-out coarse
pseudo-forecast interpolated to that same grid. This is a close package-level
check for the simple additive Delta calculation.

### Validation Results

| Fixture | Implementation | RMSE | Corr |
|---|---|---:|---:|
| Texas | DeepScale delta | 0.1080 | 0.9915 |
| Texas | python-cmethods delta_method | 0.1080 | 0.9915 |
| Ethiopia FMA | DeepScale delta | 0.2047 | 0.9881 |
| Ethiopia FMA | python-cmethods delta_method | 0.2047 | 0.9881 |

The diagnostic below is intentionally Delta-only. It compares DeepScale's
Delta implementation with a direct formula implementation and the
`python-cmethods` reference on the same Texas and Ethiopia FMA fixtures. The
nearly overlapping bars are expected: for this simple baseline, the important
question is whether the independent implementations agree.

![Delta reference agreement](figures/delta_reference_diagnostics_1991_2020.png)

### Interpretation

Delta baseline is the cleanest implementation control in this report.
DeepScale delta matches the external `python-cmethods` reference on every
controlled fixture. Because the coarse predictor is constructed directly from
the high-resolution CHIRPS target, this experiment is especially favorable to
delta-style reconstruction. The result is therefore a strong behavior check,
not a claim that delta will always be best in operational forecasting.

## Detrended Quantile Mapping

### Method Overview

Detrended quantile mapping extends quantile mapping by separating a trend or
mean-change component from the distributional adjustment. It is commonly used
in climate-change bias adjustment when preserving a modeled change signal is
important. In this controlled validation, it is useful as a companion to QM:
it tests whether detrending changes the behavior of the empirical correction
and whether the same convention issues appear.

The external comparison again uses `xsdba`, this time with its detrended
quantile-mapping adjustment. The validation harness fits DeepScale empirical
DQM and xsdba DQM on the same leave-one-year-out folds, then scores the
held-out predictions against CHIRPS. As with QM, this is a package-level
functional reference rather than an independent forecast benchmark.

### Validation Results

| Fixture | Implementation | RMSE | Corr |
|---|---|---:|---:|
| Texas | DeepScale | 0.2019 | 0.9714 |
| Texas | xsdba | 0.1418 | 0.9855 |
| Ethiopia FMA | DeepScale | 0.3454 | 0.9662 |
| Ethiopia FMA | xsdba | 0.3177 | 0.9714 |

The xsdba comparison shows the same broad pattern as QM. xsdba DQM scores
better than DeepScale empirical DQM on both fixtures, especially on Texas. That
suggests detrending does not remove the convention difference seen in the
non-detrended QM results; it does not by itself show that xsdba DQM is more
skillful in operational forecasts.

The maps below show one example held-out year for each implementation. Each map
uses the same three-panel format: CHIRPS observed, prediction, and prediction
minus CHIRPS. The table above summarizes performance across all validation
years.

![DQM Texas DeepScale](figures/dqm_reference_maps_long_texas_1991_2020.png)

![DQM Texas xsdba](figures/dqm_xsdba_maps_long_texas_1991_2020.png)

![DQM Ethiopia DeepScale](figures/dqm_reference_maps_ethiopia_fma_1991_2020.png)

![DQM Ethiopia xsdba](figures/dqm_xsdba_maps_ethiopia_fma_1991_2020.png)

### Interpretation

DQM behaves consistently with the QM findings. DeepScale empirical DQM is
stable on the controlled fixtures, but the xsdba additive-adjustment convention
scores better here. Because this fixture is built by blurring CHIRPS, it
favors additive, delta-style reconstruction; the result should be read as a
convention difference amplified by this test, not as an operational skill
ranking. The recommended development path is to keep the current empirical
default for compatibility and add an explicit additive-factor convention for
both QM and DQM.

## Quantile Mapping

### Method Overview

Quantile mapping corrects a predictor by matching its cumulative distribution
to the observed cumulative distribution. It is widely used for precipitation
and temperature bias correction because it can adjust more than the mean: it
can also change spread, skew, and extremes. Its main risks are sample-size
sensitivity, extrapolation behavior, and convention differences between
implementations.

The external comparison uses `xsdba`, a bias-adjustment package developed for
climate data workflows. The validation harness trains `xsdba` empirical
quantile mapping on the same leave-one-year-out folds used by DeepScale, then
adjusts the held-out coarse pseudo-forecast and interpolates it back to the
CHIRPS grid. This makes `xsdba` a useful functional reference for the empirical
QM family. It is not an independent forecast product; it is a package-level
comparison for how the quantile-mapping correction is applied.

### Validation Results

| Fixture | Implementation | RMSE | Corr |
|---|---|---:|---:|
| Texas | DeepScale | 0.1897 | 0.9747 |
| Texas | xsdba | 0.1414 | 0.9853 |
| Ethiopia FMA | DeepScale | 0.3405 | 0.9671 |
| Ethiopia FMA | xsdba | 0.3170 | 0.9715 |

The xsdba comparison is most useful as a convention check. It scores better
than DeepScale empirical QM on both fixtures, with the largest improvement on
Texas. The gap points to a difference in the correction convention rather than
a failure to run quantile mapping at all: DeepScale applies direct CDF mapping,
while xsdba applies additive adjustment factors.

The maps below show one example held-out year for each implementation. Each map
uses the same three-panel format: CHIRPS observed, prediction, and prediction
minus CHIRPS. The table above summarizes performance across all validation
years.

![QM Texas DeepScale](figures/qm_reference_maps_long_texas_1991_2020.png)

![QM Texas xsdba](figures/qm_xsdba_maps_long_texas_1991_2020.png)

![QM Ethiopia DeepScale](figures/qm_reference_maps_ethiopia_fma_1991_2020.png)

![QM Ethiopia xsdba](figures/qm_xsdba_maps_ethiopia_fma_1991_2020.png)

### Interpretation

QM behaves sensibly on the CHIRPS coarse/fine fixtures, but the xsdba reference
scores better here. The main finding is a convention difference, not a generic
failure. Because this fixture is built by blurring CHIRPS, it favors additive,
delta-style reconstruction, so xsdba's convention is advantaged by construction.
This should not be read as xsdba QM being more skillful operationally. The
recommended development path is to keep the current empirical default for
compatibility and add an explicit additive-factor option rather than silently
changing existing behavior.

## Climatology Baseline

### Method Overview

Climatology is included only as a baseline. It does not use the coarse
pseudo-model predictor and it does not downscale anything; it simply predicts
the historical observed mean field from the training years. Its role is to make
sure the active methods are using information from the held-out predictor
rather than only reproducing the average rainfall pattern.

### Validation Results

| Fixture | Baseline | RMSE | Corr |
|---|---|---:|---:|
| Texas | Historical mean | 0.6607 | 0.6442 |
| Ethiopia FMA | Historical mean | 0.7633 | 0.8131 |

### Interpretation

Climatology is clearly weaker than the active methods on both controlled
fixtures. That is the expected result for this setup: because the predictor is
a coarsened version of CHIRPS, a working downscaling method should beat the
historical-mean baseline.

## Regenerating Results and Figures

The report source, renderer, validation scripts, generated figures, and
machine-readable result artifacts live under `validation/`. The main controlled
CHIRPS artifacts can be regenerated from the repository root with:

```bash
validation/regenerate_report_artifacts.sh
```

Use `validation/regenerate_report_artifacts.sh --skip-pycpt` to refresh only
the non-PyCPT artifacts. See `validation/README.md` for prerequisites, optional
environment variables, generated outputs, and the extra local data required for
the Lorenz et al. BCSD replication.

## Recommended Actions

1. Treat Delta and BCSD as the current reference-stable statistical baselines
   for the controlled CHIRPS workflow.
2. Keep CCA under parity review. The current implementation is plausible and
   close to PyCPT, but CCA conventions around modes, signs, scaling, and
   cross-validation still need targeted follow-up before calling it exact
   parity.
3. Add explicit additive-factor options for QM and DQM. The existing empirical
   defaults should remain stable for compatibility, but users should be able to
   request the additive convention used by the `xsdba` references.
4. Keep the bilinear interpolation baseline in future validation runs. It is a
   useful guardrail for this coarse/fine fixture because simple regridding is
   sometimes competitive with learned or calibrated methods.
5. Run production-like forecast validation separately before making operational
   skill claims. The CHIRPS coarse/fine fixtures verify implementation behavior;
   they do not replace validation on real forecast inputs.

## Summary

Within this controlled implementation check, the active downscaling methods all
beat the historical-mean climatology baseline on both fixtures, which is the
first sanity check this report needs.

Delta is the cleanest reference result: DeepScale matches the direct formula
and `python-cmethods` implementation on the same controlled inputs. BCSD also
passes its functional-reference check, landing close to the scikit-downscale
adapter on both Texas and Ethiopia while retaining strong scores against
CHIRPS. The Lorenz et al. research-product replication strengthens the BCSD
case further: DeepScale BCSD nearly matches the published WDCC SEAS5-BCSD
product over the D03 basin and sharply improves on raw SEAS5.

CCA is plausible but less strong on this particular coarse/fine reconstruction
task. Its PyCPT comparison is close enough to support the broad implementation
shape, but not close enough to treat as exact parity. QM and DQM are stable in
this controlled setup, but the `xsdba` references score better here because
their additive-adjustment convention is favored by the CHIRPS coarse/fine fixture.

The practical takeaway is that Delta and BCSD have the clearest current
validation support, CCA should remain under targeted parity review, and QM/DQM
need an explicit additive-adjustment option if DeepScale should match the xsdba
behavior without changing existing defaults silently.

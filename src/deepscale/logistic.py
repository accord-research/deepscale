"""Logistic-regression seasonal forecast from a scalar predictor index.

This is the WVG/logit stream: per grid cell, fit a logistic regression of
tercile-category occurrence (below / normal / above normal) on a single
predictor index (e.g. the Western-V Gradient SST index from
:class:`deepscale.Index`), then evaluate it at the forecast-year index value to
get below/normal/above probabilities.

Unlike the downscaling methods (CCA, eReg) this is NOT a gridded-field
calibration and so is not a ``seasonal_mme`` method; it is a standalone
forecast primitive whose predictor is a 1-D index series.

    idx = deepscale.Index.named("wvg")
    p = deepscale.logistic_forecast(idx.reduce(sst_hcst), obs, idx_fcst)
    # p: (tercile, lat, lon), tercile=[0,1,2] = below/normal/above, sums to 1

Two formulations (``model=``):

- ``"icpac_independent"`` (default): three independent binomial logits, one per
  category indicator, then renormalize to sum to 1. Matches ICPAC's operational
  per-category ``glm(..., family="binomial")`` recipe.
- ``"multinomial"``: a single multinomial logit over the 3-class label;
  probabilities are coherent by construction (no renormalization needed).

Two estimator backends (``backend=``):

- ``"sklearn"`` (default): ``sklearn.linear_model.LogisticRegression``. Supports
  L2 ``regularization``. No coefficient p-values.
- ``"statsmodels"``: ``statsmodels`` GLM/MNLogit. Unregularized, but exposes the
  predictor p-value, which enables ``significance_mask`` (drop cells where the
  index is not a significant predictor) — the way the ICPAC R masks insignificant
  cells.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

_TERCILE_COORD = [0, 1, 2]  # below, normal, above — matches deepscale.tercile


def _labels_from_obs(obs_vals: np.ndarray):
    """Per-cell tercile category labels (0/1/2) and the boundaries.

    obs_vals: (year, ncell). Returns (labels (year, ncell) int with -1 for NaN,
    t33 (ncell,), t67 (ncell,)).
    """
    with np.errstate(invalid="ignore"):
        t33 = np.nanpercentile(obs_vals, 100.0 / 3.0, axis=0)
        t67 = np.nanpercentile(obs_vals, 200.0 / 3.0, axis=0)
    labels = np.full(obs_vals.shape, -1, dtype=int)
    below = obs_vals < t33[None, :]
    above = obs_vals > t67[None, :]
    normal = (~below) & (~above) & np.isfinite(obs_vals)
    labels[below & np.isfinite(obs_vals)] = 0
    labels[normal] = 1
    labels[above & np.isfinite(obs_vals)] = 2
    return labels, t33, t67


def _fit_one_binomial_sklearn(x, y, x_f, regularization):
    """P(y=1 | x_f) for a single binomial logit, with sklearn."""
    from sklearn.linear_model import LogisticRegression

    if regularization is None:
        clf = LogisticRegression(penalty=None, max_iter=1000)
    else:
        clf = LogisticRegression(penalty="l2", C=1.0 / regularization, max_iter=1000)
    clf.fit(x[:, None], y)
    return float(clf.predict_proba([[x_f]])[0, 1]), None


def _fit_one_binomial_statsmodels(x, y, x_f, regularization):
    """P(y=1 | x_f) and the predictor p-value, with statsmodels GLM(Binomial).

    Per-cell logistic fits over a grid routinely hit benign numerical noise
    (overflow in ``exp``, perfect-separation on cells with a strong signal); we
    silence those locally so a gridded call doesn't emit thousands of warnings.
    """
    import warnings

    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

    X = sm.add_constant(x, has_constant="add")
    model = sm.GLM(y, X, family=sm.families.Binomial())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerfectSeparationWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        if regularization is None:
            res = model.fit()
            pval = float(res.pvalues[1])  # slope (index 1; 0 is the intercept)
        else:
            res = model.fit_regularized(alpha=regularization, L1_wt=0.0)
            pval = np.nan  # regularized GLM does not yield standard p-values
        prob = float(res.predict(np.array([[1.0, x_f]]))[0])
    return prob, pval


def _binomial_prob(x, y, x_f, *, backend, regularization, min_years):
    """P(y=1 | x_f) for one cell+category. Returns (prob, pvalue).

    Mirrors the reference recipe's edge cases: too few finite samples -> NaN;
    a degenerate label (all 0 or all 1) -> the base rate, no fit.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < min_years:
        return np.nan, np.nan
    xo, yo = x[ok], y[ok]
    if np.unique(yo).size < 2:
        return float(yo.mean()), np.nan  # never/always in this category
    if backend == "sklearn":
        return _fit_one_binomial_sklearn(xo, yo, x_f, regularization)
    if backend == "statsmodels":
        return _fit_one_binomial_statsmodels(xo, yo, x_f, regularization)
    raise ValueError(f"backend must be 'sklearn' or 'statsmodels'; got {backend!r}.")


def _multinomial_probs(x, labels, x_f, *, backend, regularization, min_years):
    """(p_below, p_normal, p_above) from one multinomial logit at one cell."""
    ok = np.isfinite(x) & (labels >= 0)
    if ok.sum() < min_years:
        return np.array([np.nan, np.nan, np.nan])
    xo, yo = x[ok], labels[ok]
    present = np.unique(yo)
    if present.size < 2:
        out = np.zeros(3)
        out[int(present[0])] = 1.0
        return out
    if backend == "sklearn":
        from sklearn.linear_model import LogisticRegression

        # multi_class is left at default: lbfgs on a >2-class problem solves the
        # multinomial logit (the explicit kwarg was deprecated in sklearn 1.5).
        kw = dict(max_iter=1000)
        clf = LogisticRegression(penalty=None, **kw) if regularization is None \
            else LogisticRegression(penalty="l2", C=1.0 / regularization, **kw)
        clf.fit(xo[:, None], yo)
        proba = clf.predict_proba([[x_f]])[0]
        out = np.zeros(3)
        for cls, p in zip(clf.classes_, proba):
            out[int(cls)] = p
        return out
    if backend == "statsmodels":
        import warnings

        import statsmodels.api as sm

        X = sm.add_constant(xo, has_constant="add")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            res = sm.MNLogit(yo, X).fit(disp=0)
            pred = np.asarray(res.predict(np.array([[1.0, x_f]])))[0]
        out = np.zeros(3)
        for cls, p in zip(np.sort(present), pred):
            out[int(cls)] = p
        return out
    raise ValueError(f"backend must be 'sklearn' or 'statsmodels'; got {backend!r}.")


def logistic_forecast(
    index,
    obs,
    index_forecast,
    *,
    model: str = "icpac_independent",
    backend: str = "sklearn",
    regularization: float | None = None,
    significance_mask: float | None = None,
    min_years: int = 10,
):
    """Per-cell logistic tercile forecast from a scalar predictor index.

    Parameters
    ----------
    index : xr.DataArray or array-like
        Predictor index over the training years, aligned to ``obs.year``.
    obs : xr.DataArray
        Predictand, dims ``(year, lat, lon)`` (lat/lon may be named latitude/
        longitude). Tercile boundaries are computed from this climatology.
    index_forecast : float or scalar DataArray
        The predictor index value for the forecast year.
    model : {"icpac_independent", "multinomial"}
        Formulation. ``icpac_independent`` fits one binomial logit per category
        then renormalizes; ``multinomial`` fits a single coherent 3-class logit.
    backend : {"sklearn", "statsmodels"}
        Estimator library. ``statsmodels`` is required for ``significance_mask``.
    regularization : float or None
        L2 strength (sklearn ``C = 1/regularization``). ``None`` = unregularized,
        which matches the ICPAC R ``glm``.
    significance_mask : float or None
        If set (e.g. ``0.1``), cells whose below-normal predictor p-value exceeds
        this threshold are masked to NaN. Requires ``backend="statsmodels"`` and
        ``regularization=None``.
    min_years : int
        Minimum finite (index, obs) pairs required to fit a cell; else NaN.

    Returns
    -------
    xr.DataArray
        dims ``(tercile, lat, lon)``, ``tercile=[0, 1, 2]`` = below/normal/above,
        probabilities in ``[0, 1]`` summing to 1 per cell.
    """
    if model not in ("icpac_independent", "multinomial"):
        raise ValueError(
            f"model must be 'icpac_independent' or 'multinomial'; got {model!r}."
        )
    if significance_mask is not None:
        if backend != "statsmodels":
            raise ValueError(
                "significance_mask requires backend='statsmodels' (sklearn does "
                "not expose coefficient p-values)."
            )
        if regularization is not None:
            raise ValueError(
                "significance_mask requires regularization=None (regularized GLM "
                "does not yield standard p-values)."
            )

    lat_dim = next(d for d in ("lat", "latitude", "Y", "y") if d in obs.dims)
    lon_dim = next(d for d in ("lon", "longitude", "X", "x") if d in obs.dims)
    obs = obs.transpose("year", lat_dim, lon_dim)
    nlat, nlon = obs.sizes[lat_dim], obs.sizes[lon_dim]
    obs_vals = obs.values.reshape(obs.sizes["year"], -1)
    ncell = obs_vals.shape[1]

    if isinstance(index, xr.DataArray) and "year" in index.coords and "year" in obs.coords:
        obs_years = obs.year.values
        index_years = index.year.values
        if len(np.unique(index_years)) != len(index_years):
            raise ValueError("index.year must not contain duplicate years.")
        if set(index_years.tolist()) != set(obs_years.tolist()):
            raise ValueError(
                "index.year values must match obs.year values; logistic "
                "calibration aligns predictor and obs by year."
            )
        index = index.sel(year=obs_years)

    x = np.asarray(index, dtype=float).reshape(-1)
    if x.size != obs.sizes["year"]:
        raise ValueError(
            f"index length ({x.size}) must match obs.year ({obs.sizes['year']})."
        )
    x_f_arr = np.asarray(index_forecast).reshape(-1)
    if x_f_arr.size != 1:
        raise ValueError(
            f"index_forecast must contain exactly one value; got {x_f_arr.size}."
        )
    x_f = float(x_f_arr[0])

    labels, _t33, _t67 = _labels_from_obs(obs_vals)

    probs = np.full((3, ncell), np.nan)
    pmask = np.zeros(ncell, dtype=bool)  # True -> mask this cell to NaN

    for g in range(ncell):
        if model == "multinomial":
            probs[:, g] = _multinomial_probs(
                x, labels[:, g], x_f, backend=backend,
                regularization=regularization, min_years=min_years,
            )
            continue
        # icpac_independent: one binomial logit per category, then renormalize.
        cat_probs = np.full(3, np.nan)
        below_pval = np.nan
        for cat in (0, 1, 2):
            y = (labels[:, g] == cat).astype(float)
            y[labels[:, g] < 0] = np.nan  # keep NaN obs out of the fit
            p, pval = _binomial_prob(
                x, y, x_f, backend=backend,
                regularization=regularization, min_years=min_years,
            )
            cat_probs[cat] = p
            if cat == 0:
                below_pval = pval
        s = np.nansum(cat_probs)
        if np.all(np.isfinite(cat_probs)) and np.isfinite(s) and s > 0:
            cat_probs = cat_probs / s
        elif np.any(np.isfinite(cat_probs)):
            cat_probs[:] = np.nan
        probs[:, g] = cat_probs
        if significance_mask is not None and not (
            np.isfinite(below_pval) and below_pval <= significance_mask
        ):
            pmask[g] = True

    if significance_mask is not None:
        probs[:, pmask] = np.nan

    out = xr.DataArray(
        probs.reshape(3, nlat, nlon),
        dims=["tercile", lat_dim, lon_dim],
        coords={
            "tercile": _TERCILE_COORD,
            lat_dim: obs[lat_dim],
            lon_dim: obs[lon_dim],
        },
        name="tercile_forecast",
    )
    return out

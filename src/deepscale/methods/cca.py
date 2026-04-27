"""CCA downscaling — SVD-based, matching CPT Fortran 17.8.3 (cca.F95).

Algorithm (CPT's full_cca + cca_predict):
  1. Center data (remove mean), apply sqrt(cos(lat)) weighting
  2. SVD of weighted anomalies → EOF patterns, singular values, unit-norm PC scores
  3. SVD of cross-product matrix (Y_pcs.T @ X_pcs) → canonical correlations (mu),
     CCA rotations (r for Y, s for X)
  4. Predict: project new X → normalize by svx → rotate by s → scale by mu →
     rotate by r → scale by svy → project through eofy → add mean
"""
import numpy as np
import xarray as xr
from scipy.stats import kendalltau
from .base import MethodBase
from ..registry import register_method


def _svd_pca(X, n_components):
    """SVD-based PCA matching CPT's get_pcs().

    Returns:
        eof: (n_features, n_components) — spatial loadings (V from SVD)
        ts: (n_components, n_samples) — unit-norm PC scores (U.T from SVD)
        sv: (n_components,) — singular values
    """
    # X is (n_samples, n_features), already centered+weighted
    # CPT does gesdd on (m, n) = (features, samples), i.e., X.T
    U, s, Vt = np.linalg.svd(X.T, full_matrices=False)
    # U: (features, min(f,s)) — EOF loadings
    # s: singular values
    # Vt: (min(f,s), samples) — unit-norm PC time series
    nc = min(n_components, len(s))
    return U[:, :nc], Vt[:nc, :], s[:nc]


@register_method("cca")
class CCAMethod(MethodBase):
    def __init__(self, n_modes=3, x_eof_modes=None, y_eof_modes=None,
                 cca_modes=None, standardize=False):
        self.n_modes = n_modes
        self.x_eof_modes = x_eof_modes
        self.y_eof_modes = y_eof_modes
        self.cca_modes = cca_modes
        self.standardize = standardize

    def fit(self, hindcast, obs, **kwargs):
        x_eof_modes = kwargs.get("x_eof_modes", self.x_eof_modes)
        y_eof_modes = kwargs.get("y_eof_modes", self.y_eof_modes)
        cca_modes = kwargs.get("cca_modes", self.cca_modes) or kwargs.get("n_modes", self.n_modes)

        gcm_mean = hindcast.mean("member")
        n_years = len(gcm_mean.year)

        X = gcm_mean.values.reshape(n_years, -1)
        Y = obs.values.reshape(n_years, -1)

        # Mask out columns that are all-NaN
        self.x_valid_ = ~np.isnan(X).all(axis=0)
        self.y_valid_ = ~np.isnan(Y).all(axis=0)
        X = X[:, self.x_valid_]
        Y = Y[:, self.y_valid_]
        X = np.nan_to_num(X, nan=np.nanmean(X))
        Y = np.nan_to_num(Y, nan=np.nanmean(Y))

        # Center
        self.x_mean_ = X.mean(axis=0)
        self.y_mean_ = Y.mean(axis=0)
        X_c = X - self.x_mean_
        Y_c = Y - self.y_mean_

        # Standardize: divide by per-column sample std (ddof=1).
        # CPT does this before lat-weighting and SVD. Confirmed by matching
        # EOF explained variances to 6 decimal places vs CPT Fortran output.
        if self.standardize:
            self.x_std_ = X_c.std(axis=0, ddof=1)
            self.x_std_[self.x_std_ < 1e-20] = 1.0
            self.y_std_ = Y_c.std(axis=0, ddof=1)
            self.y_std_[self.y_std_ < 1e-20] = 1.0
            X_c = X_c / self.x_std_
            Y_c = Y_c / self.y_std_
        else:
            self.x_std_ = None
            self.y_std_ = None

        # Latitude area weighting (CPT latitude_weight before SVD)
        x_lats = np.repeat(gcm_mean.lat.values, len(gcm_mean.lon))
        y_lats = np.repeat(obs.lat.values, len(obs.lon))
        self.x_wt_ = np.sqrt(np.cos(np.deg2rad(x_lats)))[self.x_valid_]
        self.y_wt_ = np.sqrt(np.cos(np.deg2rad(y_lats)))[self.y_valid_]
        X_c = X_c * self.x_wt_
        Y_c = Y_c * self.y_wt_

        # EOF truncation via SVD (matching CPT get_pcs)
        if x_eof_modes is None:
            x_eof_modes = min(n_years - 1, X.shape[1], 10)
        if y_eof_modes is None:
            y_eof_modes = min(n_years - 1, Y.shape[1], 10)

        self.eofx_, self.tsx_, self.svx_ = _svd_pca(X_c, x_eof_modes)
        self.eofy_, self.tsy_, self.svy_ = _svd_pca(Y_c, y_eof_modes)
        nxe = len(self.svx_)
        nye = len(self.svy_)

        # CCA via SVD of cross-product matrix (CPT cca.F95 L111-112)
        # tsx and tsy are unit-norm: shape (n_eof, n_years)
        ce = self.tsy_ @ self.tsx_.T  # (nye, nxe)
        U_ce, mu, Vt_ce = np.linalg.svd(ce, full_matrices=False)

        ncc = min(cca_modes, nxe, nye, n_years - 1)
        self.ncc_ = ncc
        self.mu_ = mu[:ncc]             # canonical correlations
        self.r_ = U_ce[:, :ncc]         # Y CCA weights (nye, ncc)
        self.s_ = Vt_ce[:ncc, :]        # X CCA weights transposed (ncc, nxe)

        self.n_train_ = n_years
        self.x_eof_modes_ = nxe
        self.obs_shape_ = obs.isel(year=0).shape
        self.obs_coords_ = {"lat": obs.lat, "lon": obs.lon}

    def leverage(self, forecast):
        """CPT-compatible leverage (cca.F95 L602, L618-620).

        xvp = 1/n + Sum(prjc(1:ncc))**2
        where prjc = s @ (eofx.T @ x_anom / svx)
        """
        n = self.n_train_
        x = forecast.mean("member").values.reshape(1, -1)
        x = x[:, self.x_valid_]
        x = np.nan_to_num(x, nan=np.nanmean(x))
        x_anom = x - self.x_mean_
        if self.x_std_ is not None:
            x_anom = x_anom / self.x_std_
        x_anom = x_anom * self.x_wt_
        rwk = self.eofx_.T @ x_anom.ravel() / self.svx_
        prjc = self.s_ @ rwk
        return 1.0 / n + float(np.sum(prjc)) ** 2

    def predict(self, forecast, **kwargs):
        """CPT-compatible prediction (cca.F95 L605-638).

        For each member:
          1. x_anom = (x - xm) [/ x_std] * lat_wt
          2. rwk = eofx.T @ x_anom          (project onto X EOFs)
          3. rwk = rwk / svx                 (normalize to unit variance)
          4. prjc = s @ rwk                  (project onto CCA modes)
          5. prjc = prjc * mu               (scale by canonical correlations)
          6. rwk = r @ prjc                  (back-project to Y EOF space)
          7. rwk = rwk * svy                 (scale by Y singular values)
          8. fcast = eofy @ rwk              (back-project to Y grid space)
          9. fcast = fcast / lat_wt [* y_std] + ym  (undo transforms, add mean)
        """
        results = []
        for m in range(len(forecast.member)):
            x = forecast.isel(member=m).values.reshape(1, -1)
            x = x[:, self.x_valid_]
            x = np.nan_to_num(x, nan=np.nanmean(x))

            x_anom = x.ravel() - self.x_mean_
            if self.x_std_ is not None:
                x_anom = x_anom / self.x_std_
            x_anom = x_anom * self.x_wt_

            rwk = self.eofx_.T @ x_anom / self.svx_
            prjc = self.s_ @ rwk * self.mu_
            rwk_y = self.r_ @ prjc * self.svy_
            fcast = self.eofy_ @ rwk_y

            y_pred_valid = fcast / self.y_wt_
            if self.y_std_ is not None:
                y_pred_valid = y_pred_valid * self.y_std_
            y_pred_valid = y_pred_valid + self.y_mean_

            y_full = np.full(len(self.y_valid_), np.nan)
            y_full[self.y_valid_] = y_pred_valid
            results.append(y_full.reshape(self.obs_shape_))

        return xr.DataArray(
            np.stack(results),
            dims=["member", "lat", "lon"],
            coords={"member": forecast.member, **self.obs_coords_},
        )


def select_modes(gcm, obs, years, window, x_eof_range=(1, 8), y_eof_range=(1, 6),
                 cca_range=(1, 3)):
    """CPT-compatible mode auto-selection via cross-validated Kendall's tau.

    Matches CPT's cv_cca (cca.F95 L223-539) + goodness (scores.F95 L3589-3726):
    1. For each CV fold, fit CCA for every mode combination, predict held-out year
    2. Compute average Kendall's tau across grid points for each combination
    3. Return the combination with highest goodness

    Returns (best_x_eof, best_y_eof, best_cca, goodness_value, cv_predictions, leverages)
    """
    from ..cv import loyo

    # Enumerate all valid mode combinations
    combos = []
    for xe in range(x_eof_range[0], x_eof_range[1] + 1):
        for ye in range(y_eof_range[0], y_eof_range[1] + 1):
            for cc in range(cca_range[0], min(cca_range[1], xe, ye) + 1):
                combos.append((xe, ye, cc))

    n_combos = len(combos)
    n_years = len(years)

    # Collect CV predictions for every combo: shape (n_combos, n_years, n_grid)
    obs_flat = obs.values.reshape(n_years, -1)
    valid_mask = ~np.isnan(obs_flat).all(axis=0)
    preds_all = np.full((n_combos, n_years, valid_mask.sum()), np.nan)
    levs_all = np.full((n_combos, n_years), np.nan)

    print(f"  Mode selection: {n_combos} combinations x {n_years} CV folds...")
    for fold_idx, (train_yrs, test_yr) in enumerate(loyo(years, window=window)):
        yr_idx = years.index(test_yr)
        forecast = gcm.sel(year=[test_yr]).isel(year=0, drop=True)

        for ci, (xe, ye, cc) in enumerate(combos):
            try:
                m = CCAMethod(x_eof_modes=xe, y_eof_modes=ye, cca_modes=cc)
                m.fit(gcm.sel(year=train_yrs), obs.sel(year=train_yrs))
                pred = m.predict(forecast).mean("member")
                pred_flat = pred.values.reshape(-1)
                preds_all[ci, yr_idx, :] = pred_flat[valid_mask]
                levs_all[ci, yr_idx] = m.leverage(forecast)
            except Exception:
                # CCA can fail for degenerate mode counts; leave as NaN
                pass

    # Compute goodness for each combo: average Kendall's tau across grid points
    # (CPT igood=3, scores.F95 L3675-3679)
    obs_valid = obs_flat[:, valid_mask]
    n_grid = obs_valid.shape[1]
    best_goodness = -np.inf
    best_idx = 0

    for ci in range(n_combos):
        preds = preds_all[ci]
        if np.isnan(preds).all():
            continue
        tau_sum = 0.0
        n_valid_pts = 0
        for gi in range(n_grid):
            o = obs_valid[:, gi]
            p = preds[:, gi]
            mask = np.isfinite(o) & np.isfinite(p)
            if mask.sum() < 4:
                continue
            tau, _ = kendalltau(p[mask], o[mask])
            if np.isfinite(tau):
                tau_sum += tau
                n_valid_pts += 1
        if n_valid_pts > 0:
            g = tau_sum / n_valid_pts
            if g > best_goodness:
                best_goodness = g
                best_idx = ci

    xe, ye, cc = combos[best_idx]
    print(f"  Optimal modes: x_eof={xe}, y_eof={ye}, cca={cc} (Kendall tau={best_goodness:+.4f})")

    # Reconstruct xarray predictions and leverages for the best combo
    best_preds_flat = preds_all[best_idx]
    best_preds = np.full((n_years, *obs.isel(year=0).shape), np.nan)
    best_preds.reshape(n_years, -1)[:, valid_mask] = best_preds_flat

    cv = xr.DataArray(best_preds, dims=["year", "lat", "lon"],
                      coords={"year": years, "lat": obs.lat, "lon": obs.lon})
    leverages = levs_all[best_idx]

    return xe, ye, cc, best_goodness, cv, leverages

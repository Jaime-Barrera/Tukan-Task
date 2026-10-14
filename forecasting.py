from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from statsmodels.tsa.api import VAR
from statsmodels.tsa.ar_model import AutoReg

from pca_analysis import PCAResult, LBL, _fix_sign_convention

HORIZON_OPTIONS = {
    "1 mes": 1,
    "3 meses (1 trimestre)": 3,
    "6 meses": 6,
    "12 meses (4 trimestres)": 12,
}


@dataclass
class ForecastResult:
    model_name: str
    horizon: int
    score_forecasts: pd.DataFrame
    curve_forecasts: pd.DataFrame
    last_observed_date: pd.Timestamp
    forecast_dates: pd.DatetimeIndex
    curve_stds: pd.DataFrame = field(default_factory=pd.DataFrame)
    score_covariances: list[np.ndarray] = field(default_factory=list)
    rmse_by_tenor: pd.Series = field(default_factory=pd.Series)


def _build_forecast_dates(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq="MS")


def _reconstruct_curve(
    score_forecasts: pd.DataFrame,
    loadings: np.ndarray,
    n_pcs: int = 3,
    feature_means: pd.Series | None = None,
    feature_stds: pd.Series | None = None,
    last_observed_rates: pd.Series | None = None,
    is_changes: bool = False,
) -> pd.DataFrame:
    V = loadings[:n_pcs]
    scores_arr = score_forecasts.iloc[:, :n_pcs].values
    reconstructed = scores_arr @ V

    if feature_means is not None and feature_stds is not None:
        reconstructed = reconstructed * feature_stds.values + feature_means.values
        if is_changes and last_observed_rates is not None:
            reconstructed = last_observed_rates.values + np.cumsum(reconstructed, axis=0)

    return pd.DataFrame(reconstructed, index=score_forecasts.index, columns=LBL)


def _compute_curve_uncertainty(
    score_covariances: list[np.ndarray],
    loadings: np.ndarray,
    forecast_dates: pd.DatetimeIndex,
    n_pcs: int = 3,
    feature_stds: pd.Series | None = None,
    is_changes: bool = False,
) -> pd.DataFrame:
    V = loadings[:n_pcs]
    horizon = len(score_covariances)
    n_tenors = V.shape[1]
    step_stds = np.zeros((horizon, n_tenors))

    for h in range(horizon):
        cov_h = score_covariances[h]
        var_h = np.diag(V.T @ cov_h @ V)
        step_stds[h] = np.sqrt(np.maximum(var_h, 0.0))

    if feature_stds is not None:
        step_stds = step_stds * feature_stds.values

    if is_changes:
        step_stds = np.sqrt(np.cumsum(step_stds ** 2, axis=0))

    return pd.DataFrame(step_stds, index=forecast_dates, columns=LBL)


def forecast_random_walk(
    scores: pd.DataFrame,
    loadings: np.ndarray,
    horizon: int,
    feature_means: pd.Series | None = None,
    feature_stds: pd.Series | None = None,
    last_observed_rates: pd.Series | None = None,
    is_changes: bool = False,
) -> ForecastResult:
    if scores.empty:
        raise ValueError("scores DataFrame cannot be empty.")
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, received: {horizon}")

    last_date = scores.index[-1]
    forecast_dates = _build_forecast_dates(last_date, horizon)

    last_scores = scores.iloc[-1]
    score_forecasts = pd.DataFrame(
        np.tile(last_scores.values, (horizon, 1)),
        index=forecast_dates,
        columns=scores.columns,
    )

    diffs = scores.iloc[:, :3].diff().dropna()
    cov_diff = np.cov(diffs.values, rowvar=False) if len(diffs) > 1 else np.eye(3) * 0.1
    score_covariances = [cov_diff * (h + 1) for h in range(horizon)]

    curve_forecasts = _reconstruct_curve(
        score_forecasts,
        loadings,
        feature_means=feature_means,
        feature_stds=feature_stds,
        last_observed_rates=last_observed_rates,
        is_changes=is_changes,
    )
    curve_stds = _compute_curve_uncertainty(
        score_covariances,
        loadings,
        forecast_dates,
        feature_stds=feature_stds,
        is_changes=is_changes,
    )

    return ForecastResult(
        model_name="Random Walk",
        horizon=horizon,
        score_forecasts=score_forecasts,
        curve_forecasts=curve_forecasts,
        last_observed_date=last_date,
        forecast_dates=forecast_dates,
        curve_stds=curve_stds,
        score_covariances=score_covariances,
    )


def forecast_ou_ar1(
    scores: pd.DataFrame,
    loadings: np.ndarray,
    horizon: int,
    feature_means: pd.Series | None = None,
    feature_stds: pd.Series | None = None,
    last_observed_rates: pd.Series | None = None,
    is_changes: bool = False,
) -> ForecastResult:
    if scores.empty:
        raise ValueError("scores DataFrame cannot be empty.")
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, received: {horizon}")

    last_date = scores.index[-1]
    forecast_dates = _build_forecast_dates(last_date, horizon)

    n_pcs = min(3, scores.shape[1])
    score_matrix = np.zeros((horizon, scores.shape[1]))
    score_covariances: list[np.ndarray] = []

    variances_by_step = np.zeros((horizon, n_pcs))
    for i, col in enumerate(scores.columns[:n_pcs]):
        series = scores[col].dropna()
        model = AutoReg(series, lags=1).fit()
        pred = model.predict(start=len(series), end=len(series) + horizon - 1)
        score_matrix[:, i] = pred.values

        sigma2 = float(model.sigma2)
        phi = float(model.params.iloc[1]) if len(model.params) > 1 else 0.0
        for h in range(horizon):
            cum_factor = (
                (1.0 - phi ** (2 * (h + 1))) / max(1e-6, (1.0 - phi ** 2))
                if abs(phi) < 0.999 else float(h + 1)
            )
            variances_by_step[h, i] = max(1e-6, sigma2 * cum_factor)

    for h in range(horizon):
        score_covariances.append(np.diag(variances_by_step[h]))

    score_forecasts = pd.DataFrame(score_matrix, index=forecast_dates, columns=scores.columns)
    curve_forecasts = _reconstruct_curve(
        score_forecasts,
        loadings,
        feature_means=feature_means,
        feature_stds=feature_stds,
        last_observed_rates=last_observed_rates,
        is_changes=is_changes,
    )
    curve_stds = _compute_curve_uncertainty(
        score_covariances,
        loadings,
        forecast_dates,
        feature_stds=feature_stds,
        is_changes=is_changes,
    )

    return ForecastResult(
        model_name="O-U / AR(1)",
        horizon=horizon,
        score_forecasts=score_forecasts,
        curve_forecasts=curve_forecasts,
        last_observed_date=last_date,
        forecast_dates=forecast_dates,
        curve_stds=curve_stds,
        score_covariances=score_covariances,
    )


def forecast_var(
    scores: pd.DataFrame,
    loadings: np.ndarray,
    horizon: int,
    max_lags: int = 6,
    n_pcs: int = 3,
    feature_means: pd.Series | None = None,
    feature_stds: pd.Series | None = None,
    last_observed_rates: pd.Series | None = None,
    is_changes: bool = False,
) -> ForecastResult:
    if scores.empty:
        raise ValueError("scores DataFrame cannot be empty.")
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, received: {horizon}")

    last_date = scores.index[-1]
    forecast_dates = _build_forecast_dates(last_date, horizon)

    working_scores = scores.iloc[:, :n_pcs].dropna()
    model = VAR(working_scores)
    optimal_lag = model.select_order(maxlags=max_lags).selected_orders.get("aic", 1)
    optimal_lag = max(1, optimal_lag)

    fitted = model.fit(optimal_lag)
    raw_forecast = fitted.forecast(working_scores.values[-optimal_lag:], steps=horizon)
    raw_cov = fitted.forecast_cov(steps=horizon)
    score_covariances = [raw_cov[h] for h in range(horizon)]

    score_forecasts = pd.DataFrame(raw_forecast, index=forecast_dates, columns=working_scores.columns)
    curve_forecasts = _reconstruct_curve(
        score_forecasts,
        loadings,
        n_pcs=n_pcs,
        feature_means=feature_means,
        feature_stds=feature_stds,
        last_observed_rates=last_observed_rates,
        is_changes=is_changes,
    )
    curve_stds = _compute_curve_uncertainty(
        score_covariances,
        loadings,
        forecast_dates,
        feature_stds=feature_stds,
        is_changes=is_changes,
    )

    return ForecastResult(
        model_name=f"VAR({optimal_lag})",
        horizon=horizon,
        score_forecasts=score_forecasts,
        curve_forecasts=curve_forecasts,
        last_observed_date=last_date,
        forecast_dates=forecast_dates,
        curve_stds=curve_stds,
        score_covariances=score_covariances,
    )


def evaluate_models_oos(
    working_rates: pd.DataFrame,
    horizon: int,
    is_changes: bool = True,
    test_periods: int = 12,
) -> pd.DataFrame:
    if working_rates.empty:
        raise ValueError("working_rates DataFrame cannot be empty.")
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, received: {horizon}")

    n = len(working_rates)
    train_end = n - test_periods - horizon
    if train_end < 30:
        raise ValueError("Not enough historical data for out-of-sample evaluation.")

    results: dict[str, list[float]] = {"Random Walk": [], "O-U / AR(1)": [], "VAR": []}

    for step in range(test_periods):
        cutoff = train_end + step
        train_raw = working_rates.iloc[:cutoff]
        actual_rates = working_rates.iloc[cutoff : cutoff + horizon]

        if is_changes:
            train_diff = train_raw.diff().dropna()
            means = train_diff.mean()
            stds = train_diff.std()
            X = (train_diff - means) / stds
        else:
            means = train_raw.mean()
            stds = train_raw.std()
            X = (train_raw - means) / stds

        pca = PCA(n_components=3).fit(X)
        V = _fix_sign_convention(pca.components_.copy())
        scores = pd.DataFrame(X.values @ V.T, index=X.index, columns=["PC1", "PC2", "PC3"])
        last_rates = train_raw.iloc[-1]

        fc_rw = forecast_random_walk(
            scores, V, horizon,
            feature_means=means, feature_stds=stds,
            last_observed_rates=last_rates, is_changes=is_changes,
        )
        fc_ou = forecast_ou_ar1(
            scores, V, horizon,
            feature_means=means, feature_stds=stds,
            last_observed_rates=last_rates, is_changes=is_changes,
        )
        fc_var = forecast_var(
            scores, V, horizon, max_lags=4,
            feature_means=means, feature_stds=stds,
            last_observed_rates=last_rates, is_changes=is_changes,
        )

        n_eval = min(horizon, len(actual_rates))
        for m_name, fc_obj in [("Random Walk", fc_rw), ("O-U / AR(1)", fc_ou), ("VAR", fc_var)]:
            err = fc_obj.curve_forecasts.values[:n_eval] - actual_rates.values[:n_eval]
            rmse = float(np.sqrt(np.mean(err ** 2)))
            results[m_name].append(rmse)

    summary = pd.DataFrame(
        {name: [np.nanmean(vals)] for name, vals in results.items()},
        index=[f"RMSE real (% tasa, h={horizon})"],
    ).T
    summary.index.name = "Modelo"
    return summary.round(4)

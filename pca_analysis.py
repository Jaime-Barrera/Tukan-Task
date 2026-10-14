from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

TENOR = [
    "CETEs, 28 days", "CETEs, 91 days", "CETEs, 182 days", "CETEs, 364 days",
    "MBONOs, 3 years", "MBONOs, 5 years", "MBONOs, 10 years",
    "MBONOs, 20 years", "MBONOs, 30 years",
]
LBL = ["28d", "91d", "182d", "364d", "3Y", "5Y", "10Y", "20Y", "30Y"]

CRISIS_BANDS = [
    ("2008-06", "2009-06", "Crisis Financiera 2008"),
    ("2020-01", "2020-12", "COVID-19 2020"),
]


@dataclass
class PCAResult:
    loadings: np.ndarray
    scores: pd.DataFrame
    explained_variance: np.ndarray
    label: str
    feature_means: pd.Series | None = None
    feature_stds: pd.Series | None = None


def _fix_sign_convention(V: np.ndarray) -> np.ndarray:
    if V[0].sum() < 0:      V[0] *= -1
    if V[1][0] > V[1][-1]:  V[1] *= -1
    if V[2][4] < 0:         V[2] *= -1
    return V


def run_pca(series: pd.DataFrame, label: str) -> PCAResult:
    if series.isnull().any().any():
        raise ValueError("Input series contains NaN values. Apply ffill/dropna before calling run_pca.")

    n_components = min(9, series.shape[1])
    means = series.mean()
    stds = series.std()
    X = (series - means) / stds
    pca = PCA(n_components=n_components).fit(X)
    V = _fix_sign_convention(pca.components_.copy())

    scores = pd.DataFrame(
        X.values @ V.T,
        index=series.index,
        columns=[f"PC{i + 1}" for i in range(n_components)],
    )
    explained = pca.explained_variance_ratio_ * 100
    return PCAResult(
        loadings=V,
        scores=scores,
        explained_variance=explained,
        label=label,
        feature_means=means,
        feature_stds=stds,
    )


def prepare_working_series(wide: pd.DataFrame) -> pd.DataFrame:
    available = [col for col in TENOR if col in wide.columns]
    if len(available) < 9:
        raise ValueError(f"Expected 9 tenors, found {len(available)}: {available}")
    w = wide[available].astype(float)
    ff = w.ffill().dropna()
    if not ff.notna().all().all():
        raise ValueError("NaN values remain after ffill + dropna.")
    return ff


def compute_pca_results(wide: pd.DataFrame) -> tuple[PCAResult, PCAResult]:
    ff = prepare_working_series(wide)
    result_levels = run_pca(ff, label="Niveles")
    result_changes = run_pca(ff.diff().dropna(), label="Cambios")
    return result_levels, result_changes

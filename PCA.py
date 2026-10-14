"""
PCA.py — Q1: estructura de la curva de tasas Banxico (9 instrumentos)

Decisiones (ver log):
  - ffill sobre TODO el histórico y recién después dropna
    (recortar antes huérfanaba la única observación previa de la 30Y)
  - PCA sobre la matriz de correlación (estandarizando las series)
  - Dos versiones: niveles y cambios mensuales
  - Convención de signos: PC1 positivo, PC2 creciente con el plazo, PC3 joroba +
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.decomposition import PCA

from EDA import load_and_prepare_rate_data

# ------------------------------------------------------------------
# 0. Serie de trabajo
# ------------------------------------------------------------------
DATASET_PATH = Path("banxico_auction_rates_monthly.csv")
wide = load_and_prepare_rate_data(DATASET_PATH)

TENOR = ["CETEs, 28 days", "CETEs, 91 days", "CETEs, 182 days", "CETEs, 364 days",
        "MBONOs, 3 years", "MBONOs, 5 years", "MBONOs, 10 years",
        "MBONOs, 20 years", "MBONOs, 30 years"]
LBL = ["28d", "91d", "182d", "364d", "3Y", "5Y", "10Y", "20Y", "30Y"]

w = wide[TENOR].astype(float)
ff = w.ffill().dropna()              # llenar TODO el histórico, luego recortar
assert ff.notna().all().all(), "quedan NaNs en la serie de trabajo"

ini, fin = pd.Timestamp(ff.index.min()), pd.Timestamp(ff.index.max())
print(f"Serie de trabajo: {ini:%Y-%m} → {fin:%Y-%m} ({len(ff)} meses)\n")

# ------------------------------------------------------------------
# 1. PCA sobre la matriz de correlación
# ------------------------------------------------------------------
def correr_pca(serie):
    """Devuelve V (loadings 9x9), scores (series temporales) y var (% explicada)."""
    X = (serie - serie.mean()) / serie.std()
    pca = PCA(n_components=9).fit(X)
    V = pca.components_.copy()

    # Convención de signos (sklearn los elige al azar en cada corrida):
    if V[0].sum() < 0:      V[0] *= -1   # PC1: todo positivo (nivel)
    if V[1][0] > V[1][-1]:  V[1] *= -1   # PC2: creciente con el plazo (pendiente)
    if V[2][4] < 0:         V[2] *= -1   # PC3: joroba positiva (índice 4 = 3Y, el vientre)

    scores = pd.DataFrame(X.values @ V.T, index=serie.index, columns=[f"PC{i+1}" for i in range(9)])
    var = pca.explained_variance_ratio_ * 100
    return V, scores, var


V_lvl, S_lvl, var_lvl = correr_pca(ff)                  # niveles
V_chg, S_chg, var_chg = correr_pca(ff.diff().dropna())  # cambios

print("Varianza explicada acumulada (%)")
print(pd.DataFrame({"Niveles": np.round(var_lvl.cumsum(), 1),"Cambios": np.round(var_chg.cumsum(), 1)},index=[f"PC{i+1}" for i in range(9)]), "\n")

print("Loadings — cambios (PC1–PC3)")
print(pd.DataFrame(np.round(V_chg[:3], 3), index=["PC1", "PC2", "PC3"],columns=LBL), "\n")

# ------------------------------------------------------------------
# 2. Figuras
# ------------------------------------------------------------------
OUT = Path("figs"); OUT.mkdir(exist_ok=True)

# --- 2a. Scree plot (cambios): ¿cuántos factores tiene la curva?
fig1, ax = plt.subplots(figsize=(7, 4))
ax.bar(range(1, 10), var_chg, color="lightsteelblue")
ax.set_xticks(range(1, 10)); ax.set_xlabel("Componente")
ax.set_ylabel("% varianza"); ax.set_title("Scree plot — PCA sobre cambios")
ax2 = ax.twinx()
ax2.plot(range(1, 10), var_chg.cumsum(), "o-", color="tab:red")
ax2.axhline(95, ls="--", color="grey", lw=1); ax2.set_ylabel("% acumulado")
fig1.tight_layout(); fig1.savefig(OUT / "01_scree_cambios.png", dpi=150)

# --- 2b. Loadings vs plazo: ¿qué significa cada factor? (LA figura)
fig2, ax = plt.subplots(figsize=(8, 5))
for j, c in zip([0, 1, 2], ["tab:blue", "tab:orange", "tab:green"]):
    ax.plot(LBL, V_chg[j], "o-", color=c, label=f"PC{j+1} ({var_chg[j]:.0f}%)")
ax.axhline(0, color="grey", lw=0.8)
ax.set_xlabel("Plazo"); ax.set_ylabel("Loading"); ax.legend()
ax.set_title("Loadings — PCA sobre cambios")
fig2.tight_layout(); fig2.savefig(OUT / "02_loadings_cambios.png", dpi=150)

# --- 2c. Scores en el tiempo: ¿cómo se comportó cada factor?
fig3, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
for j, ax in enumerate(axes):
    ax.plot(S_chg.index, S_chg[f"PC{j+1}"], lw=0.9)
    ax.axhline(0, color="grey", lw=0.5); ax.set_ylabel(f"PC{j+1}")
    for ini_ev, fin_ev in [("2008-06", "2009-06"), ("2020-01", "2020-12")]:
        ax.axvspan(pd.Timestamp(ini_ev), pd.Timestamp(fin_ev), color="red", alpha=0.08)
axes[0].set_title("Scores — PCA sobre cambios")
axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
fig3.tight_layout(); fig3.savefig(OUT / "03_scores_cambios.png", dpi=150)

plt.show()
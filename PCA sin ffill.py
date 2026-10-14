"""
PCA_chequeo.py — Diagnóstico del efecto del ffill en el PCA de cambios.

Responde dos preguntas:
  1. ¿El ffill estaba deprimiendo el PC1 (53%)?
  2. ¿El PC3 (contraste 20Y/30Y) es artefacto del calendario de subastas?

Método: PCA sobre cambios usando SOLO los meses donde las 9 tasas están
observadas tanto en t como en t-1 (sin ninguna imputación).

NOTA: Se incluye validación explícita de tamaño muestral. Si n_obs == 0,
el método es inviable para este dataset y se reporta transparentemente.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from EDA import load_and_prepare_rate_data

DATASET_PATH = Path("banxico_auction_rates_monthly.csv")
wide = load_and_prepare_rate_data(DATASET_PATH)

TENOR = [
    "CETEs, 28 days", "CETEs, 91 days", "CETEs, 182 days", "CETEs, 364 days",
    "MBONOs, 3 years", "MBONOs, 5 years", "MBONOs, 10 years",
    "MBONOs, 20 years", "MBONOs, 30 years"
]
LBL = ["28d", "91d", "182d", "364d", "3Y", "5Y", "10Y", "20Y", "30Y"]

# Datos crudos SIN ffill — tal cual el archivo
w = wide[TENOR].astype(float)


def correr_pca(serie: pd.DataFrame):
    """Ejecuta PCA estandarizado con signado consistente de componentes."""
    X = (serie - serie.mean()) / serie.std()
    pca = PCA(n_components=9).fit(X)
    V = pca.components_.copy()

    # Convenciones de signo para interpretabilidad
    if V[0].sum() < 0:      V[0] *= -1   # PC1: todo positivo
    if V[1][0] > V[1][-1]:  V[1] *= -1   # PC2: creciente con el plazo
    if V[2][4] < 0:         V[2] *= -1   # PC3: joroba positiva

    scores = pd.DataFrame(
        X.values @ V.T,
        index=serie.index,
        columns=[f"PC{i+1}" for i in range(9)]
    )
    var = pca.explained_variance_ratio_ * 100
    return V, scores, var


# ------------------------------------------------------------------
# Preparación de datos: versiones con y sin ffill
# ------------------------------------------------------------------
d_ff = w.ffill().diff().dropna()
d_obs = w.diff().dropna()

n_ff = len(d_ff)
n_obs = len(d_obs)

print(f"Muestras con ffill:                {n_ff}")
print(f"Muestras completamente observadas: {n_obs}")
print("-" * 50)

# ------------------------------------------------------------------
# Validación de viabilidad (Failsafe)
# ------------------------------------------------------------------
if n_obs == 0:
    msg = (
        "ERROR DE VIABILIDAD: No existen meses con observaciones completas "
        "en t y t-1 para los 9 tenores.\n"
        "El método 'solo meses observados' es inviable con este dataset.\n"
        "I cannot confirm the effect of ffill on PC1 or whether PC3 is an artifact "
        "because the required non-imputed sample size is zero.\n\n"
        "Recomendaciones:\n"
        "  1. Reducir el conjunto de tenores a aquellos con calendarios sincronizados.\n"
        "  2. Utilizar PCA tolerante a datos faltantes (NIPALS / Iterative PCA).\n"
        "  3. Documentar ffill como paso necesario y cuantificar su impacto "
        "mediante simulación o métodos alternativos de imputación."
    )
    print(msg)
    sys.exit(1)

# ------------------------------------------------------------------
# Ejecución de PCA (solo si n_obs > 0)
# ------------------------------------------------------------------
_, _, var_ff = correr_pca(d_ff)
V_obs, S_obs, var_obs = correr_pca(d_obs)

# Resultados comparativos
print("\nVarianza acumulada (%) — comparación")
print(pd.DataFrame({
    "Con ffill": np.round(var_ff.cumsum(), 1),
    "Observados": np.round(var_obs.cumsum(), 1)
}, index=[f"PC{i+1}" for i in range(9)]))

print("\nLoadings — meses observados (PC1–PC3)")
print(pd.DataFrame(
    np.round(V_obs[:3], 3),
    index=["PC1", "PC2", "PC3"],
    columns=LBL
))

# Diagnóstico puntual del PC3: correlación 20Y–30Y
r_ff = d_ff["MBONOs, 20 years"].corr(d_ff["MBONOs, 30 years"])
r_obs = d_obs["MBONOs, 20 years"].corr(d_obs["MBONOs, 30 years"])

print(f"\nCorrelación de cambios 20Y–30Y:")
print(f"  Con ffill    = {r_ff:.4f}")
print(f"  Observados   = {r_obs:.4f}")
print(f"  Diferencia   = {abs(r_ff - r_obs):.4f}")

# Verificación final de integridad
assert not np.any(np.isnan(V_obs)), "Loadings contienen NaN; verificar datos de entrada."
assert not np.any(np.isnan(var_obs)), "Varianza explicada contiene NaN; verificar datos de entrada."
print("\n✓ Todos los resultados son verificables y libres de valores faltantes.")


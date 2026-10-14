import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from EDA import (
    load_and_prepare_rate_data,
    calculate_missingness_summary,
    calculate_annual_missingness,
)
from pca_analysis import (
    compute_pca_results,
    prepare_working_series,
    LBL,
    TENOR,
    CRISIS_BANDS,
    PCAResult,
)
from forecasting import (
    forecast_random_walk,
    forecast_ou_ar1,
    forecast_var,
    evaluate_models_oos,
    ForecastResult,
    HORIZON_OPTIONS,
)

DATASET_PATH = Path("banxico_auction_rates_monthly.csv")

PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
}

BASE_LAYOUT = {
    "font": {"family": "sans-serif", "size": 11},
    "margin": {"l": 50, "r": 30, "t": 40, "b": 40},
}


@st.cache_data
def get_cached_rates_data(file_path: Path) -> pd.DataFrame:
    return load_and_prepare_rate_data(file_path)


@st.cache_data
def get_cached_missingness_summary(rates_matrix: pd.DataFrame) -> pd.DataFrame:
    return calculate_missingness_summary(rates_matrix)


@st.cache_data
def get_cached_annual_missingness(rates_matrix: pd.DataFrame) -> pd.DataFrame:
    return calculate_annual_missingness(rates_matrix)


@st.cache_data
def get_cached_pca_results(rates_matrix: pd.DataFrame, version: str = "v3") -> tuple[PCAResult, PCAResult]:
    return compute_pca_results(rates_matrix)


@st.cache_data
def get_cached_forecasts(
    scores_key: str,
    scores: pd.DataFrame,
    loadings_flat: bytes,
    horizon: int,
    feature_means: pd.Series | None = None,
    feature_stds: pd.Series | None = None,
    last_observed_rates: pd.Series | None = None,
    is_changes: bool = False,
) -> dict[str, ForecastResult]:
    loadings = np.frombuffer(loadings_flat, dtype=np.float64).reshape(-1, len(LBL))
    return {
        "Random Walk": forecast_random_walk(
            scores, loadings, horizon,
            feature_means=feature_means, feature_stds=feature_stds,
            last_observed_rates=last_observed_rates, is_changes=is_changes,
        ),
        "O-U / AR(1)": forecast_ou_ar1(
            scores, loadings, horizon,
            feature_means=feature_means, feature_stds=feature_stds,
            last_observed_rates=last_observed_rates, is_changes=is_changes,
        ),
        "VAR": forecast_var(
            scores, loadings, horizon,
            feature_means=feature_means, feature_stds=feature_stds,
            last_observed_rates=last_observed_rates, is_changes=is_changes,
        ),
    }


@st.cache_data
def get_cached_oos_eval(
    eval_key: str,
    working_rates: pd.DataFrame,
    horizon: int,
    is_changes: bool = True,
) -> pd.DataFrame:
    return evaluate_models_oos(working_rates, horizon, is_changes=is_changes)


def get_instrument_sort_key(instrument_name: str) -> tuple[int, int]:
    clean_name = instrument_name.strip()
    if "MBONO" in clean_name.upper():
        family_rank = 0
    elif "CETE" in clean_name.upper():
        family_rank = 1
    else:
        family_rank = 2

    import re
    numbers = re.findall(r"\d+", clean_name)
    duration_val = int(numbers[0]) if numbers else 0
    return (family_rank, -duration_val)


def sort_instruments_descending(instruments: list[str]) -> list[str]:
    return sorted(instruments, key=get_instrument_sort_key)


def build_rates_time_series_chart(rates_matrix: pd.DataFrame, selected_instruments: list[str]) -> go.Figure:
    fig = go.Figure()
    ordered_instruments = sort_instruments_descending(selected_instruments)
    for instrument in ordered_instruments:
        if instrument in rates_matrix.columns:
            series = rates_matrix[instrument].dropna()
            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series.values,
                    mode="lines+markers",
                    name=instrument,
                    line={"width": 1.5},
                    marker={"size": 3},
                )
            )
    fig.update_layout(
        title="Historial de Tasas de Interés por Instrumento (Banxico)",
        xaxis_title="Fecha",
        yaxis_title="Tasa Promedio Anualizada (%)",
        xaxis={"gridcolor": "#f0f2f5"},
        yaxis={"gridcolor": "#f0f2f5"},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.3, "xanchor": "center", "x": 0.5},
        **BASE_LAYOUT,
    )
    return fig


def build_missingness_heatmap_chart(rates_matrix: pd.DataFrame, selected_instruments: list[str]) -> go.Figure:
    ordered_top_to_bottom = sort_instruments_descending(selected_instruments)
    visual_y_order = list(reversed(ordered_top_to_bottom))

    sub_matrix = rates_matrix[visual_y_order]
    missing_matrix = sub_matrix.isna().T.astype(int)

    date_strings = [ts.strftime("%Y-%m") for ts in missing_matrix.columns]
    instrument_names = list(missing_matrix.index)

    fig = go.Figure(
        data=go.Heatmap(
            z=missing_matrix.values,
            x=date_strings,
            y=instrument_names,
            colorscale=[[0.0, "#eef4fa"], [1.0, "#1f77b4"]],
            showscale=False,
            hoverongaps=False,
            hovertemplate="Instrumento: %{y}<br>Fecha: %{x}<br>Estado: %{customdata}<extra></extra>",
            customdata=[["Faltante" if val == 1 else "Con Dato" for val in row] for row in missing_matrix.values],
        )
    )

    fig.update_layout(
        title="Mapa de Vacíos",
        xaxis_title="Fecha",
        yaxis_title="Instrumento",
        xaxis={"tickangle": -45, "nticks": 24},
        **BASE_LAYOUT,
    )
    return fig


def build_annual_missingness_bar_chart(annual_df: pd.DataFrame, selected_instruments: list[str]) -> go.Figure:
    ordered_instruments = sort_instruments_descending(selected_instruments)
    filtered_annual = annual_df[ordered_instruments].copy()
    fig = px.bar(
        filtered_annual,
        x=filtered_annual.index,
        y=ordered_instruments,
        barmode="stack",
        title="Conteo Anual de Meses Faltantes por Instrumento",
        labels={"value": "Total Meses Faltantes", "index": "Año", "variable": "Instrumento"},
    )
    fig.update_layout(
        xaxis_title="Año",
        yaxis_title="Meses Faltantes",
        xaxis={"type": "category"},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.3, "xanchor": "center", "x": 0.5},
        **BASE_LAYOUT,
    )
    return fig


def render_instrument_filter(all_instruments: list[str], key_prefix: str) -> list[str]:
    filter_container = st.container()
    with filter_container:
        segmented_col, customize_col = st.columns([1.8, 2.2])

        with segmented_col:
            family_choice = st.segmented_control(
                "Familia de Instrumentos",
                options=["Todos", "Solo MBONOs", "Solo CETEs", "Personalizado"],
                default="Todos",
                key=f"{key_prefix}_family",
                label_visibility="collapsed",
            )

        if family_choice == "Solo MBONOs":
            current_selection = [inst for inst in all_instruments if "MBONO" in inst.upper()]
        elif family_choice == "Solo CETEs":
            current_selection = [inst for inst in all_instruments if "CETE" in inst.upper()]
        elif family_choice == "Todos":
            current_selection = all_instruments
        else:
            with customize_col:
                current_selection = st.multiselect(
                    "Selecciona instrumentos",
                    options=all_instruments,
                    default=all_instruments,
                    key=f"{key_prefix}_custom_picker",
                    label_visibility="collapsed",
                )

        if family_choice != "Personalizado":
            with customize_col:
                st.caption(f"Mostrando {len(current_selection)} instrumentos ({family_choice})")

    return current_selection


def render_eda_page() -> None:
    st.title("Análisis Exploratorio")
    st.caption("Inspección y diagnóstico de los datos con los que se trabaja.")

    if not DATASET_PATH.exists():
        st.error(f"Archivo de datos no encontrado: {DATASET_PATH}")
        st.stop()

    rates_matrix = get_cached_rates_data(DATASET_PATH)
    summary_table = get_cached_missingness_summary(rates_matrix)
    annual_missingness = get_cached_annual_missingness(rates_matrix)

    all_instruments = sort_instruments_descending(list(rates_matrix.columns))

    total_months = len(rates_matrix)
    total_data_points = rates_matrix[all_instruments].size
    missing_data_points = int(rates_matrix[all_instruments].isna().sum().sum())
    missing_pct = (missing_data_points / total_data_points) * 100 if total_data_points > 0 else 0.0

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("Total Meses en Rango", f"{total_months:,}")
    with metric_col2:
        st.metric("Instrumentos Analizados", f"{len(all_instruments)}")
    with metric_col3:
        st.metric("Puntos Faltantes", f"{missing_data_points:,}")
    with metric_col4:
        st.metric("Porcentaje de Vacíos", f"{missing_pct:.2f}%")

    st.markdown("---")

    tab_rates, tab_missing, tab_annual, tab_table = st.tabs([
        "Series de Tiempo",
        "Mapa de Vacíos",
        "Vacíos por Año",
        "Tabla Resumen",
    ])

    with tab_rates:
        selected_rates = render_instrument_filter(all_instruments, "rates")
        if selected_rates:
            rates_chart = build_rates_time_series_chart(rates_matrix, selected_rates)
            st.plotly_chart(rates_chart, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            st.warning("Selecciona al menos un instrumento para visualizar la serie de tiempo.")

    with tab_missing:
        selected_missing = render_instrument_filter(all_instruments, "missing")
        if selected_missing:
            heatmap_chart = build_missingness_heatmap_chart(rates_matrix, selected_missing)
            st.plotly_chart(heatmap_chart, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            st.warning("Selecciona al menos un instrumento para visualizar el mapa de vacíos.")

    with tab_annual:
        st.subheader("Huecos por Instrumento por Año")
        selected_annual = render_instrument_filter(all_instruments, "annual")
        if selected_annual:
            annual_chart = build_annual_missingness_bar_chart(annual_missingness, selected_annual)
            st.plotly_chart(annual_chart, use_container_width=True, config=PLOTLY_CONFIG)

            with st.expander("Ver Matriz de Datos Anual"):
                ordered_annual_cols = sort_instruments_descending(selected_annual)
                annual_display = annual_missingness[ordered_annual_cols].copy()
                annual_display.index.name = "Año"
                st.dataframe(annual_display, use_container_width=True)
        else:
            st.warning("Selecciona al menos un instrumento para visualizar el desglose anual.")

    with tab_table:
        st.subheader("Resumen Cuantitativo de Calidad de Datos")
        selected_table = render_instrument_filter(all_instruments, "table")
        if selected_table:
            filtered_summary = summary_table.loc[selected_table].copy()
            filtered_summary.index.name = "Instrumento"
            st.dataframe(filtered_summary, use_container_width=True)
        else:
            st.warning("Selecciona al menos un instrumento para visualizar la tabla resumen.")


def build_scree_chart(result: PCAResult) -> go.Figure:
    n = len(result.explained_variance)
    components = [f"PC{i + 1}" for i in range(n)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=components,
        y=result.explained_variance,
        name="% varianza",
        marker_color="lightsteelblue",
        yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=components,
        y=result.explained_variance.cumsum(),
        mode="lines+markers",
        name="% acumulado",
        line={"color": "crimson"},
        yaxis="y2",
    ))
    fig.add_hline(y=95, line_dash="dash", line_color="grey", annotation_text="95%", yref="y2")
    fig.update_layout(
        title=f"Scree Plot — PCA sobre {result.label}",
        xaxis_title="Componente",
        yaxis={"title": "% varianza"},
        yaxis2={"title": "% acumulado", "overlaying": "y", "side": "right", "range": [0, 105]},
        legend={"orientation": "h", "y": -0.25},
        **BASE_LAYOUT,
    )
    return fig


def build_loadings_chart(result: PCAResult, n_pcs: int = 3) -> go.Figure:
    palette = px.colors.qualitative.Plotly
    fig = go.Figure()
    for j in range(n_pcs):
        fig.add_trace(go.Scatter(
            x=LBL,
            y=result.loadings[j],
            mode="lines+markers",
            name=f"PC{j + 1} ({result.explained_variance[j]:.1f}%)",
            line={"color": palette[j % len(palette)]},
            marker={"size": 7},
        ))
    fig.add_hline(y=0, line_color="grey", line_width=0.8)
    fig.update_layout(
        title=f"Loadings por Plazo — PCA sobre {result.label}",
        xaxis_title="Plazo",
        yaxis_title="Loading",
        legend={"orientation": "h", "y": -0.25},
        **BASE_LAYOUT,
    )
    return fig


def build_scores_chart(result: PCAResult, n_pcs: int = 3) -> go.Figure:
    palette = px.colors.qualitative.Plotly
    fig = go.Figure()
    for j in range(n_pcs):
        col = f"PC{j + 1}"
        fig.add_trace(go.Scatter(
            x=result.scores.index,
            y=result.scores[col],
            mode="lines",
            name=f"PC{j + 1} ({result.explained_variance[j]:.1f}%)",
            line={"color": palette[j % len(palette)], "width": 1.2},
        ))

    for start, end, name in CRISIS_BANDS:
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="red", opacity=0.06,
            layer="below", line_width=0,
            annotation_text=name,
            annotation_position="top left",
            annotation_font_size=9,
        )

    fig.add_hline(y=0, line_color="grey", line_width=0.5)
    fig.update_layout(
        title=f"Scores en el Tiempo — PCA sobre {result.label}",
        xaxis_title="Fecha",
        yaxis_title="Score",
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.2},
        **BASE_LAYOUT,
    )
    return fig


def build_cumulative_variance_table(lvl: PCAResult, chg: PCAResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Niveles (%)": [f"{v:.1f}" for v in lvl.explained_variance.cumsum()],
            "Cambios (%)": [f"{v:.1f}" for v in chg.explained_variance.cumsum()],
        },
        index=[f"PC{i + 1}" for i in range(len(lvl.explained_variance))],
    )


def render_modeling_page() -> None:
    st.title("Modelado de los Datos")
    st.caption("Análisis de Componentes Principales (PCA) sobre la curva de tasas Banxico.")

    if not DATASET_PATH.exists():
        st.error(f"Archivo de datos no encontrado: {DATASET_PATH}")
        st.stop()

    rates_matrix = get_cached_rates_data(DATASET_PATH)

    try:
        result_lvl, result_chg = get_cached_pca_results(rates_matrix)
    except ValueError as exc:
        st.error(f"Error al ejecutar PCA: {exc}")
        st.stop()

    ini = result_chg.scores.index.min().strftime("%Y-%m")
    fin = result_chg.scores.index.max().strftime("%Y-%m")
    n_months = len(result_chg.scores)

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.metric("Serie de Trabajo", f"{ini} → {fin}")
    with mc2:
        st.metric("Meses Analizados", f"{n_months}")
    with mc3:
        pcs_95 = int((result_chg.explained_variance.cumsum() >= 95).argmax()) + 1
        st.metric("PCs para 95% varianza (cambios)", f"{pcs_95}")

    st.markdown("---")

    mode = st.segmented_control(
        "Versión del PCA",
        options=["Cambios Mensuales", "Niveles"],
        default="Cambios Mensuales",
        key="pca_mode",
    )
    active = result_chg if mode == "Cambios Mensuales" else result_lvl

    tab_scree, tab_loadings, tab_scores, tab_tabla = st.tabs([
        "Scree Plot", "Loadings", "Scores en el Tiempo", "Varianza Acumulada",
    ])

    with tab_scree:
        st.plotly_chart(build_scree_chart(active), use_container_width=True, config=PLOTLY_CONFIG)

    with tab_loadings:
        n_pcs = st.slider("Componentes a mostrar", min_value=1, max_value=9, value=3, key="pca_n_loadings")
        st.plotly_chart(build_loadings_chart(active, n_pcs=n_pcs), use_container_width=True, config=PLOTLY_CONFIG)

    with tab_scores:
        n_pcs_sc = st.slider("Componentes a mostrar", min_value=1, max_value=9, value=3, key="pca_n_scores")
        st.plotly_chart(build_scores_chart(active, n_pcs=n_pcs_sc), use_container_width=True, config=PLOTLY_CONFIG)

    with tab_tabla:
        st.subheader("Varianza Explicada Acumulada")
        st.dataframe(
            build_cumulative_variance_table(result_lvl, result_chg),
            use_container_width=True,
        )
        with st.expander("Loadings completos (PC1–PC3)"):
            st.dataframe(
                pd.DataFrame(
                    active.loadings[:3].round(3),
                    index=["PC1", "PC2", "PC3"],
                    columns=LBL,
                ),
                use_container_width=True,
            )


def build_curve_forecast_chart(
    forecasts: dict[str, ForecastResult],
    step: int,
    last_observed_rates: pd.Series,
) -> go.Figure:
    if not forecasts:
        raise ValueError("Forecasts dictionary cannot be empty.")
    if step < 1:
        raise ValueError(f"Step must be at least 1, received: {step}")

    palette = px.colors.qualitative.Plotly
    fig = go.Figure()

    last_date = list(forecasts.values())[0].last_observed_date
    observed_curve = pd.Series(last_observed_rates.values, index=LBL)

    fig.add_trace(go.Scatter(
        x=LBL,
        y=observed_curve.values,
        mode="lines+markers",
        name=f"Observado ({last_date.strftime('%Y-%m')})",
        line={"color": "black", "width": 2.2},
        marker={"size": 8, "color": "black"},
    ))

    for idx, (model_name, fc_result) in enumerate(forecasts.items()):
        if step <= len(fc_result.curve_forecasts):
            forecast_curve = fc_result.curve_forecasts.iloc[step - 1]
            fc_date = fc_result.forecast_dates[step - 1].strftime("%Y-%m")
            fig.add_trace(go.Scatter(
                x=LBL,
                y=forecast_curve.values,
                mode="lines+markers",
                name=f"{model_name} (h={step}, {fc_date})",
                line={"color": palette[(idx + 1) % len(palette)], "width": 1.8, "dash": "dash"},
                marker={"size": 6},
            ))

    fig.update_layout(
        title=f"Curva de Tasas Pronosticada — Paso h={step}",
        xaxis_title="Plazo",
        yaxis_title="Tasa de Interés (%)",
        legend={"orientation": "h", "y": -0.25},
        **BASE_LAYOUT,
    )
    return fig


def build_score_forecast_chart(forecasts: dict[str, ForecastResult], pc_index: int) -> go.Figure:
    if not forecasts:
        raise ValueError("Forecasts dictionary cannot be empty.")
    if pc_index < 0 or pc_index > 2:
        raise ValueError(f"pc_index must be between 0 and 2, got: {pc_index}")

    palette = px.colors.qualitative.Plotly
    fig = go.Figure()
    col = f"PC{pc_index + 1}"

    for idx, (model_name, fc_result) in enumerate(forecasts.items()):
        if col in fc_result.score_forecasts.columns:
            fig.add_trace(go.Scatter(
                x=fc_result.forecast_dates,
                y=fc_result.score_forecasts[col],
                mode="lines+markers",
                name=model_name,
                line={"color": palette[idx % len(palette)], "dash": "dot"},
                marker={"size": 5},
            ))

    pc_labels = {0: "PC1 — Nivel", 1: "PC2 — Pendiente", 2: "PC3 — Curvatura"}
    fig.update_layout(
        title=f"Forecast de Scores: {pc_labels.get(pc_index, col)}",
        xaxis_title="Fecha",
        yaxis_title="Score",
        legend={"orientation": "h", "y": -0.25},
        **BASE_LAYOUT,
    )
    return fig


def build_instrument_fan_chart(
    working_rates: pd.DataFrame,
    forecasts: dict[str, ForecastResult],
    instrument_name: str,
    model_name: str,
    lookback: int = 6,
) -> go.Figure:
    if not isinstance(working_rates, pd.DataFrame):
        raise TypeError("working_rates must be a DataFrame.")
    if instrument_name not in working_rates.columns:
        raise KeyError(f"Instrument '{instrument_name}' not found in working rates.")
    if model_name not in forecasts:
        raise KeyError(f"Model '{model_name}' not found in forecasts.")

    fc = forecasts[model_name]
    inst_idx = list(working_rates.columns).index(instrument_name)
    lbl = LBL[inst_idx]

    hist_series = working_rates[instrument_name].dropna().tail(lookback)
    hist_dates = hist_series.index
    hist_values = hist_series.values
    last_date = hist_dates[-1]
    last_val = hist_values[-1]

    fc_dates = fc.forecast_dates
    fc_vals = fc.curve_forecasts[lbl].values
    fc_stds = fc.curve_stds[lbl].values if not fc.curve_stds.empty else np.zeros_like(fc_vals)

    conn_dates = [last_date] + list(fc_dates)
    conn_mean = np.array([last_val] + list(fc_vals))
    conn_std = np.array([0.0] + list(fc_stds))

    up_95 = conn_mean + 1.95996 * conn_std
    lo_95 = conn_mean - 1.95996 * conn_std
    up_80 = conn_mean + 1.28155 * conn_std
    lo_80 = conn_mean - 1.28155 * conn_std
    up_50 = conn_mean + 0.67449 * conn_std
    lo_50 = conn_mean - 0.67449 * conn_std

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist_dates,
        y=hist_values,
        mode="lines+markers",
        name=f"Histórico ({lookback}m)",
        line={"color": "#111827", "width": 2.5},
        marker={"size": 7, "color": "#111827"},
        hovertemplate="%{x|%Y-%m}: %{y:.2f}%<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=conn_dates,
        y=up_95,
        mode="lines",
        line={"width": 0},
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=conn_dates,
        y=lo_95,
        mode="lines",
        line={"width": 0},
        fill="tonexty",
        fillcolor="rgba(37, 99, 235, 0.12)",
        name="IC 95% (Densidad)",
    ))

    fig.add_trace(go.Scatter(
        x=conn_dates,
        y=up_80,
        mode="lines",
        line={"width": 0},
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=conn_dates,
        y=lo_80,
        mode="lines",
        line={"width": 0},
        fill="tonexty",
        fillcolor="rgba(37, 99, 235, 0.20)",
        name="IC 80% (Densidad)",
    ))

    fig.add_trace(go.Scatter(
        x=conn_dates,
        y=up_50,
        mode="lines",
        line={"width": 0},
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=conn_dates,
        y=lo_50,
        mode="lines",
        line={"width": 0},
        fill="tonexty",
        fillcolor="rgba(37, 99, 235, 0.32)",
        name="IC 50% (Densidad)",
    ))

    fig.add_trace(go.Scatter(
        x=conn_dates,
        y=conn_mean,
        mode="lines+markers",
        name=f"Pronóstico {model_name}",
        line={"color": "#1d4ed8", "width": 2.2, "dash": "dash"},
        marker={"size": 6, "color": "#1d4ed8"},
        hovertemplate="%{x|%Y-%m}: %{y:.2f}%<extra></extra>",
    ))

    fig.add_shape(
        type="line",
        x0=last_date,
        x1=last_date,
        y0=0,
        y1=1,
        yref="paper",
        line={"dash": "dot", "color": "gray", "width": 1.5},
    )
    fig.add_annotation(
        x=last_date,
        y=1.02,
        yref="paper",
        text="Último dato real",
        showarrow=False,
        xanchor="right",
        font={"size": 10, "color": "gray"},
    )

    fig.update_layout(
        title=f"Trayectoria y Cono de Probabilidad — {instrument_name}",
        xaxis_title="Fecha",
        yaxis_title="Tasa de Interés (%)",
        yaxis={"tickformat": ".2f", "ticksuffix": "%"},
        legend={"orientation": "h", "y": -0.22},
        height=520,
        **BASE_LAYOUT,
    )
    return fig


def render_forecasting_page() -> None:
    st.title("Forecasting de la Curva de Tasas")
    st.caption(
        "Tres modelos compiten: Random Walk (benchmark), O-U / AR(1) (mean reversion físico) y VAR(p) "
        "(acoplamiento entre modos normales de la curva)."
    )

    if not DATASET_PATH.exists():
        st.error(f"Archivo de datos no encontrado: {DATASET_PATH}")
        st.stop()

    rates_matrix = get_cached_rates_data(DATASET_PATH)
    working_rates = prepare_working_series(rates_matrix)

    try:
        result_lvl, result_chg = get_cached_pca_results(rates_matrix, version="v3")
    except ValueError as exc:
        st.error(f"Error al ejecutar PCA: {exc}")
        st.stop()

    col_hz, col_ver = st.columns([2, 2])
    with col_hz:
        horizon_label = st.selectbox(
            "Horizonte de pronóstico",
            options=list(HORIZON_OPTIONS.keys()),
            index=1,
            key="fc_horizon",
        )
    with col_ver:
        pca_version = st.segmented_control(
            "Versión del PCA",
            options=["Cambios Mensuales", "Niveles"],
            default="Cambios Mensuales",
            key="fc_pca_version",
        )

    horizon = HORIZON_OPTIONS[horizon_label]
    is_changes = (pca_version == "Cambios Mensuales")
    active = result_chg if is_changes else result_lvl

    if is_changes:
        diff_data = working_rates.diff().dropna()
        f_means = active.feature_means if active.feature_means is not None else diff_data.mean()
        f_stds = active.feature_stds if active.feature_stds is not None else diff_data.std()
    else:
        f_means = active.feature_means if active.feature_means is not None else working_rates.mean()
        f_stds = active.feature_stds if active.feature_stds is not None else working_rates.std()

    loadings_flat = active.loadings.astype("float64").tobytes()
    scores_key = f"{pca_version}_{len(active.scores)}_{horizon}_v3"
    last_rates = working_rates.iloc[-1]

    with st.spinner("Calculando pronósticos..."):
        forecasts = get_cached_forecasts(
            scores_key,
            active.scores,
            loadings_flat,
            horizon,
            feature_means=f_means,
            feature_stds=f_stds,
            last_observed_rates=last_rates,
            is_changes=is_changes,
        )

    st.markdown("---")

    tab_fan, tab_curve, tab_scores_fc, tab_eval = st.tabs([
        "Densidad por Instrumento (Fan Chart)",
        "Curva Pronosticada",
        "Scores Pronosticados",
        "Evaluación OOS",
    ])

    with tab_fan:
        st.subheader("Cono de Incertidumbre Predictiva")
        st.caption(
            "Muestra los últimos datos reales observados de un instrumento conectándose continuamente con su cono de incertidumbre predictiva a los horizontes futuros."
        )
        col_inst, col_mod, col_look = st.columns([2, 2, 1])
        with col_inst:
            selected_instrument = st.selectbox("Instrumento", options=TENOR, index=0, key="fan_inst_sel")
        with col_mod:
            selected_model = st.radio("Modelo", options=["VAR", "O-U / AR(1)", "Random Walk"], horizontal=True, key="fan_mod_sel")
        with col_look:
            lookback_val = st.slider("Histórico (meses)", min_value=3, max_value=24, value=6, key="fan_lookback")

        st.plotly_chart(
            build_instrument_fan_chart(
                working_rates,
                forecasts,
                selected_instrument,
                selected_model,
                lookback=lookback_val,
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    with tab_curve:
        step = st.slider(
            "Paso hacia adelante (meses)",
            min_value=1, max_value=horizon, value=min(3, horizon),
            key="fc_step_slider",
        )
        st.plotly_chart(
            build_curve_forecast_chart(forecasts, step, last_rates),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    with tab_scores_fc:
        pc_sel = st.radio(
            "Componente",
            options=["PC1 — Nivel", "PC2 — Pendiente", "PC3 — Curvatura"],
            horizontal=True,
            key="fc_pc_radio",
        )
        pc_idx = {"PC1 — Nivel": 0, "PC2 — Pendiente": 1, "PC3 — Curvatura": 2}[pc_sel]
        st.plotly_chart(
            build_score_forecast_chart(forecasts, pc_idx),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    with tab_eval:
        st.subheader("RMSE fuera de muestra (walk-forward)")
        st.caption(
            "En cada paso de la evaluación, el PCA y la normalización se reentrenan "
            "exclusivamente con los datos pasados, evaluando el pronóstico desestandarizado contra las tasas reales de mercado de Banxico."
        )
        try:
            eval_key = f"{pca_version}_{len(working_rates)}_{horizon}"
            oos_table = get_cached_oos_eval(eval_key, working_rates, horizon, is_changes=is_changes)
            st.dataframe(oos_table, use_container_width=True)
        except ValueError as exc:
            st.warning(f"No hay suficientes datos para evaluación OOS: {exc}")

        with st.expander("Tablas de pronóstico por modelo (tasas en %)"):
            for model_name, fc_result in forecasts.items():
                st.markdown(f"**{model_name}**")
                display_df = fc_result.curve_forecasts.copy()
                display_df.index = fc_result.forecast_dates.strftime("%Y-%m")
                display_df.index.name = "Fecha"
                st.dataframe(display_df.round(4), use_container_width=True)


def main() -> None:

    st.set_page_config(
        page_title="Banxico Analytics",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "current_page" not in st.session_state:
        st.session_state.current_page = "Análisis Exploratorio"

    with st.sidebar:
        st.header("Banxico Analytics")

        eda_type = "primary" if st.session_state.current_page == "Análisis Exploratorio" else "secondary"
        if st.button("Análisis Exploratorio", use_container_width=True, type=eda_type):
            st.session_state.current_page = "Análisis Exploratorio"
            st.rerun()

        model_type = "primary" if st.session_state.current_page == "Modelado de los Datos" else "secondary"
        if st.button("Modelado de los Datos", use_container_width=True, type=model_type):
            st.session_state.current_page = "Modelado de los Datos"
            st.rerun()

        fc_type = "primary" if st.session_state.current_page == "Forecasting" else "secondary"
        if st.button("Forecasting", use_container_width=True, type=fc_type):
            st.session_state.current_page = "Forecasting"
            st.rerun()

    if st.session_state.current_page == "Análisis Exploratorio":
        render_eda_page()
    elif st.session_state.current_page == "Modelado de los Datos":
        render_modeling_page()
    else:
        render_forecasting_page()


if __name__ == "__main__":
    main()



#   python -m streamlit run Dashboard.py

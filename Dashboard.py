from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from EDA import (
    load_and_prepare_rate_data,
    calculate_missingness_summary,
    calculate_annual_missingness,
)
from pca_analysis import (
    compute_pca_results,
    LBL,
    TENOR,
    CRISIS_BANDS,
    PCAResult,
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
def get_cached_pca_results(rates_matrix: pd.DataFrame) -> tuple[PCAResult, PCAResult]:
    return compute_pca_results(rates_matrix)


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
    st.title("Análisis Exploratorio de Datos (EDA)")
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


def main() -> None:

    st.set_page_config(
        page_title="Banxico Analytics",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "current_page" not in st.session_state:
        st.session_state.current_page = "Análisis Exploratorio (EDA)"

    with st.sidebar:
        st.header("Banxico Analytics")

        eda_type = "primary" if st.session_state.current_page == "Análisis Exploratorio (EDA)" else "secondary"
        if st.button("Análisis Exploratorio (EDA)", use_container_width=True, type=eda_type):
            st.session_state.current_page = "Análisis Exploratorio (EDA)"
            st.rerun()

        model_type = "primary" if st.session_state.current_page == "Modelado de los Datos" else "secondary"
        if st.button("Modelado de los Datos", use_container_width=True, type=model_type):
            st.session_state.current_page = "Modelado de los Datos"
            st.rerun()

    if st.session_state.current_page == "Análisis Exploratorio (EDA)":
        render_eda_page()
    else:
        render_modeling_page()


if __name__ == "__main__":
    main()



#   python -m streamlit run Dashboard.py

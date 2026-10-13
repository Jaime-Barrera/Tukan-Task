from pathlib import Path
import logging
import pandas as pd
import matplotlib.pyplot as plt

REQUIRED_COLUMNS: list[str] = ["DATE", "INSTRUMENT", "VALUE_PCT"]


def validate_dataframe(df: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in dataset: {missing_columns}")
    return df


def load_and_prepare_rate_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    raw_data = pd.read_csv(file_path)
    validated_data = validate_dataframe(raw_data, REQUIRED_COLUMNS).copy()
    validated_data["DATE"] = pd.to_datetime(validated_data["DATE"])

    pivoted_series = validated_data.pivot(
        index="DATE",
        columns="INSTRUMENT",
        values="VALUE_PCT"
    ).sort_index()

    complete_monthly_index = pd.date_range(
        start=pivoted_series.index.min(),
        end=pivoted_series.index.max(),
        freq="MS"
    )

    return pivoted_series.reindex(complete_monthly_index)


def calculate_missingness_summary(pivoted_rates: pd.DataFrame) -> pd.DataFrame:
    summary_table = pd.DataFrame(
        {
            "Total_Meses": len(pivoted_rates),
            "Meses_Con_Dato": pivoted_rates.notna().sum(),
            "Meses_Faltantes": pivoted_rates.isna().sum(),
            "Pct_Faltante": (pivoted_rates.isna().mean() * 100).round(2),
            "Primer_Mes_Valido": pivoted_rates.apply(
                lambda s: s.first_valid_index().strftime("%Y-%m") if s.first_valid_index() is not None else "N/A"
            ),
            "Ultimo_Mes_Valido": pivoted_rates.apply(
                lambda s: s.last_valid_index().strftime("%Y-%m") if s.last_valid_index() is not None else "N/A"
            ),
            "Max_Meses_Consecutivos_Faltantes": pivoted_rates.apply(
                lambda s: s.isna().astype(int).groupby(s.notna().cumsum()).sum().max()
            ),
        }
    )
    return summary_table.sort_values(by="Meses_Faltantes", ascending=False)


def calculate_annual_missingness(pivoted_rates: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(pivoted_rates.index, pd.DatetimeIndex):
        raise TypeError("pivoted_rates DataFrame must have a DatetimeIndex.")
    return pivoted_rates.isna().groupby(pivoted_rates.index.year).sum()


def plot_rates_and_missing_values(pivoted_rates: pd.DataFrame, summary_table: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(15, 12))
    grid = fig.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=[2.2, 1.2, 1.4],
        top=0.95,
        bottom=0.06,
        left=0.08,
        right=0.88,
        hspace=0.45
    )

    ax_rates = fig.add_subplot(grid[0])
    ax_heatmap = fig.add_subplot(grid[1])
    ax_table = fig.add_subplot(grid[2])

    for instrument in pivoted_rates.columns:
        series = pivoted_rates[instrument]
        ax_rates.plot(
            series.index,
            series.values,
            marker="o",
            markersize=2,
            linewidth=1.2,
            label=instrument
        )

    ax_rates.set_title("Tasas de Interes por Instrumento (Banxico)", fontsize=12, fontweight="bold", pad=8)
    ax_rates.set_xlabel("Fecha", fontsize=9)
    ax_rates.set_ylabel("Tasa Promedio Anualizada (%)", fontsize=9)
    ax_rates.grid(True, linestyle="--", alpha=0.5)
    ax_rates.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, fontsize=8)

    missing_matrix = pivoted_rates.isna().T

    cax = ax_heatmap.imshow(
        missing_matrix.values,
        aspect="auto",
        cmap="Blues",
        interpolation="nearest"
    )

    num_instruments = len(missing_matrix.index)
    ax_heatmap.set_yticks(range(num_instruments))
    ax_heatmap.set_yticklabels(missing_matrix.index, fontsize=8)

    num_dates = len(missing_matrix.columns)
    step = max(1, num_dates // 12)
    selected_indices = list(range(0, num_dates, step))
    if (num_dates - 1) not in selected_indices:
        selected_indices.append(num_dates - 1)

    ax_heatmap.set_xticks(selected_indices)
    ax_heatmap.set_xticklabels(
        [missing_matrix.columns[idx].strftime("%Y-%m") for idx in selected_indices],
        rotation=35,
        ha="right",
        fontsize=8
    )

    ax_heatmap.set_title("Mapa de Vacios: Meses Faltantes de Tasa (Azul = Dato Faltante)", fontsize=11, fontweight="bold", pad=8)
    ax_heatmap.set_xlabel("Fecha", fontsize=9)
    ax_heatmap.set_ylabel("Instrumento", fontsize=9)

    cbar = fig.colorbar(cax, ax=ax_heatmap, orientation="vertical", pad=0.015, fraction=0.02)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Con Dato", "Faltante"])
    cbar.ax.tick_params(labelsize=8)

    ax_table.axis("off")
    table_display = summary_table.reset_index()
    table_display.columns = [
        "Instrumento",
        "Total Meses",
        "Meses Con Dato",
        "Meses Faltantes",
        "% Faltante",
        "Primer Mes",
        "Ultimo Mes",
        "Max Gap Consecutivo"
    ]

    rendered_table = ax_table.table(
        cellText=table_display.values,
        colLabels=table_display.columns,
        loc="center",
        cellLoc="center"
    )
    rendered_table.auto_set_font_size(False)
    rendered_table.set_fontsize(8)
    rendered_table.scale(1.0, 1.3)

    for (row_idx, _), cell in rendered_table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#2b5c8f")
            cell.set_text_props(color="white", weight="bold")
        elif row_idx % 2 == 0:
            cell.set_facecolor("#f2f5f9")

    ax_table.set_title("Resumen Cuantitativo de Meses Faltantes por Instrumento", fontsize=11, fontweight="bold", pad=10)

    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    dataset_path = Path("banxico_auction_rates_monthly.csv")

    rates_matrix = load_and_prepare_rate_data(dataset_path)
    summary_report = calculate_missingness_summary(rates_matrix)
    annual_missingness = calculate_annual_missingness(rates_matrix)

    logging.info("Tabla de resumen de meses faltantes por instrumento:\n\n%s\n", summary_report.to_string())
    logging.info("Meses faltantes por instrumento por año:\n\n%s\n", annual_missingness.to_string())

    plot_rates_and_missing_values(rates_matrix, summary_report)
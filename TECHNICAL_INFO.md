# Technical Documentation

This document describes the modules, functions, and architecture of the Banxico Auction Rates Explorer repository.

---

## 1. Module: `EDA.py`

This module provides data loading, schema validation, indexing, and statistical calculation routines for Banxico auction rates.

### Constants

- `REQUIRED_COLUMNS` (`list[str]`): List of required columns `["DATE", "INSTRUMENT", "VALUE_PCT"]`.

### Functions

#### `validate_dataframe(df: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame`
- **Purpose**: Verifies that all expected columns exist in the incoming DataFrame. Fails fast by raising `ValueError` if any required column is missing.
- **Parameters**:
  - `df`: Input DataFrame to check.
  - `expected_columns`: List of required column names.
- **Returns**: The validated DataFrame.

#### `load_and_prepare_rate_data(file_path: Path) -> pd.DataFrame`
- **Purpose**: Reads CSV data from the provided path, enforces date formatting, pivots the data on `DATE` and `INSTRUMENT`, and reindexes across a complete monthly date sequence (`MS` frequency).
- **Parameters**:
  - `file_path`: Path pointing to the CSV file.
- **Returns**: Pivoted DataFrame indexed by monthly dates with instruments as columns.

#### `calculate_missingness_summary(pivoted_rates: pd.DataFrame) -> pd.DataFrame`
- **Purpose**: Computes missing data statistics per instrument, including count of missing observations, total observations, and missingness percentage.
- **Parameters**:
  - `pivoted_rates`: Pivoted DataFrame containing instruments as columns.
- **Returns**: DataFrame sorted descending by missing count.

#### `calculate_annual_missingness(pivoted_rates: pd.DataFrame) -> pd.DataFrame`
- **Purpose**: Calculates the proportion of missing months for each instrument grouped by calendar year.
- **Parameters**:
  - `pivoted_rates`: Pivoted DataFrame indexed by DatetimeIndex.
- **Returns**: DataFrame indexed by Year with instruments as columns representing missing rates.

#### `calculate_distribution_metrics(pivoted_rates: pd.DataFrame) -> pd.DataFrame`
- **Purpose**: Generates descriptive summary statistics (mean, median, standard deviation, IQR, min, max, skewness) per instrument.
- **Parameters**:
  - `pivoted_rates`: Pivoted DataFrame of rates.
- **Returns**: DataFrame summarizing distribution metrics.

#### `calculate_outlier_bounds(series: pd.Series) -> tuple[float, float]`
- **Purpose**: Computes upper and lower outlier thresholds using the Interquartile Range (IQR) method ($Q_1 - 1.5 \times IQR$ and $Q_3 + 1.5 \times IQR$).
- **Parameters**:
  - `series`: Numerical pandas Series.
- **Returns**: Tuple of `(lower_bound, upper_bound)`.

---

## 2. Module: `Dashboard.py`

Streamlit-based user interface displaying interactive dashboards and analytical views.

### Functions

#### `get_cached_rates_data(file_path: Path) -> pd.DataFrame`
- **Purpose**: Cached wrapper (`@st.cache_data`) around `load_and_prepare_rate_data`.

#### `get_cached_missingness_summary(rates_matrix: pd.DataFrame) -> pd.DataFrame`
- **Purpose**: Cached wrapper (`@st.cache_data`) around `calculate_missingness_summary`.

#### `get_cached_annual_missingness(rates_matrix: pd.DataFrame) -> pd.DataFrame`
- **Purpose**: Cached wrapper (`@st.cache_data`) around `calculate_annual_missingness`.

#### `get_instrument_sort_key(instrument_name: str) -> tuple[int, int]`
- **Purpose**: Generates sorting tuples for financial instruments to order CETES, BONDES, and UDIBONOS logically by maturity length.

#### `render_overview_section(rates_matrix: pd.DataFrame) -> None`
- **Purpose**: Renders the high-level metrics cards and dataset overview.

#### `render_time_series_section(rates_matrix: pd.DataFrame) -> None`
- **Purpose**: Renders multi-series interactive time-series plots with date filtering.

#### `render_data_quality_section(rates_matrix: pd.DataFrame) -> None`
- **Purpose**: Displays missing data summaries, annual heatmaps, and coverage matrices.

#### `render_distribution_section(rates_matrix: pd.DataFrame) -> None`
- **Purpose**: Displays box plots, violin plots, and statistical metrics across instruments.

---

## 3. Module: `data profiling.py`

Script responsible for generating standalone HTML profiling reports using profiling libraries over `banxico_auction_rates_monthly.csv`.

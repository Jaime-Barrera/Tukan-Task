# Banxico Auction Rates Explorer & Dashboard

Interactive analysis and dashboard exploring Banxico monthly auction rates for government securities (CETES, BONDES, UDIBONOS, BONOS).

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

## Installation

```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Running the Dashboard

```bash
streamlit run Dashboard.py
```

## Running the Exploratory Data Analysis

```bash
python EDA.py
```

## Project Structure

- `Dashboard.py`: Interactive Streamlit application with analytical views and charts.
- `EDA.py`: Core analytical functions, data cleaning, and statistical calculations.
- `data profiling.py`: Data profiling generation script.
- `banxico_auction_rates_monthly.csv`: Dataset containing historical auction rates.
- `TECHNICAL_INFO.md`: Complete technical documentation of functions, modules, and architecture.

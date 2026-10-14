# Banxico Auction Rates Explorer & Dashboard

Interactive analysis and dashboard exploring Banxico monthly auction rates for government securities (CETEs and MBONOs).

## Live Application

The interactive dashboard is deployed on Streamlit Community Cloud:
- URL: https://banxico-rates.streamlit.app/

## Project Modules

- `Dashboard.py`: Main entry point for the Streamlit web application.
- `EDA.py`: Ingestion, schema validation, and missingness analysis routines.
- `pca_analysis.py`: Principal Component Analysis (PCA) and yield curve factor extraction.
- `forecasting.py`: Econometric forecasting models (Random Walk, Ornstein-Uhlenbeck / AR(1), VAR) and uncertainty cone estimation.
- `banxico_auction_rates_monthly.csv`: Historical monthly auction rates dataset.
- `requirements.txt`: Python package dependencies for Streamlit Cloud deployment.

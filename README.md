# Banxico Auction Rates Explorer & Dashboard

Interactive analysis and dashboard exploring Banxico monthly auction rates for government securities (CETES, BONDES, UDIBONOS, BONOS).

## Live App

The application is deployed and available directly on Streamlit Community Cloud:
- Access the dashboard via your Streamlit Community Cloud workspace or public deployment link.

## Project Structure

- `Dashboard.py`: Main entry point for the Streamlit application.
- `EDA.py`: Core analytical routines, schema validation, and statistical metrics.
- `data profiling.py`: Data profiling report generator.
- `banxico_auction_rates_monthly.csv`: Dataset containing historical auction rates.
- `requirements.txt`: Python package dependencies used by Streamlit Cloud to build the environment.
- `TECHNICAL_INFO.md`: Technical documentation covering module architecture and function contracts.

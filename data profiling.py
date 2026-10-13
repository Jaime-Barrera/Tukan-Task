import pandas as pd
from data_profiling import ProfileReport

df = pd.read_csv('banxico_auction_rates_monthly.csv')

reporte = ProfileReport (df, title="Analisis Exploratorio", explorative = True)

reporte.to_file("Data Profiling.html")


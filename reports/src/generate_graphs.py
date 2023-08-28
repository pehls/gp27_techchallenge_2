import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from src.indicators import generate_graph

def _grafico_historico(df, crossovers):
    return generate_graph(df, crossovers, just_candles=False, just_return=True)

def _seasonal_decompose(series):
    result = seasonal_decompose(series, model='additive', period=5)
    return result.plot()
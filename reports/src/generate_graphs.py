import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from src.indicators import generate_graph


def _grafico_historico(df, crossovers):
    return generate_graph(df, crossovers, just_candles=False, just_return=True)

def _seasonal_decompose(series, period=5):
    result = seasonal_decompose(series, model='additive', period=period)
    return result.plot()

def _adf(df):
    df_ts = pd.DataFrame(df['Close'].to_list(), columns=['close'], index=df.index)
    df_ts.index = pd.to_datetime(df['Date'], dayfirst=True)
    ma = df_ts.rolling(12).mean()

    fig, ax = plt.subplots()
    df_ts.plot(ax=ax, legend=False)
    ma.plot(ax=ax, legend=False)

    return fig, df_ts.close.values

def _adf_diff(df):
    df_ts = pd.DataFrame(df['Close'].to_list(), columns=['close'])
    df_ts.index = pd.to_datetime(df['Date'], dayfirst=True)

    df_ts = df_ts.diff(1)

    ma = df_ts.rolling(12).mean()
    std = df_ts.rolling(12).std()

    fig, ax = plt.subplots()
    df_ts.plot(ax=ax, legend=False)
    ma.plot(ax=ax, legend=False)
    std.plot(ax=ax, legend=False)

    return fig, df_ts.dropna()['close'].values

def _grafico_acf(df):
    return plot_acf(df['Close'])

# def _grafico_pacf(df):
#     return plot_pacf(df['Close'])

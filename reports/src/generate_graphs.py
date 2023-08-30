import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, SeasonalNaive, SeasonalWindowAverage
from src.indicators import generate_graph
import src.get_data as get_data 


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

def _grafico_models_ts():
    train, test, h = get_data._get_data_for_models_ts()
    
    model_all = StatsForecast(models=[
        SeasonalNaive(season_length=7),
        SeasonalWindowAverage(season_length=7, window_size=2),
        AutoARIMA(season_length=7)], freq='D', n_jobs=-1)
    
    model_all.fit(train)

    forecast_all = model_all.predict(h=h)
    forecast_all = forecast_all.reset_index().merge(test, on=['ds', 'unique_id'], how='left')

    return model_all.plot(train, forecast_all, unique_ids=['IBOV'], engine='matplotlib')

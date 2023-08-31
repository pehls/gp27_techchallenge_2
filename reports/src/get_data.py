import pandas as pd
import numpy as np
import config
import streamlit as st
from statsmodels.tsa.stattools import adfuller

from src.indicators import Indicators

@st.cache_data
def _df_ibovespa():
    # deixar data como indice
    df = pd\
        .read_csv(f'{config.BASE_PATH}/raw/dados_ibovespa.csv')\
        .rename(columns={
            'Data':'Date'
            , 'Último':'Close'
            , 'Abertura':'Open'
            , 'Máxima':'High'
            , 'Mínima':'Low'
            , 'Vol.':'Volume'
        })
    df['Adj Close'] =  df['Close']
    df.Date = pd.to_datetime(df['Date']).dt.date
    df = df.sort_values(['Date'])
    df['Datetime'] = pd.to_datetime(df['Date'])
    df.Open = df.Open.astype(float)
    df.Close = df.Close.astype(float)
    df.High = df.High.astype(float)
    df.Low = df.Low.astype(float)
    df.Volume = df.Volume.str.replace('M','000000').str.replace(',','').str.replace('K','000')
    df.Volume = df.Volume.astype(float)
    df['Base Volume'] = df.Volume.astype(float)
    df = df.sort_values(['Datetime'])

    return df

@st.cache_data
def _get_all_indicators_data():
    indicators = Indicators(settings='')
    # df, crossovers, ma_crossovers, bb_crossovers, hammers, suportes, resistencias, high_trend, low_trend, close_trend
    return indicators.gen_all(_df_ibovespa(), 9999, macd_rsi_BB=False, rsi_window=14)

@st.cache_data
def _series_for_seasonal():
    df = _df_ibovespa()
    series = df['Close']
    series.index = pd.to_datetime(df['Date'])
    return series

@st.cache_data
def _adfuller(series):
    return adfuller(series)

@st.cache_data
def _get_modelling_data(df = _df_ibovespa(), indicators=True,
                        regressors=['RSI',
                                    'EMA_Short','EMA_Long','12MME','9MME','26MME','MACD','MACD_Signal_line','MACD_Histogram',
                                    'Standard_Deviation','Middle_Band','Upper_Band','Lower_Band'
                                    ]):
    if (indicators):
        df['Close'] = df.Close.shift(1) # realizando shift para nao utilizar dados do dia sendo previsto nos indicadores
        df['Volume'] = df.Volume.shift(1) # realizando shift para nao utilizar dados do dia sendo previsto nos indicadores
        df, _, _ = indicators.gen_all(_df_ibovespa(), 9999, macd_rsi_BB=False, rsi_window=14)
        df = df\
                .dropna()\
                .reset_index(drop=True)[['Date', 'Adj Close'] + regressors]\
                .rename(columns={'Date':'ds', 'Adj Close':'y'})
    else:
          df = df\
                .dropna()\
                .reset_index(drop=True)[['Date', 'Adj Close']]\
                .rename(columns={'Date':'ds', 'Adj Close':'y'})
    df['ds'] = pd.to_datetime(df['ds'])
    return df

@st.cache_data
def _get_data_for_models_ts():
    df = _df_ibovespa()
    df_ts = pd.DataFrame(df[['Date', 'Close']].values, columns=['ds', 'y'])
    df_ts['unique_id'] = 'IBOV'
    df_ts['ds'] = pd.to_datetime(df_ts['ds'])
    df_ts.sort_values('ds', inplace=True)

    date_limit = '2023-01-01'

    train = df_ts.loc[df_ts['ds'] < date_limit]
    test = df_ts.loc[df_ts['ds'] >= date_limit]
    h = test.index.nunique()

    return train, test, h

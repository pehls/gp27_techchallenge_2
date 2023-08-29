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

def _adfuller(series):
    return adfuller(series)
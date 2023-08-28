import streamlit as st
import src.generate_graphs as generate_graphs
import src.get_data as get_data
from datetime import date

st.set_page_config(
    page_title="Tech Challenge #02 - Grupo 27",
    layout="wide"
)

st.write("""
    # Tech Challenge #02 - Grupo 27 
    ## Modelo Preditivo / IBOVESPA
    by. Eduardo Gomes, Igor Brito e Gabriel Pehls
""")
         
st.info("""
    Com objetivo de predizer a tendência de fechamento do IBOVESPA, mostramos nesse trabalho
    todo o processo para criação do nosso modelo.
    
    Os dados aqui utilizados foram baixados do site [investing](https://br.investing.com/indices/bovespa-historical-data) 
    e contemplam o período de 01-01-2020 até 31-07-2023.
""")
        

tab_grafico_historico, tab_seasonal, tab_adf, tab_acf = st.tabs(['Gráfico Histórico', 'Decompondo sazonalidade', 'Teste ADFuller', 'Autocorrelação - ACF/PACF'])

with tab_grafico_historico:
    df, crossover, _, _, _, _, _, _, _, _ = get_data._get_all_indicators_data()
    st.plotly_chart(
        generate_graphs._grafico_historico(df, crossover),
        use_container_width=True,
    )

with tab_seasonal:
    st.plotly_chart(
        generate_graphs._seasonal_decompose(get_data._series_for_seasonal()),
        use_container_width=True,
    )

with tab_adf:
    st.plotly_chart(
        generate_graphs._grafico_adf(get_data._df_ibovespa()),
        use_container_width=True,
    )

    st.divider()

    st.plotly_chart(
        generate_graphs._grafico_adf_diff(get_data._df_ibovespa()),
        use_container_width=True,                                 
    )

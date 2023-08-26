import streamlit as st
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
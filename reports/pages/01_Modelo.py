import streamlit as st
from src import get_data, train_model, generate_graphs
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import (mean_absolute_error, 
                             mean_squared_error, 
                             mean_absolute_percentage_error,
                             r2_score)

st.title('Modelo')


tab_modelagem_inicial, tab_resultados_iniciais, tab_conceitos, tab_hiperparametrizacao, tab_hiperparametrizacao_resultados = st.tabs(['Modelagem inicial', "Resultados Iniciais", 'Conceitos', 'Hiperparametrização - definição', "Resultados"])

df = get_data._get_modelling_data(indicators=False)

with tab_modelagem_inicial:
    st.markdown("""
        Iniciando a etapa de modelagem, optamos por experimentar o modelo Prophet, criado pela Meta/Facebook em 2017, 
        sendo um algoritmo de previsão de séries temporais que lida de forma eficiente com séries onde temos uma presença 
        forte e destacada de Sazonalidades, Feriados previamente conhecidoss e uma tendência de crescimento destacada.
        O mesmo define a série temporal tendo três componentes principais, tendência (g), sazonalidade (s) e feriados (h), 
        combinados na seguinte equação:

        `y(t)=g(t)+s(t)+h(t)+εt`, onde εt é o termo de erro, basicamente.
        
        Iniciaremos com um modelo simples, para verificar seu desempenho, e partiremos para conceitos mais elaborados, 
        chegando a um modelo hiperparametrizado, e com um desempenho superior para a aplicação.
                
    """)

    _model, X_test, pred, X_train, forecast_ = train_model._train_simple_prophet(df)
    baseline_mape = round(mean_absolute_percentage_error(X_test['y'].values, pred['yhat'].values)*100, 3)
    baseline_r2 = round(r2_score(X_train['y'].values, forecast_['yhat'].values), 4)

with tab_resultados_iniciais:
    st.plotly_chart(
        plot_plotly(_model, forecast_.dropna()),
        use_container_width=True,
    )

    st.markdown(f"""
        De acordo com o gráfico acima, podemos ver que a previsão do modelo, embora com resultados interessantes,
        ainda carece de um ajuste melhor. 
                
        No período de teste, datado entre {min(pred.ds).date()} e {max(pred.ds).date()}, temos um erro médio absoluto percentual de 
        **{baseline_mape}%**,
        e um R2 (medida de ajuste na etapa de treinamento) de 
        **{baseline_r2}**
                
    """)

with tab_conceitos:
    st.markdown(f"""
        A partir dessa modelagem inicial, e da descoberta de uma sazonalidade de 365 dias, iremos utilizar alguns 
        conceitos mais avançados para melhorar o desempenho de nossa previsão:
                
        #### Validação Cruzada com Time Series Split
    """)
    st.image("https://www.kaggleusercontent.com/kf/50808689/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..tVZyFZd7IlyTsDrLi_TosQ.jWh6fFzrp9Vakt4OkuHZd5IUKgm3H4yip9jAPinkqJnAt9uxVKVRlDyj_ovJbvQOCQ3dSqV4lk1bXKLsVRZoLnbb0iRWnWSvLc3gnXxchP1jyYe4gOdA-4w2SJRcmxglrcPS1nBdWu4w1oUB04FBDDflpJRoiDGonzAfjV_GSa2vf7RpA0nwLxuYuPllRfA0ka-VggM2jVKsrtuXnCGqKr1nKyVBwz_aKzsauB923-dFDNsHVrqwkX1xOFGpjM0B9DNHYu1UNXrSuZYxTzSCAVG6vMvwpYmAokZDphjs4VqD-rI1ePypXTuDBQ9MW3Ef5iWH_b7jbWlqYKCrVhggDK0-dmeYIs4D_FMJk2HtXznjzfvv_bqkE8bn6CEpM2dvv2j9aVQ8wMicH1IoNhExf1kMnP28CJ_SQSj0nUQQKnTRDRTOaYiEdzVkzAl_CQgLgzxHhDUKdisTUpuidm-rnJTk3xK_RT5tapbGWJ025gInD_kaKpR1ubRzNa2YYXcMidmnWcK8nJAtRsyHw0EO3StlY0u91OeNZQohRde5FLpGaqacUQVgHQHZWXhDJT1HE91AlI3G9n2DEDvB7NVgm-6Wkqnp4xSWkndUsmonvHfbUIBsPlFQH9hIBogllfavqHYz5hBAdraKIoAo6Yij-JiFLn7d6Hz7CfVTWQTW5WI.yqL8_aruoGUh1reTxb6Fgw/__results___files/__results___7_1.png",
            caption="Time Series Split",
            width=600,
    )

    st.markdown(f"""     
        #### Adição de feriados na modelagem do Prophet
    """)
    st.markdown(f"""            
        #### Hiperparametrização Bayesiana
                
    """)

with tab_hiperparametrizacao:
    st.markdown("""
        Após tais definições conceituais, definimos nossa função objetivo como:
    """)
    st.code("""            
        def objective(params):
            metrics, cv_mape = _run_cv_prophet(
                            df_model=X_train.dropna(),
                            params=params,
                            n_splits=5, test_size=test_size
                        )
            return {
                          'loss':cv_mape
                        , 'status': STATUS_OK
                    }
    """, language='python')  
    st.markdown("""          
        Ou seja, vamos minimizar o mape, definido como uma lista dos mapes dos 5 splits mencionados na função, 
        mantendo um test_size (ou, tamanho da base de teste) como 30 pontos (aproximadamente um mês);

        O espaço de busca do algoritmo vai ser definido como:
    """)
    st.code("""
        space = {
            'yearly_seasonality':365
            , 'daily_seasonality':hp.choice('daily_seasonality', [True, False])
            , 'weekly_seasonality':hp.choice('weekly_seasonality', [True, False])
            , 'seasonality_mode' : hp.choice('seasonality_mode', ['multiplicative','additive'])
            , "seasonality_prior_scale": hp.uniform("seasonality_prior_scale", 7, 10)
            , "changepoint_prior_scale": hp.uniform("changepoint_prior_scale", 0.4, 0.5)
            , "changepoint_range": hp.uniform("changepoint_range", 0.4, 0.5)
            , 'holidays_prior_scale' : hp.uniform('holidays_prior_scale', 7, 10)
            , "regressors":''
        }
    """, language='python')    
    st.markdown("""        
        Aqui, fixamos a sazonalidade anual como 365, mantendo a diária e semanal como o padrão do algoritmo,
        bem como deixamos o algoritmo de hiperparametrização definir qual a melhor configuração para os demais hiperparâmetros do modelo.
                
        Com tais configurações, chegamos ao seguinte melhor resultado:
    """)   

    st.code("best_params = "+str(train_model._get_best_params()).replace(",",",\n\t\t"), language='python')

    st.markdown("""        
        Para melhor visualizar o resultado da hiperparametrização, podemos verificar no seguinte gráfico, as áreas onde temos espaços mais
        "pretos", onde estão concentrados os resultados com menor erro percentual; Nota-se que existem vários "vales" de bons resultados, onde
        nossa hiperparametrização poderia ter retornado bons parâmetros;
        Para modificar o parâmetro sendo analisado, basta selecionar abaixo:
    """)   
    trials_df = get_data._trials()

    col1, col2 = st.columns(2)

    with col1:
        hyperparam_1 = st.selectbox(
            'Hiperparâmetro 1',
            list(set(trials_df.columns) - set(['loss']))
        )
    with col2:    
        hyperparam_2 = st.selectbox(
            'Hiperparâmetro 2',
            list(set(trials_df.columns) - set(['loss']))
        )

    st.plotly_chart(
        generate_graphs._plot_trials(trials_df, hyperparam_1, hyperparam_2)
    )

    _model, X_test, pred, X_train, forecast_ = train_model._train_cv_prophet(df)
    second_mape = round(mean_absolute_percentage_error(X_test['y'].values, pred['yhat'].values)*100, 3)
    second_r2 = round(r2_score(X_train['y'].values, 
                          forecast_.loc[forecast_.ds.isin(X_train.ds.to_list())]['yhat'].values)
                , 4)
    sec_melhoria_mape = round(second_mape - baseline_mape, 2)

with tab_hiperparametrizacao_resultados:
    st.plotly_chart(
        plot_plotly(_model, forecast_.dropna()),
        use_container_width=True,
    )

    st.markdown(f"""
        De acordo com o gráfico acima, podemos ver que a previsão do modelo, embora com resultados interessantes,
        ainda carece de um ajuste melhor. 
                
        No período de teste, datado entre {min(pred.ds).date()} e {max(pred.ds).date()}, temos um erro médio absoluto percentual de 
        **{second_mape}%**,
        e um R2 (medida de ajuste na etapa de treinamento) de 
        **{second_r2}**.

        Tais resultados, mostram uma melhoria de {sec_melhoria_mape}% em mape, em porcentagem absoluta!
                
    """)
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression


st.header("Previsão de Vendas")


# Dados: [Investimento em Marketing] -> Faturamento
dados_vendas = pd.DataFrame({
    'investimento': [0,100, 200, 300, 400, 500, 600],
    'faturamento': [0,1200, 2400, 3600, 4800, 6000, 7200]
})


modelo_faturamento =  LinearRegression()
modelo_faturamento.fit(dados_vendas[['investimento']], dados_vendas[['faturamento']])


valor =st.number_input('Quanto vai investir: ')


faturamento  =  modelo_faturamento.predict([[valor]])


st.subheader(f'O Seu faturamento seria:{min(faturamento[0]):.2f}')

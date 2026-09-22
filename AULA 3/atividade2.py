import streamlit as st    # interface grafica 
import pandas as pd       # tratamento de dados
from sklearn.linear_model import LinearRegression    # classificação
import numpy as np 


dados  =  pd.read_csv('vendas.csv')


st.header('PREVISÃO DE VENDAS POR MêS')



d =  pd.DataFrame({
     'mes' :dados['mes'],
     'vendas':dados['vendas']      
        
    }  
)




X = d[['mes']]


st.write('MESES:', X)


y = d['vendas']


st.write(y)


modelo =  LinearRegression().fit(X,y)


mes =  st.number_input('Digite um Mes para prever a vebda')


p  = modelo.predict([[mes]])


previsao = f'{p[0]:.2f}'


print(f'mes 9 - R$  {p[0]:.2f}')


st.subheader(previsao)
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Base de dados fornecida
gamer = pd.DataFrame({
    'horas_jogo':[1,2,4,6,8,10] ,
    'cansaco': [1, 2, 3, 5, 8, 10]
})

# 2. Preparação dos dados para o algoritmo
# O scikit-learn exige que a variável explicativa (X) seja uma matriz bidimensional
X = gamer[['horas_jogo']]  # Variável independente (Causa)
y = gamer['cansaco']       # Variável dependente (Efeito)

# 3. Inicialização e Treinamento do Modelo de IA
modelo = LinearRegression()
modelo.fit(X, y)

# 4. Extração dos Parâmetros Estatísticos
coeficiente_angular = modelo.coef_[0]
intercepto = modelo.intercept_
precisao_r2 = modelo.score(X, y)

# 5. Função de Previsão Automatizada
def prever_cansaco(horas):
    """
    Calcula o cansaço estimado com base nas horas de jogo informadas.
    """
    # Garante que horas negativas não sejam processadas
    if horas < 0:
        return 0.0
    
    # Executa a previsão usando o modelo treinado
    horas_array = np.array([[horas]])
    previsao = modelo.predict(horas_array)[0]
    
    # Limita o resultado entre 0 e 10 (limites lógicos da escala de cansaço)
    return round(max(0.0, min(10.0, previsao)), 2)

# --- Relatório Analítico e de Validação no Terminal ---
print("-" * 50)
print("📊 SISTEMA DE PREVISÃO DE CANSAÇO DO JOGADOR")
print("-" * 50)
print(f"Equação Matemática: Cansaço = ({coeficiente_angular:.4f} * Horas) + ({intercepto:.4f})")
print(f"Precisão do Modelo (R²): {precisao_r2 * 100:.2f}%")
print("-" * 50)

# Exemplo Prático de Aplicação (Previsão para 5 horas e 12 horas de jogo)
teste_5h = prever_cansaco(5)
teste_12h = prever_cansaco(12)

print(f"💡 Simulação - 5 horas de jogo contínuo: Nível de Cansaço {teste_5h}/10")
print(f"💡 Simulação - 12 horas de jogo contínuo: Nível de Cansaço {teste_12h}/10 (Limite crítico atingido)")
print("-" * 50)

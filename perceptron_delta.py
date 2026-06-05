import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from dataset import Iris_Data

# ==========================================
# 1. PREPARAÇÃO DOS DADOS
# ==========================================
# Lendo o arquivo CSV (usando pandas devido à formatação de vírgulas e aspas)
df = pd.read_csv('TEIATRON\\Iris_data.csv')

# Substituindo vírgulas por pontos e convertendo para float
for col in df.columns[:-1]:
    if df[col].dtype == object:
        df[col] = df[col].str.replace(',', '.').astype(float)

X = df.iloc[:, [0, 2]].values
# Ajustando alvos para 1 (Setosa) e -1 (Outras) - melhor para a Regra Delta
y = np.where(df['Species'] == 'setosa', 1, -1) 

# Divisão Treino / Teste
random.seed(42)
indices = list(range(len(X)))
random.shuffle(indices)
split = int(0.8 * len(X))
X_treino, y_treino = X[indices[:split]], y[indices[:split]]

class AdalineRegraDelta:
    def __init__(self, taxa_aprendizado=0.01, epocas=50):
        self.pesos = [0.0, 0.0]
        self.bias = 0.0
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas
        self.custo_por_epoca = [] # Guardará a curva suave

    def calcular_soma_linear(self, x):
        return (x[0] * self.pesos[0]) + (x[1] * self.pesos[1]) + self.bias

    def treinar(self, X_treino, y_treino):
        for _ in range(self.epocas):
            erro_quadratico_total = 0
            
            for x, alvo in zip(X_treino, y_treino):
                # Usamos o valor contínuo em vez da função degrau (0 ou 1)
                saida_linear = self.calcular_soma_linear(x)
                
                # O erro agora é a distância contínua do alvo
                erro = alvo - saida_linear
                
                # Regra Delta: Atualização baseada na saída contínua
                self.pesos[0] += self.taxa_aprendizado * erro * x[0]
                self.pesos[1] += self.taxa_aprendizado * erro * x[1]
                self.bias += self.taxa_aprendizado * erro
                
                # Acumula o quadrado do erro (Erro Quadrático)
                erro_quadratico_total += (erro ** 2)
                
            # O custo da época é a média dos erros quadráticos (MSE)
            custo_medio = erro_quadratico_total / 2.0
            self.custo_por_epoca.append(custo_medio)

# Precisamos de uma taxa de aprendizado menor para o Gradiente Descendente não explodir
modelo_suave = AdalineRegraDelta(taxa_aprendizado=0.005, epocas=15)
modelo_suave.treinar(X_treino, y_treino)

# --- PLOTANDO A CURVA ---
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(modelo_suave.custo_por_epoca) + 1), modelo_suave.custo_por_epoca, 
         marker='o', color='#2ecc71', lw=2, markersize=4)

plt.title('Convergência com a Regra Delta (Adaline)', fontsize=14)
plt.xlabel('Épocas', fontsize=12)
plt.ylabel('Erro Quadrático Total (Custo)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
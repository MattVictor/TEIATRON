import pandas as pd
import pingouin as pg
import urllib.request
import io

# 1. Carregando a base Iris diretamente para um DataFrame do Pandas
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
response = urllib.request.urlopen(url)
csv_data = response.read().decode('utf-8')

# Definindo colunas
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv(io.StringIO(csv_data), header=None, names=col_names)
df = df.dropna() # Removendo a última linha vazia, se houver

# 2. Separando por classes
setosa = df[df['species'] == 'Iris-setosa'].drop(columns=['species'])
versicolor = df[df['species'] == 'Iris-versicolor'].drop(columns=['species'])
virginica = df[df['species'] == 'Iris-virginica'].drop(columns=['species'])

# 3. Executando o Teste de Henze-Zirkler
print("--- Teste de Normalidade Multivariada (Henze-Zirkler) ---")

# O teste retorna um booleano (Normal = True/False) e o p-valor
hz_setosa = pg.multivariate_normality(setosa, alpha=0.05)
print(f"Setosa: Normal? {hz_setosa.normal} | p-valor: {hz_setosa.pval:.4f}")

hz_versicolor = pg.multivariate_normality(versicolor, alpha=0.05)
print(f"Versicolor: Normal? {hz_versicolor.normal} | p-valor: {hz_versicolor.pval:.4f}")

hz_virginica = pg.multivariate_normality(virginica, alpha=0.05)
print(f"Virginica: Normal? {hz_virginica.normal} | p-valor: {hz_virginica.pval:.4f}")

import csv
import urllib.request
import numpy as np

# =====================================================================
# 1. CARREGAMENTO E DIVISÃO DA BASE DE DADOS (70% Treino / 30% Teste)
# =====================================================================

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
response = urllib.request.urlopen(url)
lines = [l.decode('utf-8') for l in response.readlines()]
data = list(csv.reader(lines))

X = []
y = []
# Mapeando nomes para inteiros para facilitar a matemática
classes_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
inverse_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

for row in data:
    if len(row) == 5:
        X.append([float(x) for x in row[:-1]])
        y.append(classes_map[row[-1]])

X = np.array(X)
y = np.array(y)

# Embaralhando os dados para garantir uma divisão justa
np.random.seed(42) # Semente fixa para reprodutibilidade
indices = np.random.permutation(len(X))
X_shuffled = X[indices]
y_shuffled = y[indices]

# Divisão 70% Treino / 30% Teste
split_index = int(0.7 * len(X))
X_train, X_test = X_shuffled[:split_index], X_shuffled[split_index:]
y_train, y_test = y_shuffled[:split_index], y_shuffled[split_index:]


# =====================================================================
# 2. CLASSIFICADOR DE BAYES ÓTIMO (GAUSSIANO MULTIVARIADO) - REGRA MAP
# =====================================================================

class OptimalBayesMAP:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.parameters = {}
        
        for c in self.classes:
            X_c = X[y == c]
            mean = np.mean(X_c, axis=0)
            cov = np.cov(X_c, rowvar=False) 
            self.parameters[c] = {'mean': mean, 'cov': cov}
            
    def _predict_single(self, x):
        posteriors = []
        for c in self.classes:
            mean = self.parameters[c]['mean']
            cov = self.parameters[c]['cov']
            
            # Função discriminante (Prior suprimido pois P(C1) = P(C2) = P(C3))
            inv_cov = np.linalg.inv(cov)
            det_cov = np.linalg.det(cov)
            diff = x - mean
            
            # log(P(x|C)) -> Omissão do log(P(C))
            term1 = -0.5 * np.log(det_cov)
            term2 = -0.5 * np.dot(np.dot(diff.T, inv_cov), diff)
            
            posterior = term1 + term2
            posteriors.append(posterior)
            
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])


# =====================================================================
# 3. CLASSIFICADOR NAIVE BAYES - REGRA MAP
# =====================================================================

class NaiveBayesMAP:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.parameters = {}
        
        for c in self.classes:
            X_c = X[y == c]
            mean = np.mean(X_c, axis=0)
            var = np.var(X_c, axis=0)
            self.parameters[c] = {'mean': mean, 'var': var}
            
    def _predict_single(self, x):
        posteriors = []
        for c in self.classes:
            mean = self.parameters[c]['mean']
            var = self.parameters[c]['var']
            
            # Soma das log-probabilidades normais individuais
            # Prior suprimido pois P(C1) = P(C2) = P(C3)
            term1 = -0.5 * np.sum(np.log(2 * np.pi * var))
            term2 = -0.5 * np.sum(((x - mean) ** 2) / var)
            
            posterior = term1 + term2
            posteriors.append(posterior)
            
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])


# =====================================================================
# 4. CÁLCULO DAS SUPERFÍCIES DE DECISÃO (BAYES ÓTIMO)
# =====================================================================
# A superfície de decisão d_i(x) - d_j(x) = 0 define a fronteira.
# A equação geral quadrática é: x^T * W * x + w^T * x + w0 = 0
# Onde:
# W = -0.5 * (InvCov_i - InvCov_j)
# w = (InvCov_i * Mean_i) - (InvCov_j * Mean_j)
# w0 = -0.5 * (Mean_i^T * InvCov_i * Mean_i - Mean_j^T * InvCov_j * Mean_j) 
#      - 0.5 * ln(|Cov_i| / |Cov_j|)

def calcular_superficie(modelo, classe_i, classe_j):
    m_i = modelo.parameters[classe_i]['mean']
    cov_i = modelo.parameters[classe_i]['cov']
    inv_cov_i = np.linalg.inv(cov_i)
    
    m_j = modelo.parameters[classe_j]['mean']
    cov_j = modelo.parameters[classe_j]['cov']
    inv_cov_j = np.linalg.inv(cov_j)
    
    # Calculando os termos da equação quadrática
    W = -0.5 * (inv_cov_i - inv_cov_j)
    w = np.dot(inv_cov_i, m_i) - np.dot(inv_cov_j, m_j)
    
    term_w0_1 = -0.5 * (np.dot(np.dot(m_i.T, inv_cov_i), m_i) - np.dot(np.dot(m_j.T, inv_cov_j), m_j))
    term_w0_2 = -0.5 * np.log(np.linalg.det(cov_i) / np.linalg.det(cov_j))
    w0 = term_w0_1 + term_w0_2
    
    print(f"\nFronteira {inverse_map[classe_i]} x {inverse_map[classe_j]}:")
    print(f"Matriz W (Quadrática):\n{np.round(W, 3)}")
    print(f"Vetor w (Linear): {np.round(w, 3)}")
    print(f"Constante w0: {np.round(w0, 3)}")


# =====================================================================
# 5. EXECUÇÃO E RESULTADOS
# =====================================================================

print("--- TREINANDO MODELOS (70% dos dados) ---")
model_optimal = OptimalBayesMAP()
model_optimal.fit(X_train, y_train)

model_naive = NaiveBayesMAP()
model_naive.fit(X_train, y_train)

print("\n--- AVALIANDO NO CONJUNTO DE TESTE (30% dos dados) ---")
preds_optimal = model_optimal.predict(X_test)
acc_optimal = np.mean(preds_optimal == y_test) * 100

preds_naive = model_naive.predict(X_test)
acc_naive = np.mean(preds_naive == y_test) * 100

print(f"Acurácia (Bayes Ótimo MAP): {acc_optimal:.2f}%")
print(f"Acurácia (Naive Bayes MAP): {acc_naive:.2f}%")

print("\n--- SUPERFÍCIES DE DECISÃO (d_i(x) - d_j(x) = 0) ---")
calcular_superficie(model_optimal, 0, 2) # i) Setosa x Virginica
calcular_superficie(model_optimal, 1, 2) # ii) Versicolor x Virginica
calcular_superficie(model_optimal, 0, 1) # iii) Setosa x Versicolor
import math

# --- Funções Matemáticas Auxiliares (Puro Python) ---
def calcular_centroide(X):
    n_atributos = len(X[0])
    n_amostras = len(X)
    return [sum(x[i] for x in X) / n_amostras for i in range(n_atributos)]

def distancia_euclidiana(vetor1, vetor2):
    return math.sqrt(sum((v1 - v2) ** 2 for v1, v2 in zip(vetor1, vetor2)))

def produto_escalar(vetor1, vetor2):
    return sum(v1 * v2 for v1, v2 in zip(vetor1, vetor2))


# --- I) Classificador de Distância Mínima (3 Classes) ---
class ClassificadorDistanciaMinima:
    def __init__(self):
        self.centroides = {}

    def treinar(self, X, y):
        # Agrupa os dados por classe
        dados_por_classe = {}
        for atributos, classe in zip(X, y):
            if classe not in dados_por_classe:
                dados_por_classe[classe] = []
            dados_por_classe[classe].append(atributos)
        
        # Calcula o centroide (média) de cada classe (Aula PR_3)
        for classe, dados in dados_por_classe.items():
            self.centroides[classe] = calcular_centroide(dados)

    def prever(self, x):
        menor_distancia = float('inf')
        classe_prevista = None
        
        for classe, centroide in self.centroides.items():
            dist = distancia_euclidiana(x, centroide)
            if dist < menor_distancia:
                menor_distancia = dist
                classe_prevista = classe
                
        return classe_prevista


# --- II) Função de Decisão - Classificador Máximo (3 Classes) ---
class ClassificadorMaximo:
    def __init__(self):
        self.centroides = {}

    def treinar(self, X, y):
        # Treinamento é idêntico ao da Distância Mínima (encontrar mi)
        dados_por_classe = {}
        for atributos, classe in zip(X, y):
            if classe not in dados_por_classe:
                dados_por_classe[classe] = []
            dados_por_classe[classe].append(atributos)
        
        for classe, dados in dados_por_classe.items():
            self.centroides[classe] = calcular_centroide(dados)

    def prever(self, x):
        # Aplica a Função Discriminante Linear: di(x) = x^T * mi - 0.5 * mi^T * mi
        maior_valor = float('-inf')
        classe_prevista = None
        
        for classe, mi in self.centroides.items():
            termo1 = produto_escalar(x, mi)
            termo2 = 0.5 * produto_escalar(mi, mi)
            di_x = termo1 - termo2
            
            # Escolhe a classe que maximiza a função de decisão
            if di_x > maior_valor:
                maior_valor = di_x
                classe_prevista = classe
                
        return classe_prevista


# --- III) Superfície de Decisão (Para 2 Classes) ---
class SuperficieDecisaoPares:
    def __init__(self, classe_A, classe_B):
        self.classe_A = classe_A
        self.classe_B = classe_B
        self.vetor_peso = []
        self.bias = 0.0

    def treinar(self, X, y):
        # Filtra apenas os dados das duas classes de interesse
        X_A = [x for x, rotulo in zip(X, y) if rotulo == self.classe_A]
        X_B = [x for x, rotulo in zip(X, y) if rotulo == self.classe_B]
        
        mi_A = calcular_centroide(X_A)
        mi_B = calcular_centroide(X_B)
        
        # Conforme a Aula PR_3: A superfície é o bissetor perpendicular
        # w = mi_A - mi_B
        self.vetor_peso = [a - b for a, b in zip(mi_A, mi_B)]
        
        # bias = -0.5 * (mi_A^T * mi_A - mi_B^T * mi_B)
        mi_A_quadrado = produto_escalar(mi_A, mi_A)
        mi_B_quadrado = produto_escalar(mi_B, mi_B)
        self.bias = -0.5 * (mi_A_quadrado - mi_B_quadrado)

    def prever(self, x):
        # w^T * x + b > 0
        resultado = produto_escalar(self.vetor_peso, x) + self.bias
        return self.classe_A if resultado > 0 else self.classe_B
        
    def obter_equacao_str(self):
        termos = [f"{w:.2f}*x{i+1}" for i, w in enumerate(self.vetor_peso)]
        soma_w = " + ".join(termos)
        return f"{soma_w} + ({self.bias:.2f}) = 0"
    
# --- IV) Perceptron Clássico (Classificador Binário) ---
class PerceptronClassico:
    def __init__(self, n_atributos=4, taxa_aprendizado=0.03, max_epocas=100):
        # Item B: Peso inicial w(1) = (0,0,0,0,0) -> 4 pesos + 1 bias
        self.pesos = [0.0] * n_atributos
        self.bias = 0.0
        self.taxa_aprendizado = taxa_aprendizado
        self.max_epocas = max_epocas
        
        # Guardamos os nomes para a função de decisão
        self.classe_positiva = None
        self.classe_negativa = None

    def prever(self, x):
        # Retorna 1 se for a classe positiva, 0 se for a negativa
        soma = produto_escalar(self.pesos, x) + self.bias
        return 1 if soma >= 0 else 0

    def treinar(self, X, y, classe_pos, classe_neg):
        self.classe_positiva = classe_pos
        self.classe_negativa = classe_neg
        
        epoca_parada = self.max_epocas
        
        # Item E: Treina no máximo 100 épocas
        for epoca in range(self.max_epocas):
            erros_na_epoca = 0
            
            for xi, yi in zip(X, y):
                previsao = self.prever(xi)
                erro = yi - previsao
                
                if erro != 0:
                    erros_na_epoca += 1
                    # Atualiza os pesos: w = w + taxa * erro * xi
                    for j in range(len(self.pesos)):
                        self.pesos[j] += self.taxa_aprendizado * erro * xi[j]
                    # Atualiza o bias
                    self.bias += self.taxa_aprendizado * erro
            
            # Se não houve erros, os dados são linearmente separáveis!
            if erros_na_epoca == 0:
                epoca_parada = epoca + 1
                break
                
        return epoca_parada # Retornamos a época em que parou para o relatório
        
    def obter_equacao_str(self):
        termos = [f"{w:.2f}*x{i+1}" for i, w in enumerate(self.pesos)]
        soma_w = " + ".join(termos)
        return f"{soma_w} + ({self.bias:.2f}) = 0"
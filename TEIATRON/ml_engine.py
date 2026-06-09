# ml_engine.py
import math

class MinDistanceClassifier:
    def __init__(self):
        self.centroids = {}
        self.classes_trained = []

    def train(self, X_train, y_train):
        """
        Calcula e armazena o centróide de cada classe.
        X_train: lista de listas (cada sublista é um ponto ex: [5.1, 3.5, 1.4, 0.2])
        y_train: lista de strings com as classes respectivas.
        """
        grupos = {}

        # Agrupar pontos por classe
        for ponto, classe in zip(X_train, y_train):
            if classe not in grupos:
                grupos[classe] = []
            grupos[classe].append(ponto)

        self.centroids = {}

        # Calcular média geométrica (Centróide)
        for classe, pontos in grupos.items():
            n = len(pontos)
            dimensao = len(pontos[0])
            media = [0.0] * dimensao

            for ponto in pontos:
                for i in range(dimensao):
                    media[i] += ponto[i]

            for i in range(dimensao):
                media[i] /= n

            self.centroids[classe] = media
            
        self.classes_trained = list(self.centroids.keys())
        return self.centroids

    def predict(self, novo_ponto):
        """
        Retorna a classe do centróide mais próximo e as distâncias calculadas.
        """
        if not self.centroids:
            raise Exception("O modelo ainda não foi treinado.")
            
        melhor_classe = None
        menor_distancia = float('inf')
        distancias_calculadas = {} # Guarda os valores para exibir no pop-up

        for classe, centroide in self.centroids.items():
            soma = 0.0
            for i in range(len(novo_ponto)):
                soma += (novo_ponto[i] - centroide[i]) ** 2
            d = soma ** 0.5
            
            distancias_calculadas[classe] = d

            if d < menor_distancia:
                menor_distancia = d
                melhor_classe = classe

        return melhor_classe, distancias_calculadas
    
class MaxDistanceClassifier:
    def __init__(self):
        self.X_train = []
        self.y_train = []
        self.classes_trained = []

    def train(self, X_train, y_train):
        """
        Armazena os dados para cálculo posterior da distância máxima.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.classes_trained = list(sorted(set(y_train)))
        return self.classes_trained

    def predict(self, novo_ponto):
        """
        Classifica o novo ponto e retorna os valores das distâncias máximas de cada classe.
        """
        if not self.X_train:
            raise Exception("O modelo ainda não foi treinado.")
            
        distancia_maxima = {classe: 0.0 for classe in self.classes_trained}
        
        for ponto, classe in zip(self.X_train, self.y_train):
            soma = 0.0
            for i in range(len(novo_ponto)):
                soma += (novo_ponto[i] - ponto[i]) ** 2
            d = soma ** 0.5
            
            if d > distancia_maxima[classe]:
                distancia_maxima[classe] = d
                
        melhor_classe = min(distancia_maxima, key=distancia_maxima.get)
        return melhor_classe, distancia_maxima
    
class PerceptronClassifier:
    def __init__(self):
        self.pesos = []
        self.class_map = {}
        self.reverse_map = {}
        self.historico_erros = []

    def train(self, X_train, y_train, classe_alvo, epocas, learning_rate, pesos_iniciais, regra_delta=False):
        # 1. Mapeamento de Classes (Um contra todos ou Binário Clássico)
        # O alvo vira 1, o resto vira 0.
        self.class_map = {classe_alvo: 1}
        
        # Descobre qual é a segunda classe (se for binário) ou chama de "Resto" (se for OvA)
        outras_classes = [y for y in set(y_train) if y != classe_alvo]
        nome_classe_zero = outras_classes[0] if len(outras_classes) == 1 else "Resto"
        self.reverse_map = {1: classe_alvo, 0: nome_classe_zero}
        
        y_mapped = [1 if y == classe_alvo else 0 for y in y_train]
        
        self.pesos = list(pesos_iniciais)
        self.historico_erros = []
        
        # 2. Loop de Treinamento
        for epoca in range(epocas):
            erros_de_classificacao = 0 
            
            for vetor, classe_real in zip(X_train, y_mapped):
                # Soma ponderada: Inicia com o Bias (pesos[0])
                ativacao_continua = self.pesos[0]
                for i in range(len(vetor)):
                    ativacao_continua += self.pesos[i + 1] * vetor[i]
                    
                # Predição Discreta (Degrau)
                classe_predita = 1 if ativacao_continua >= 0.0 else 0
                
                if classe_real != classe_predita:
                    erros_de_classificacao += 1
                
                # Escolha da regra de ajuste
                if regra_delta:
                    erro = classe_real - ativacao_continua
                else:
                    erro = classe_real - classe_predita
                
                # Atualização
                if erro != 0:
                    self.pesos[0] += learning_rate * erro
                    for i in range(len(vetor)):
                        self.pesos[i + 1] += learning_rate * erro * vetor[i]
                        
            # Salva histórico para o Gráfico de Épocas
            self.historico_erros.append(erros_de_classificacao)
            
            # Condição de parada antecipada
            if erros_de_classificacao == 0 and not regra_delta:
                break
                
        return self.pesos

    def predict(self, novo_ponto):
        if not self.pesos:
            raise Exception("O modelo ainda não foi treinado.")
            
        ativacao = self.pesos[0]
        for i in range(len(novo_ponto)):
            ativacao += self.pesos[i + 1] * novo_ponto[i]
            
        classe_predita = 1 if ativacao >= 0.0 else 0
        nome_classe = self.reverse_map[classe_predita]
        
        return nome_classe, {"Ativação (Soma Ponderada)": ativacao}
    
class ClassificadorMetricas:
    def __init__(self, matriz):
        """Recebe uma matriz de confusão genérica NxN (lista de listas)."""
        self.matriz = matriz
        self.n = len(matriz)
        self.total = sum(sum(linha) for linha in matriz)

    def acerto_geral(self):
        if self.total == 0: return 0
        acertos = sum(self.matriz[i][i] for i in range(self.n))
        return acertos / self.total

    def acuracia_produtor(self):
        # Recall Médio (Macro-Averaging) para lidar com múltiplas classes
        somas = 0
        for i in range(self.n):
            linha_total = sum(self.matriz[i])
            if linha_total > 0:
                somas += self.matriz[i][i] / linha_total
        return somas / self.n if self.n > 0 else 0

    def acuracia_usuario(self):
        # Precision Médio (Macro-Averaging)
        somas = 0
        for i in range(self.n):
            coluna_total = sum(self.matriz[j][i] for j in range(self.n))
            if coluna_total > 0:
                somas += self.matriz[i][i] / coluna_total
        return somas / self.n if self.n > 0 else 0

    def coeficiente_kappa(self):
        if self.total == 0: return 0
        po = self.acerto_geral()
        pe_sum = 0
        for i in range(self.n):
            linha_total = sum(self.matriz[i])
            coluna_total = sum(self.matriz[j][i] for j in range(self.n))
            pe_sum += linha_total * coluna_total
        pe = pe_sum / (self.total ** 2)
        return (po - pe) / (1 - pe) if (1 - pe) != 0 else 0

    def coeficiente_tau(self):
        if self.n == 0: return 0
        c = self.n
        po = self.acerto_geral()
        return (po - (1/c)) / (1 - (1/c)) if c > 1 else 0

    def coeficiente_matthews(self):
        # MCC é originalmente binário. Mantemos o cálculo exato para 2x2.
        if self.n == 2:
            tn, fp = self.matriz[0][0], self.matriz[0][1]
            fn, tp = self.matriz[1][0], self.matriz[1][1]
            num = (tp * tn) - (fp * fn)
            den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
            return num / (den ** 0.5) if den != 0 else 0
        return 0 # Para multiclasse puro, a versão simplificada é 0.

    def fb_score(self, b):
        prec = self.acuracia_usuario()
        rec = self.acuracia_produtor()
        if (prec + rec) == 0:
            return 0
        return ((1 + b**2) * prec * rec) / ((b**2 * prec) + rec)
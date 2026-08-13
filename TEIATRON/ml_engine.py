# ml_engine.py
import math
import numpy as np

class BaseClassifier:
    def train(self, X_train, y_train, **kwargs):
        raise NotImplementedError
        
    def predict(self, novo_ponto):
        raise NotImplementedError
        
    def get_logs(self):
        return []
        
    def get_plot_data(self, **kwargs):
        """Retorna dicionário de dados geométricos agnósticos para a UI plotar"""
        return {}
        
    @classmethod
    def get_hyperparameters(cls):
        """Retorna os hiperparâmetros necessários para gerar a UI dinamicamente."""
        return []

class MinDistanceClassifier(BaseClassifier):
    @classmethod
    def get_hyperparameters(cls):
        return [
            {"name": "Multiclasse", "type": "bool", "default": True},
            {"name": "Classe 1", "type": "class_selector", "default": "Iris-setosa"},
            {"name": "Classe 2", "type": "class_selector", "default": "Iris-versicolor"}
        ]

    def __init__(self):
        self.centroids = {}
        self.classes_trained = []

    def train(self, X_train, y_train, **kwargs):
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

    def get_plot_data(self, **kwargs):
        x_key = kwargs.get('x_key')
        y_key = kwargs.get('y_key')
        keys = kwargs.get('keys')
        if not x_key or not y_key or not keys: return {}
        
        idx_x = keys.index(x_key)
        idx_y = keys.index(y_key)
        
        points = []
        for c_name, coords in self.centroids.items():
            cx, cy = coords[idx_x], coords[idx_y]
            points.append({"x": cx, "y": cy, "name": f"Centróide {c_name}", "symbol": "x", "size": 15, "color": "w"})
            
        lines = []
        if len(self.classes_trained) == 2:
            c1_coords = self.centroids[self.classes_trained[0]]
            c2_coords = self.centroids[self.classes_trained[1]]
            nx = c2_coords[idx_x] - c1_coords[idx_x]
            ny = c2_coords[idx_y] - c1_coords[idx_y]
            mx = (c1_coords[idx_x] + c2_coords[idx_x]) / 2.0
            my = (c1_coords[idx_y] + c2_coords[idx_y]) / 2.0
            
            if ny != 0:
                angulo_deg = math.degrees(math.atan2(-nx, ny))
                a = -nx / ny
                b = my - (a * mx)
                equacao = f"g(x) = {a:.2f}x {'+' if b >= 0 else '-'} {abs(b):.2f}"
            else:
                angulo_deg = 90
                equacao = f"g(x) -> x = {mx:.2f}"
            lines.append({"angle": angulo_deg, "pos": (mx, my), "name": equacao})
            
        return {"points": points, "lines": lines}
    
class MaxDistanceClassifier(BaseClassifier):
    @classmethod
    def get_hyperparameters(cls):
        return [
            {"name": "Multiclasse", "type": "bool", "default": True},
            {"name": "Classe 1", "type": "class_selector", "default": "Iris-setosa"},
            {"name": "Classe 2", "type": "class_selector", "default": "Iris-versicolor"}
        ]

    def __init__(self):
        self.X_train = []
        self.y_train = []
        self.classes_trained = []

    def train(self, X_train, y_train, **kwargs):
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
        
    def get_plot_data(self, **kwargs):
        return {"empty_legends": ["Critério: Minimização da Distância Máxima"]}
    
class PerceptronClassifier(BaseClassifier):
    @classmethod
    def get_hyperparameters(cls):
        return [
            {"name": "Regra Delta", "type": "bool", "default": False},
            {"name": "Estratégia", "type": "options", "choices": ["Clássico", "Um contra todos"], "default": "Clássico"},
            {"name": "Classe 1", "type": "class_selector", "default": "Iris-setosa"},
            {"name": "Classe 2", "type": "class_selector", "default": "Iris-versicolor"},
            {"name": "Classe Alvo", "type": "class_selector", "default": "Iris-setosa"},
            {"name": "Épocas", "type": "int", "min": 1, "max": 100000, "default": 100},
            {"name": "Learning Rate", "type": "float", "min": 0.0001, "max": 10.0, "default": 0.01},
            {"name": "Bias Inicial", "type": "float", "min": -100.0, "max": 100.0, "default": 0.0},
            {"name": "Pesos Iniciais", "type": "string", "default": "0.0, 0.0, 0.0, 0.0"}
        ]

    def __init__(self):
        self.pesos = []
        self.class_map = {}
        self.reverse_map = {}
        self.historico_erros = []

    def train(self, X_train, y_train, **kwargs):
        classe_alvo = kwargs.get('classe_alvo')
        epocas = kwargs.get('epocas', 100)
        learning_rate = kwargs.get('learning_rate', 0.01)
        pesos_iniciais = kwargs.get('pesos_iniciais', [0,0,0,0,0])
        regra_delta = kwargs.get('regra_delta', False)
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
        
    def get_plot_data(self, **kwargs):
        x_key = kwargs.get('x_key')
        y_key = kwargs.get('y_key')
        keys = kwargs.get('keys')
        dataset = kwargs.get('dataset')
        if not x_key or not y_key or not keys or not dataset: return {}
        
        idx_x = keys.index(x_key)
        idx_y = keys.index(y_key)
        
        b = self.pesos[0]
        w_x = self.pesos[idx_x + 1]
        w_y = self.pesos[idx_y + 1]
        
        effective_bias = b
        for i, k in enumerate(keys):
            if k != x_key and k != y_key:
                dados_coluna = dataset[k]
                if len(dados_coluna) > 0:
                    media_coluna = sum(dados_coluna) / len(dados_coluna)
                    effective_bias += self.pesos[i + 1] * media_coluna
                    
        lines = []
        if w_y != 0:
            a = -w_x / w_y
            intercept = -effective_bias / w_y
            angulo_deg = math.degrees(math.atan(a))
            mx, my = 0, intercept 
            sinal_wy = "+" if w_y >= 0 else "-"
            sinal_b = "+" if effective_bias >= 0 else "-"
            equacao = f"{w_x:.2f}x {sinal_wy} {abs(w_y):.2f}y {sinal_b} {abs(effective_bias):.2f} = 0"
        else:
            angulo_deg = 90
            mx, my = -effective_bias / w_x if w_x != 0 else 0, 0
            sinal_b = "+" if effective_bias >= 0 else "-"
            equacao = f"{w_x:.2f}x {sinal_b} {abs(effective_bias):.2f} = 0"
            
        lines.append({"angle": angulo_deg, "pos": (mx, my), "name": f"Fronteira 2D: {equacao}"})
        return {"lines": lines}
    
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
    
class OptimalBayesMAP(BaseClassifier):
    def train(self, X, y, **kwargs):
        self.classes = np.unique(y)
        self.parameters = {}
        
        for c in self.classes:
            X_c = X[np.array(y) == c]
            mean = np.mean(X_c, axis=0)
            # rowvar=False garante que as colunas são variáveis
            cov = np.cov(X_c, rowvar=False) 
            
            # Adiciona um pequeno valor à diagonal para evitar matriz singular
            cov += np.eye(cov.shape[0]) * 1e-6 
            
            self.parameters[c] = {'mean': mean, 'cov': cov}
            
    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        preds = []
        for x in X:
            posteriors = []
            for c in self.classes:
                mean = self.parameters[c]['mean']
                cov = self.parameters[c]['cov']
                
                inv_cov = np.linalg.inv(cov)
                det_cov = np.linalg.det(cov)
                diff = x - mean
                
                term1 = -0.5 * np.log(det_cov)
                term2 = -0.5 * np.dot(np.dot(diff.T, inv_cov), diff)
                
                posterior = term1 + term2
                posteriors.append(posterior)
            preds.append(self.classes[np.argmax(posteriors)])
        return preds[0] if len(preds) == 1 else np.array(preds)

    def get_decision_surface(self, classe_i, classe_j):
        """Retorna W, w e w0 para a fronteira quadrática entre duas classes."""
        m_i = self.parameters[classe_i]['mean']
        cov_i = self.parameters[classe_i]['cov']
        inv_cov_i = np.linalg.inv(cov_i)
        
        m_j = self.parameters[classe_j]['mean']
        cov_j = self.parameters[classe_j]['cov']
        inv_cov_j = np.linalg.inv(cov_j)
        
        W = -0.5 * (inv_cov_i - inv_cov_j)
        w = np.dot(inv_cov_i, m_i) - np.dot(inv_cov_j, m_j)
        
        term_w0_1 = -0.5 * (np.dot(np.dot(m_i.T, inv_cov_i), m_i) - np.dot(np.dot(m_j.T, inv_cov_j), m_j))
        term_w0_2 = -0.5 * np.log(np.linalg.det(cov_i) / np.linalg.det(cov_j))
        w0 = term_w0_1 + term_w0_2
        
        return W, w, w0

    def get_plot_data(self, **kwargs):
        x_key = kwargs.get('x_key')
        y_key = kwargs.get('y_key')
        keys = kwargs.get('keys')
        dataset = kwargs.get('dataset')
        x_data = kwargs.get('x_data')
        y_data = kwargs.get('y_data')
        
        if not x_key or not y_key or not keys or not dataset or len(self.classes) != 2: 
            return {}
            
        c1, c2 = self.classes[0], self.classes[1]
        W, w, w0 = self.get_decision_surface(c1, c2)
        
        idx_x = keys.index(x_key)
        idx_y = keys.index(y_key)
        hid_idx = [i for i in range(4) if i not in [idx_x, idx_y]]
        
        hid_vals = [np.mean(dataset[keys[i]]) for i in hid_idx]
        
        res = 150
        x_min, x_max = min(x_data) - 1, max(x_data) + 1
        y_min, y_max = min(y_data) - 1, max(y_data) + 1
        
        xi = np.linspace(x_min, x_max, res)
        yi = np.linspace(y_min, y_max, res)
        Z = np.zeros((res, res))
        
        for i, xv in enumerate(xi):
            for j, yv in enumerate(yi):
                vec = np.zeros(4)
                vec[idx_x] = xv
                vec[idx_y] = yv
                vec[hid_idx[0]] = hid_vals[0]
                vec[hid_idx[1]] = hid_vals[1]
                val = np.dot(vec.T, np.dot(W, vec)) + np.dot(w.T, vec) + w0
                Z[i, j] = val
                
        return {"contours": [{"Z": Z, "level": 0.0, "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max, "res": res}]}

# =====================================================================
# CLASSIFICADOR NAIVE BAYES - MAP
# =====================================================================
class NaiveBayesMAP(BaseClassifier):
    def train(self, X, y, **kwargs):
        self.classes = np.unique(y)
        self.parameters = {}
        
        for c in self.classes:
            X_c = X[np.array(y) == c]
            mean = np.mean(X_c, axis=0)
            var = np.var(X_c, axis=0) + 1e-6 # Evita divisão por zero
            self.parameters[c] = {'mean': mean, 'var': var}
            
    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        preds = []
        for x in X:
            posteriors = []
            for c in self.classes:
                mean = self.parameters[c]['mean']
                var = self.parameters[c]['var']
                
                term1 = -0.5 * np.sum(np.log(2 * np.pi * var))
                term2 = -0.5 * np.sum(((x - mean) ** 2) / var)
                
                posterior = term1 + term2
                posteriors.append(posterior)
            preds.append(self.classes[np.argmax(posteriors)])
        return preds[0] if len(preds) == 1 else np.array(preds)

# =====================================================================
# CLASSIFICADOR REDE NEURAL (MLP 1 CAMADA OCULTA)
# =====================================================================
import random

class NeuralNetworkClassifier(BaseClassifier):
    @classmethod
    def get_hyperparameters(cls):
        return [
            {"name": "Épocas", "type": "int", "min": 1, "max": 100000, "default": 10000},
            {"name": "Learning Rate", "type": "float", "min": 0.0001, "max": 10.0, "default": 0.5},
            {"name": "Neurônios Ocultos", "type": "int", "min": 1, "max": 100, "default": 2}
        ]

    def __init__(self):
        self.historico_erros = []
        self.W_ih = []
        self.b_h = []
        self.W_ho = []
        self.b_o = []
        self.reverse_map = {}

    def sigmoid(self, x):
        if x < -700: return 0.0
        return 1 / (1 + math.exp(-x))

    def train(self, X_train, y_train, **kwargs):
        epocas = kwargs.get('epocas', 10000)
        lr = kwargs.get('learning_rate', 0.5)
        hidden_neurons = kwargs.get('hidden_neurons', 2)
        
        n_inputs = len(X_train[0])
        n_outputs = 2 
        
        classes = sorted(list(set(y_train)))
        
        self.reverse_map = {0: classes[0], 1: classes[1] if len(classes) > 1 else "Resto"}
        
        y_mapped = []
        for y in y_train:
            if y == classes[0]:
                y_mapped.append([0.0, 0.0])
            else:
                y_mapped.append([1.0, 1.0])
                
        self.W_ih = [[random.uniform(-1, 1) for _ in range(n_inputs)] for _ in range(hidden_neurons)]
        self.b_h = [random.uniform(-1, 1) for _ in range(hidden_neurons)]
        self.W_ho = [[random.uniform(-1, 1) for _ in range(hidden_neurons)] for _ in range(n_outputs)]
        self.b_o = [random.uniform(-1, 1) for _ in range(n_outputs)]
        
        self.historico_erros = []
        
        for epoca in range(epocas):
            erro_epoca = 0
            
            for i in range(len(X_train)):
                entrada = X_train[i]
                alvo = y_mapped[i]
                
                # FORWARD PASS
                out_h = [0.0] * hidden_neurons
                for j in range(hidden_neurons):
                    net = sum(self.W_ih[j][k] * entrada[k] for k in range(n_inputs)) + self.b_h[j]
                    out_h[j] = self.sigmoid(net)
                    
                out_o = [0.0] * n_outputs
                for j in range(n_outputs):
                    net = sum(self.W_ho[j][k] * out_h[k] for k in range(hidden_neurons)) + self.b_o[j]
                    out_o[j] = self.sigmoid(net)
                    
                erro_padrao = 0.5 * sum((alvo[k] - out_o[k])**2 for k in range(n_outputs))
                erro_epoca += erro_padrao
                
                # BACKWARD PASS
                delta_o = [0.0] * n_outputs
                for j in range(n_outputs):
                    delta_o[j] = (alvo[j] - out_o[j]) * out_o[j] * (1 - out_o[j])
                    
                delta_h = [0.0] * hidden_neurons
                for j in range(hidden_neurons):
                    soma_erros = sum(delta_o[k] * self.W_ho[k][j] for k in range(n_outputs))
                    delta_h[j] = soma_erros * out_h[j] * (1 - out_h[j])
                    
                # UPDATE WEIGHTS
                for j in range(n_outputs):
                    for k in range(hidden_neurons):
                        self.W_ho[j][k] += lr * delta_o[j] * out_h[k]
                    self.b_o[j] += lr * delta_o[j]
                    
                for j in range(hidden_neurons):
                    for k in range(n_inputs):
                        self.W_ih[j][k] += lr * delta_h[j] * entrada[k]
                    self.b_h[j] += lr * delta_h[j]
                    
            self.historico_erros.append(erro_epoca / len(X_train))

    def predict(self, novo_ponto):
        if not self.W_ih:
            raise Exception("Modelo ainda não treinado.")
        
        n_inputs = len(novo_ponto)
        hidden_neurons = len(self.b_h)
        n_outputs = len(self.b_o)
        
        out_h = [0.0] * hidden_neurons
        for j in range(hidden_neurons):
            net = sum(self.W_ih[j][k] * novo_ponto[k] for k in range(n_inputs)) + self.b_h[j]
            out_h[j] = self.sigmoid(net)
            
        out_o = [0.0] * n_outputs
        for j in range(n_outputs):
            net = sum(self.W_ho[j][k] * out_h[k] for k in range(hidden_neurons)) + self.b_o[j]
            out_o[j] = self.sigmoid(net)
            
        classe_predita = 1 if out_o[0] >= 0.5 else 0
        nome_classe = self.reverse_map.get(classe_predita, "Desconhecido")
        
        return nome_classe, {"Ativação O1": out_o[0], "Ativação O2": out_o[1]}
import math
import numpy as np
import random
from .base import BaseClassifier

class NeuralNetworkClassifier(BaseClassifier):
    @classmethod
    def get_hyperparameters(cls):
        return [
            {"name": "Épocas", "type": "int", "min": 1, "max": 100000, "default": 1000},
            {"name": "Learning Rate", "type": "float", "min": 0.0001, "max": 10.0, "default": 0.05},
            {"name": "Camadas Ocultas", "type": "string", "default": "4,4"},
            {"name": "Semente Aleatória", "type": "int", "min": 0, "max": 99999, "default": 67}
        ]

    def __init__(self):
        self.historico_erros = []
        self.weights = []
        self.biases = []
        self.reverse_map = {}
        self.last_forward_pass = {"activations": []}
        self.feature_mins = []
        self.feature_maxs = []

    def sigmoid(self, x):
        if x < -700: return 0.0
        return 1 / (1 + math.exp(-x))
        
    def _normalize(self, x):
        if not self.feature_mins: return x
        norm_x = []
        for i in range(len(x)):
            rng = self.feature_maxs[i] - self.feature_mins[i]
            if rng == 0:
                norm_x.append(0.0)
            else:
                norm_x.append((x[i] - self.feature_mins[i]) / rng)
        return norm_x

    def train(self, X_train, y_train, **kwargs):
        epocas = kwargs.get('epocas', 1000)
        lr = kwargs.get('learning_rate', 0.05)
        semente = kwargs.get('semente_aleatoria', 67)
        camadas_str = kwargs.get('camadas_ocultas', '4,4')
        progress_callback = kwargs.get('progress_callback', None)
        
        try:
            hidden_layers = [int(x.strip()) for x in str(camadas_str).split(',') if x.strip().isdigit()]
            if not hidden_layers: hidden_layers = [4, 4]
        except:
            hidden_layers = [4, 4]
            
        n_inputs = len(X_train[0]) if len(X_train) > 0 else 0
        if n_inputs > 0:
            self.feature_mins = [min(X_train[i][j] for i in range(len(X_train))) for j in range(n_inputs)]
            self.feature_maxs = [max(X_train[i][j] for i in range(len(X_train))) for j in range(n_inputs)]
            X_train = [self._normalize(x) for x in X_train]
            
        random.seed(semente)
        np.random.seed(semente)
        
        n_inputs = len(X_train[0])
        classes = sorted(list(set(y_train)))
        n_outputs = len(classes) 
        
        self.reverse_map = {i: classes[i] for i in range(len(classes))}
        
        y_mapped = []
        for y in y_train:
            idx = classes.index(y)
            vec = [0.0] * n_outputs
            vec[idx] = 1.0
            y_mapped.append(vec)
            
        layers = [n_inputs] + hidden_layers + [n_outputs]
        self.weights = []
        self.biases = []
        
        for i in range(len(layers) - 1):
            w = [[random.uniform(-1, 1) for _ in range(layers[i])] for _ in range(layers[i+1])]
            b = [random.uniform(-1, 1) for _ in range(layers[i+1])]
            self.weights.append(w)
            self.biases.append(b)
            
        self.historico_erros = []
        paciencia = max(10, epocas // 100)
        
        for epoca in range(epocas):
            erros_classificacao = 0
            
            for entrada, alvo in zip(X_train, y_mapped):
                activations = [entrada]
                
                # FORWARD PASS
                for w_layer, b_layer in zip(self.weights, self.biases):
                    out = []
                    for j in range(len(b_layer)):
                        net = sum(w_layer[j][k] * activations[-1][k] for k in range(len(w_layer[j]))) + b_layer[j]
                        out.append(self.sigmoid(net))
                    activations.append(out)
                    
                # Contabilizar erro
                out_o = activations[-1]
                if out_o.index(max(out_o)) != alvo.index(max(alvo)):
                    erros_classificacao += 1
                
                # BACKWARD PASS
                deltas = [None] * len(self.weights)
                
                # Output delta
                L_out = len(self.weights) - 1
                deltas[L_out] = [(alvo[j] - out_o[j]) * out_o[j] * (1 - out_o[j]) for j in range(n_outputs)]
                
                # Hidden deltas
                for L in range(L_out - 1, -1, -1):
                    deltas[L] = []
                    for j in range(len(self.biases[L])):
                        soma_erros = sum(deltas[L+1][k] * self.weights[L+1][k][j] for k in range(len(self.biases[L+1])))
                        act_j = activations[L+1][j]
                        deltas[L].append(soma_erros * act_j * (1 - act_j))
                        
                # UPDATE WEIGHTS
                for L in range(len(self.weights)):
                    for j in range(len(self.biases[L])):
                        for k in range(len(activations[L])):
                            self.weights[L][j][k] += lr * deltas[L][j] * activations[L][k]
                        self.biases[L][j] += lr * deltas[L][j]
                        
            self.historico_erros.append(erros_classificacao)
            
            step = max(1, epocas // 10)
            if progress_callback and (epoca + 1) % step == 0:
                progress_callback(f"  • Progresso: {epoca + 1}/{epocas} épocas concluídas... ({erros_classificacao} erros na atual)")
                
            # --- EARLY STOPPING ---
            # Se a rede zerar os erros, aguardamos mais algumas épocas (paciência)
            # apenas para garantir uma margem de confiança melhor nos pesos antes de parar.
            if erros_classificacao == 0:
                paciencia -= 1
                if paciencia <= 0:
                    if progress_callback:
                        progress_callback(f"  » Parada Antecipada (Early Stopping) na época {epoca+1}: 0 erros de classificação!")
                    break
            else:
                paciencia = max(10, epocas // 100) # Reset da paciência

    def predict(self, novo_ponto):
        if not self.weights:
            raise Exception("Modelo ainda não treinado.")
            
        novo_ponto = self._normalize(novo_ponto)
        
        activations = [novo_ponto]
        for w_layer, b_layer in zip(self.weights, self.biases):
            out = []
            for j in range(len(b_layer)):
                net = sum(w_layer[j][k] * activations[-1][k] for k in range(len(w_layer[j]))) + b_layer[j]
                out.append(self.sigmoid(net))
            activations.append(out)
            
        self.last_forward_pass = {"activations": activations}
        
        out_o = activations[-1]
        idx_max = out_o.index(max(out_o))
        nome_classe = self.reverse_map[idx_max]
        
        probs = {self.reverse_map[i]: round(out_o[i], 4) for i in range(len(out_o))}
        
        return nome_classe, {"Probabilidades (Soft)": probs}

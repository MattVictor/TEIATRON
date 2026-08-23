import math
import numpy as np
import random
from .base import BaseClassifier

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
        self.last_forward_pass = {"inputs": [], "hidden": [], "output": []}

    def sigmoid(self, x):
        if x < -700: return 0.0
        return 1 / (1 + math.exp(-x))

    def train(self, X_train, y_train, **kwargs):
        epocas = kwargs.get('epocas', 10000)
        lr = kwargs.get('learning_rate', 0.5)
        hidden_neurons = kwargs.get('hidden_neurons', 2)
        progress_callback = kwargs.get('progress_callback', None)
        
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
                
        self.W_ih = [[random.uniform(-1, 1) for _ in range(n_inputs)] for _ in range(hidden_neurons)]
        self.b_h = [random.uniform(-1, 1) for _ in range(hidden_neurons)]
        self.W_ho = [[random.uniform(-1, 1) for _ in range(hidden_neurons)] for _ in range(n_outputs)]
        self.b_o = [random.uniform(-1, 1) for _ in range(n_outputs)]
        
        self.historico_erros = []
        
        for epoca in range(epocas):
            erros_classificacao = 0
            
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
                    
                # Contabilizar se a predição atual (argmax) foi incorreta
                if out_o.index(max(out_o)) != alvo.index(max(alvo)):
                    erros_classificacao += 1
                
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
                    
            self.historico_erros.append(erros_classificacao)
            
            # Reportar progresso a cada 10% (mínimo de 1)
            step = max(1, epocas // 10)
            if progress_callback and (epoca + 1) % step == 0:
                progress_callback(f"  ↳ Progresso: {epoca + 1}/{epocas} épocas concluídas... ({erros_classificacao} erros na atual)")
            
        # Salvar estado final
        self.last_forward_pass = {
            "inputs": X_train[-1] if len(X_train) > 0 else [],
            "hidden": out_h if 'out_h' in locals() else [],
            "output": out_o if 'out_o' in locals() else []
        }

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
            
        self.last_forward_pass = {
            "inputs": novo_ponto,
            "hidden": out_h,
            "output": out_o
        }
            
        classe_predita = out_o.index(max(out_o))
        nome_classe = self.reverse_map.get(classe_predita, "Desconhecido")
        
        dic_ativacoes = {f"Ativação O{i+1}": out_o[i] for i in range(n_outputs)}
        return nome_classe, dic_ativacoes


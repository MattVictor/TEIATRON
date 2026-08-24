import math
import numpy as np
from .base import BaseClassifier

class PerceptronClassifier(BaseClassifier):
    @classmethod
    def get_hyperparameters(cls):
        return [
            {"name": "Regra Delta", "type": "bool", "default": False},
            {"name": "Estratégia", "type": "options", "choices": ["Clássico", "Um contra todos"], "default": "Clássico"},
            {"name": "Classe 1", "type": "class_selector", "default": "Iris-setosa", "depends_on": {"field": "Estratégia", "value": "Clássico"}},
            {"name": "Classe 2", "type": "class_selector", "default": "Iris-versicolor", "depends_on": {"field": "Estratégia", "value": "Clássico"}, "prevent_same_as": "Classe 1"},
            {"name": "Classe Alvo", "type": "class_selector", "default": "Iris-setosa", "depends_on": {"field": "Estratégia", "value": "Um contra todos"}},
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
        # O alvo vira 1, o resto vira -1.
        self.class_map = {classe_alvo: 1}
        
        # Descobre qual é a segunda classe (se for binário) ou chama de "Resto" (se for OvA)
        outras_classes = [y for y in set(y_train) if y != classe_alvo]
        nome_classe_negativa = outras_classes[0] if len(outras_classes) == 1 else "Resto"
        self.reverse_map = {1: classe_alvo, -1: nome_classe_negativa}
        
        y_mapped = [1 if y == classe_alvo else -1 for y in y_train]
        
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
                    
                # Predição Discreta (Degrau) no limiar 0.0
                classe_predita = 1 if ativacao_continua >= 0.0 else -1
                
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
            
        classe_predita = 1 if ativacao >= 0.0 else -1
        nome_classe = self.reverse_map[classe_predita]
        
        return nome_classe, {"Ativação (Soma Ponderada)": ativacao}

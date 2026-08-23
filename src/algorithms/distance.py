import math
import numpy as np
from .base import BaseClassifier

class MinDistanceClassifier(BaseClassifier):
    @classmethod
    def get_hyperparameters(cls):
        return [
            {"name": "Multiclasse", "type": "bool", "default": True},
            {"name": "Classe 1", "type": "class_selector", "default": "Iris-setosa", "depends_on": {"field": "Multiclasse", "value": False}},
            {"name": "Classe 2", "type": "class_selector", "default": "Iris-versicolor", "depends_on": {"field": "Multiclasse", "value": False}, "prevent_same_as": "Classe 1"}
        ]

    def __init__(self):
        self.centroids = {}
        self.classes_trained = []

    def train(self, X_train, y_train, **kwargs):
        grupos = {}
        for ponto, classe in zip(X_train, y_train):
            if classe not in grupos:
                grupos[classe] = []
            grupos[classe].append(ponto)

        self.centroids = {}
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
        if not self.centroids:
            raise Exception("O modelo ainda não foi treinado.")
            
        melhor_classe = None
        menor_distancia = float('inf')
        distancias_calculadas = {}

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

class MaxDistanceClassifier(BaseClassifier):
    @classmethod
    def get_hyperparameters(cls):
        return [
            {"name": "Multiclasse", "type": "bool", "default": True},
            {"name": "Classe 1", "type": "class_selector", "default": "Iris-setosa", "depends_on": {"field": "Multiclasse", "value": False}},
            {"name": "Classe 2", "type": "class_selector", "default": "Iris-versicolor", "depends_on": {"field": "Multiclasse", "value": False}, "prevent_same_as": "Classe 1"}
        ]

    def __init__(self):
        self.X_train = []
        self.y_train = []
        self.classes_trained = []

    def train(self, X_train, y_train, **kwargs):
        self.X_train = X_train
        self.y_train = y_train
        self.classes_trained = list(sorted(set(y_train)))
        return self.classes_trained

    def predict(self, novo_ponto):
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

import math
import numpy as np

class BaseClassifier:
    def train(self, X_train, y_train, **kwargs):
        raise NotImplementedError
        
    def predict(self, novo_ponto):
        raise NotImplementedError
        
    def get_logs(self):
        return []

    @classmethod
    def get_hyperparameters(cls):
        """Retorna os hiperparâmetros necessários para gerar a UI dinamicamente."""
        return []

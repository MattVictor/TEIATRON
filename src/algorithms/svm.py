import math
import numpy as np
from .base import BaseClassifier

try:
    from sklearn.svm import SVC
except ImportError:
    SVC = None

class SVMClassifier(BaseClassifier):
    @classmethod
    def get_hyperparameters(cls):
        return [
            {"name": "Estratégia", "type": "options", "choices": ["Clássico", "Um contra todos"], "default": "Clássico"},
            {"name": "Classe 1", "type": "class_selector", "default": "Iris-setosa", "depends_on": {"field": "Estratégia", "value": "Clássico"}},
            {"name": "Classe 2", "type": "class_selector", "default": "Iris-versicolor", "depends_on": {"field": "Estratégia", "value": "Clássico"}, "prevent_same_as": "Classe 1"},
            {"name": "Classe Alvo", "type": "class_selector", "default": "Iris-setosa", "depends_on": {"field": "Estratégia", "value": "Um contra todos"}},
            {"name": "Kernel", "type": "options", "choices": ["linear", "poly", "rbf", "sigmoid"], "default": "linear"},
            {"name": "C (Regularização)", "type": "float", "default": 1.0},
            {"name": "Grau (Poly)", "type": "int", "default": 3, "depends_on": {"field": "Kernel", "value": "poly"}}
        ]
        
    def __init__(self):
        self.model = None
        self.reverse_map = {}
        
    def train(self, X, y, **kwargs):
        if SVC is None:
            raise Exception("A biblioteca scikit-learn não está instalada. Abra o terminal e digite: pip install scikit-learn")
            
        kernel = kwargs.get('Kernel', 'linear')
        C = kwargs.get('C (Regularização)', 1.0)
        degree = kwargs.get('Grau (Poly)', 3)
        
        # Mapeamento para 0 e 1, necessário se as classes vierem como strings
        classes = sorted(list(set(y)))
        self.reverse_map = {0: classes[0], 1: classes[1] if len(classes) > 1 else "Resto"}
        y_mapped = [0 if val == classes[0] else 1 for val in y]
        
        self.model = SVC(kernel=kernel, C=C, degree=degree)
        self.model.fit(X, y_mapped)
        
    def predict(self, X):
        if self.model is None:
            raise Exception("Modelo ainda não treinado.")
            
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        preds = self.model.predict(X)
        
        if len(preds) == 1:
            dist = self.model.decision_function(X)[0]
            classe_predita = self.reverse_map.get(preds[0], "Desconhecido")
            return classe_predita, {"Distância à Margem": dist}
            
        return [self.reverse_map.get(p, "Desconhecido") for p in preds]
        


import math
import numpy as np
from .base import BaseClassifier

class OptimalBayesMAP(BaseClassifier):
    @classmethod
    def get_hyperparameters(cls):
        return [
            {"name": "Matriz de Covariância", "type": "options", "choices": ["Individuais (QDA - Curvas)", "Agrupada (LDA - Retas)"], "default": "Individuais (QDA - Curvas)"},
            {"name": "Probabilidade a Priori", "type": "options", "choices": ["Uniforme (Equiprovável)", "Empírica (Baseada nos Dados)"], "default": "Uniforme (Equiprovável)"}
        ]
        
    def __init__(self):
        self.classes = []
        self.parameters = {}
        self.priors = {}

    def train(self, X, y, **kwargs):
        self.classes = np.unique(y)
        self.parameters = {}
        self.priors = {}
        
        tipo_cov = kwargs.get('Matriz de Covariância', 'Individuais (QDA - Curvas)')
        tipo_prior = kwargs.get('Probabilidade a Priori', 'Uniforme (Equiprovável)')
        
        for c in self.classes:
            X_c = X[y == c]
            mean = np.mean(X_c, axis=0)
            cov = np.cov(X_c, rowvar=False) 
            self.parameters[c] = {'mean': mean, 'cov': cov}
            
            if "Empírica" in tipo_prior:
                self.priors[c] = len(X_c) / len(X)
            else:
                self.priors[c] = 1.0 / len(self.classes)
                
        # Lógica para LDA (Agrupada): Tirar média ponderada das matrizes de covariância
        if "Agrupada" in tipo_cov:
            pooled_cov = np.zeros((X.shape[1], X.shape[1]))
            for c in self.classes:
                X_c = X[y == c]
                if len(X_c) > 1:
                    pooled_cov += np.cov(X_c, rowvar=False) * (len(X_c) - 1)
            pooled_cov /= (len(X) - len(self.classes))
            # Substituir a cov individual pela agrupada em todas as classes
            for c in self.classes:
                self.parameters[c]['cov'] = pooled_cov
            
    def get_decision_surface(self, c1, c2):
        mean_1 = self.parameters[c1]['mean']
        cov_1 = self.parameters[c1]['cov']
        try:
            inv_cov_1 = np.linalg.inv(cov_1)
        except:
            inv_cov_1 = np.eye(len(mean_1))
            
        mean_2 = self.parameters[c2]['mean']
        cov_2 = self.parameters[c2]['cov']
        try:
            inv_cov_2 = np.linalg.inv(cov_2)
        except:
            inv_cov_2 = np.eye(len(mean_2))
            
        prior_1 = self.priors[c1]
        prior_2 = self.priors[c2]
        
        W = -0.5 * (inv_cov_1 - inv_cov_2)
        w = np.dot(inv_cov_1, mean_1) - np.dot(inv_cov_2, mean_2)
        
        term1 = -0.5 * (np.dot(np.dot(mean_1.T, inv_cov_1), mean_1) - np.dot(np.dot(mean_2.T, inv_cov_2), mean_2))
        
        det_cov_1 = max(np.linalg.det(cov_1), 1e-10)
        det_cov_2 = max(np.linalg.det(cov_2), 1e-10)
        
        term2 = -0.5 * np.log(det_cov_1 / det_cov_2)
        term3 = np.log(prior_1) - np.log(prior_2)
        
        w0 = term1 + term2 + term3
        return W, w, w0

    def _predict_single(self, x):
        posteriors = []
        for c in self.classes:
            mean = self.parameters[c]['mean']
            cov = self.parameters[c]['cov']
            prior = self.priors[c]
            
            try:
                inv_cov = np.linalg.inv(cov)
                det_cov = np.linalg.det(cov)
                if det_cov <= 0: det_cov = 1e-10
            except:
                inv_cov = np.eye(len(mean))
                det_cov = 1e-10
                
            diff = x - mean
            
            term1 = -0.5 * np.log(det_cov)
            term2 = -0.5 * np.dot(np.dot(diff.T, inv_cov), diff)
            term3 = np.log(prior) # Inclusão do Prior
            
            posterior = term1 + term2 + term3
            posteriors.append(posterior)
            
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        preds = np.array([self._predict_single(x) for x in X])
        
        if len(preds) == 1:
            return preds[0]
        return preds

class NaiveBayesMAP(BaseClassifier):
    @classmethod
    def get_hyperparameters(cls):
        return [
            {"name": "Probabilidade a Priori", "type": "options", "choices": ["Uniforme (Equiprovável)", "Empírica (Baseada nos Dados)"], "default": "Uniforme (Equiprovável)"}
        ]
        
    def __init__(self):
        self.classes = []
        self.parameters = {}
        self.priors = {}

    def train(self, X, y, **kwargs):
        self.classes = np.unique(y)
        self.parameters = {}
        self.priors = {}
        
        tipo_prior = kwargs.get('Probabilidade a Priori', 'Uniforme (Equiprovável)')
        
        for c in self.classes:
            X_c = X[y == c]
            mean = np.mean(X_c, axis=0)
            var = np.var(X_c, axis=0)
            # Evitar variância 0
            var[var == 0] = 1e-9
            self.parameters[c] = {'mean': mean, 'var': var}
            
            if "Empírica" in tipo_prior:
                self.priors[c] = len(X_c) / len(X)
            else:
                self.priors[c] = 1.0 / len(self.classes)
            
    def _predict_single(self, x):
        posteriors = []
        for c in self.classes:
            mean = self.parameters[c]['mean']
            var = self.parameters[c]['var']
            prior = self.priors[c]
            
            term1 = -0.5 * np.sum(np.log(2 * np.pi * var))
            term2 = -0.5 * np.sum(((x - mean) ** 2) / var)
            term3 = np.log(prior) # Inclusão do Prior
            
            posterior = term1 + term2 + term3
            posteriors.append(posterior)
            
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        preds = np.array([self._predict_single(x) for x in X])
        if len(preds) == 1:
            return preds[0]
        return preds

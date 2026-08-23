import math
import numpy as np

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
    

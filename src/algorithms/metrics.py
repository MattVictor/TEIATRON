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

    def acerto_causal(self):
        if self.total == 0: return 0
        pe_sum = 0
        for i in range(self.n):
            linha_total = sum(self.matriz[i])
            coluna_total = sum(self.matriz[j][i] for j in range(self.n))
            pe_sum += linha_total * coluna_total
        return pe_sum / (self.total ** 2)

    def variancia_kappa(self):
        m = self.total
        c = self.n
        if m == 0: return 0
        
        phi_1 = self.acerto_geral()
        phi_2 = self.acerto_causal()
        
        linha_totals = [sum(self.matriz[i]) for i in range(c)]
        coluna_totals = [sum(self.matriz[j][i] for j in range(c)) for i in range(c)]
        
        sum_phi_3 = 0
        for i in range(c):
            sum_phi_3 += self.matriz[i][i] * (linha_totals[i] + coluna_totals[i])
        phi_3 = sum_phi_3 / (m ** 2)
        
        sum_phi_4 = 0
        for i in range(c):
            for j in range(c):
                sum_phi_4 += self.matriz[i][j] * ((linha_totals[j] + coluna_totals[i]) ** 2)
        phi_4 = sum_phi_4 / (m ** 3)
        
        den1 = (1 - phi_2) ** 2
        den2 = (1 - phi_2) ** 3
        den3 = (1 - phi_2) ** 4
        
        if den3 == 0: return 0
        
        term1 = (phi_1 * (1 - phi_1)) / den1
        term2 = (2 * (1 - phi_1) * (2 * phi_1 * phi_2 - phi_3)) / den2
        term3 = ((1 - phi_1) ** 2 * (phi_4 - 4 * (phi_2 ** 2))) / den3
        
        return (1 / m) * (term1 + term2 + term3)

    def coeficiente_kappa(self):
        po = self.acerto_geral()
        pe = self.acerto_causal()
        return (po - pe) / (1 - pe) if (1 - pe) != 0 else 0

    def variancia_tau(self):
        m = self.total
        c = self.n
        if m == 0 or c <= 1: return 0
        
        ag = self.acerto_geral()
        num = ag * (1 - ag)
        den = (1 - (1/c)) ** 2
        
        if den == 0: return 0
        return (1 / m) * (num / den)

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
    

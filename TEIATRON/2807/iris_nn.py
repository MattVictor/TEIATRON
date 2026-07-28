import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTableWidget, QTableWidgetItem, 
                             QLabel, QComboBox, QHeaderView, QGroupBox, QPushButton)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

# Imports de Machine Learning
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             fbeta_score, cohen_kappa_score, matthews_corrcoef)

class ClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TEIA - Comparador Dinâmico de Modelos (Iris)")
        self.resize(1100, 650)
        
        self.nomes_classes = ['Setosa', 'Versicolor', 'Virginica']
        self.iris = load_iris()
        
        self.configurar_interface()
        self.treinar_modelos() # Executa o primeiro treinamento ao abrir
        self.atualizar_interface()

    def treinar_modelos(self):
        """Sorteia os dados sem semente fixa e treina os modelos novamente"""
        
        # NOTE O PARÂMETRO 'random_state' FOI REMOVIDO PARA ALEATORIZAÇÃO REAL
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.iris.data, self.iris.target, test_size=0.3, stratify=self.iris.target
        )
        
        # Reinicializa os modelos para evitar que continuem aprendendo do estado anterior
        self.modelos = {
            "Feedforward (MLP)": MLPClassifier(hidden_layer_sizes=(10,), max_iter=2000),
            "Bayes Ótimo (QDA)": QuadraticDiscriminantAnalysis(),
            "Naive Bayes": GaussianNB()
        }
        
        # Treina e armazena predições
        self.predicoes = {}
        for nome, modelo in self.modelos.items():
            modelo.fit(self.X_train, self.y_train)
            self.predicoes[nome] = modelo.predict(self.X_test)

    def configurar_interface(self):
        widget_central = QWidget()
        self.setCentralWidget(widget_central)
        layout_principal = QHBoxLayout(widget_central)

        # ================= ESQUERDA: Tabela de Testes =================
        painel_esquerdo = QVBoxLayout()
        grupo_tabela = QGroupBox("Dados de Teste e Classificação (Modelo A)")
        layout_grupo_tabela = QVBoxLayout()
        
        self.tabela_dados = QTableWidget()
        self.tabela_dados.setColumnCount(6)
        self.tabela_dados.setHorizontalHeaderLabels(
            ["Comp. Sépala", "Larg. Sépala", "Comp. Pétala", "Larg. Pétala", "Classe Real", "Predição"]
        )
        self.tabela_dados.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout_grupo_tabela.addWidget(self.tabela_dados)
        grupo_tabela.setLayout(layout_grupo_tabela)
        painel_esquerdo.addWidget(grupo_tabela)
        
        # ================= DIREITA: Comparador e Métricas =================
        painel_direito = QVBoxLayout()
        
        # ---> NOVO: Botão de Retreinamento Aleatório <---
        self.btn_retreinar = QPushButton("🔄 Gerar Novos Dados Aleatórios e Retreinar")
        self.btn_retreinar.setStyleSheet("padding: 15px; font-weight: bold; background-color: #007bff; color: white;")
        self.btn_retreinar.clicked.connect(self.acao_retreinar)
        painel_direito.addWidget(self.btn_retreinar)
        
        grupo_metricas = QGroupBox("Comparação de Métricas de Qualidade")
        layout_grupo_metricas = QVBoxLayout()
        
        # Seletores de Modelos
        layout_seletores = QHBoxLayout()
        
        layout_mod_a = QVBoxLayout()
        layout_mod_a.addWidget(QLabel("<b>Modelo A (Principal):</b>"))
        self.combo_modelo_a = QComboBox()
        self.combo_modelo_a.addItems(["Feedforward (MLP)", "Bayes Ótimo (QDA)", "Naive Bayes"])
        self.combo_modelo_a.currentTextChanged.connect(self.atualizar_interface)
        layout_mod_a.addWidget(self.combo_modelo_a)
        
        layout_mod_b = QVBoxLayout()
        layout_mod_b.addWidget(QLabel("<b>Modelo B (Comparação):</b>"))
        self.combo_modelo_b = QComboBox()
        self.combo_modelo_b.addItems(["Feedforward (MLP)", "Bayes Ótimo (QDA)", "Naive Bayes"])
        self.combo_modelo_b.setCurrentIndex(1) # Seleciona o segundo por padrão
        self.combo_modelo_b.currentTextChanged.connect(self.atualizar_interface)
        layout_mod_b.addWidget(self.combo_modelo_b)
        
        layout_seletores.addLayout(layout_mod_a)
        layout_seletores.addLayout(layout_mod_b)
        layout_grupo_metricas.addLayout(layout_seletores)
        
        # Tabela de Métricas
        self.tabela_metricas = QTableWidget()
        self.tabela_metricas.setColumnCount(4)
        self.tabela_metricas.setHorizontalHeaderLabels(["Métrica", "Modelo A", "Modelo B", "Saldo (A - B)"])
        self.tabela_metricas.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout_grupo_metricas.addWidget(self.tabela_metricas)
        
        grupo_metricas.setLayout(layout_grupo_metricas)
        painel_direito.addWidget(grupo_metricas)
        
        # Proporções da tela (60% esquerda, 40% direita)
        layout_principal.addLayout(painel_esquerdo, 6)
        layout_principal.addLayout(painel_direito, 4)

    def acao_retreinar(self):
        """Função chamada quando o botão de retreinar é clicado"""
        self.treinar_modelos()
        self.atualizar_interface()

    def calcular_todas_metricas(self, y_true, y_pred):
        return {
            "Acerto Geral (Acurácia)": accuracy_score(y_true, y_pred),
            "Acurácia Produtor (Recall)": recall_score(y_true, y_pred, average='macro', zero_division=0),
            "Acurácia Usuário (Precisão)": precision_score(y_true, y_pred, average='macro', zero_division=0),
            "Coeficiente Kappa": cohen_kappa_score(y_true, y_pred),
            "Coef. Matthews (MCC)": matthews_corrcoef(y_true, y_pred),
            "F1 Score (b=1)": fbeta_score(y_true, y_pred, beta=1, average='macro', zero_division=0),
            "F2 Score (b=2)": fbeta_score(y_true, y_pred, beta=2, average='macro', zero_division=0)
        }

    def atualizar_interface(self):
        nome_mod_a = self.combo_modelo_a.currentText()
        nome_mod_b = self.combo_modelo_b.currentText()
        
        # Se os dados ainda não estiverem na memória (segurança)
        if not hasattr(self, 'predicoes') or nome_mod_a not in self.predicoes:
            return
            
        pred_a = self.predicoes[nome_mod_a]
        pred_b = self.predicoes[nome_mod_b]
        
        # 1. Atualizar Tabela de Dados
        self.tabela_dados.setRowCount(len(self.X_test))
        for i, (features, real, pred) in enumerate(zip(self.X_test, self.y_test, pred_a)):
            for j in range(4):
                self.tabela_dados.setItem(i, j, QTableWidgetItem(f"{features[j]:.1f}"))
            
            item_real = QTableWidgetItem(self.nomes_classes[real])
            item_pred = QTableWidgetItem(self.nomes_classes[pred])
            
            # Cores para acertos e erros
            if real == pred:
                item_pred.setBackground(QColor("#006715")) # Verde
            else:
                item_pred.setBackground(QColor("#a5000e")) # Vermelho
                
            self.tabela_dados.setItem(i, 4, item_real)
            self.tabela_dados.setItem(i, 5, item_pred)

        # 2. Atualizar Tabela de Métricas
        metricas_a = self.calcular_todas_metricas(self.y_test, pred_a)
        metricas_b = self.calcular_todas_metricas(self.y_test, pred_b)
        
        self.tabela_metricas.setRowCount(len(metricas_a))
        
        row = 0
        for chave in metricas_a.keys():
            val_a = metricas_a[chave]
            val_b = metricas_b[chave]
            saldo = val_a - val_b
            
            self.tabela_metricas.setItem(row, 0, QTableWidgetItem(chave))
            self.tabela_metricas.setItem(row, 1, QTableWidgetItem(f"{val_a:.4f}"))
            self.tabela_metricas.setItem(row, 2, QTableWidgetItem(f"{val_b:.4f}"))
            
            # Formatando o saldo
            item_saldo = QTableWidgetItem(f"{saldo:+.4f}")
            item_saldo.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            
            if saldo > 0:
                item_saldo.setForeground(QColor("green"))
                item_saldo.setFont(item_saldo.font())
            elif saldo < 0:
                item_saldo.setForeground(QColor("red"))
            else:
                item_saldo.setForeground(QColor("gray"))
                
            self.tabela_metricas.setItem(row, 3, item_saldo)
            row += 1

if __name__ == "__main__":
    app = QApplication(sys.argv)
    janela = ClassifierApp()
    janela.show()
    sys.exit(app.exec())
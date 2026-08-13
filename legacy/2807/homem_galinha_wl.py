import sys
import numpy as np
from PIL import Image
from sklearn.neural_network import MLPClassifier
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QTextEdit, QHBoxLayout)
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtCore import Qt

class ReconhecedorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # Configurações da Janela Principal
        self.setWindowTitle("Classificador de Padrões: Homem vs Galinha")
        self.setGeometry(100, 100, 500, 600)
        
        self.rede_neural = None
        
        self.initUI()
        self.treinar_rede()
        
    def initUI(self):
        # Widget central e Layout principal
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(15)
        
        # 1. Área de Log (Mostrando os dados de treinamento)
        layout.addWidget(QLabel("<b>Log e Dados de Treinamento:</b>"))
        self.log_treino = QTextEdit()
        self.log_treino.setReadOnly(True)
        self.log_treino.setMaximumHeight(150)
        self.log_treino.setStyleSheet("background-color: #000000; font-family: monospace;")
        layout.addWidget(self.log_treino)
        
        # 2. Botão de Seleção de Arquivo
        self.btn_selecionar = QPushButton("Selecionar Imagem para Teste")
        self.btn_selecionar.setMinimumHeight(40)
        self.btn_selecionar.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; font-size: 14px;")
        self.btn_selecionar.clicked.connect(self.selecionar_imagem)
        self.btn_selecionar.setEnabled(False) # Habilita apenas após treinar a rede
        layout.addWidget(self.btn_selecionar)
        
        # 3. Área para mostrar a imagem selecionada
        self.lbl_imagem = QLabel("Nenhuma imagem selecionada")
        self.lbl_imagem.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_imagem.setMinimumHeight(200)
        self.lbl_imagem.setStyleSheet("border: 1px dashed #aaa;")
        layout.addWidget(self.lbl_imagem)
        
        # 4. Área do Resultado
        self.lbl_resultado = QLabel("Aguardando teste...")
        self.lbl_resultado.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_resultado.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.lbl_resultado.setStyleSheet("color: #333333;")
        layout.addWidget(self.lbl_resultado)
        
    def preprocessar_imagem(self, caminho):
        """Converte a imagem em um vetor 1D de 64 posições normalizado"""
        try:
            img = Image.open(caminho).convert('L') # Tons de cinza
            img = img.resize((8, 8))               # Reduz para 8x8
            vetor = np.array(img).flatten() / 255.0
            return vetor
        except Exception as e:
            self.log_treino.append(f"<font color='red'>Erro ao carregar '{caminho}': {str(e)}</font>")
            return None
            
    def treinar_rede(self):
        """Treina a rede neural assim que o aplicativo abre"""
        self.log_treino.append("Iniciando extração de características (Pré-processamento)...")
        QApplication.processEvents() # Atualiza a tela imediatamente
        
        vetor_homem = self.preprocessar_imagem("C:/Users/ACER/Documents/GitHub/TEIATRON/TEIATRON/2807/homem.jpg")
        vetor_galinha = self.preprocessar_imagem("C:/Users/ACER/Documents/GitHub/TEIATRON/TEIATRON/2807/galinha.png")
        
        if vetor_homem is None or vetor_galinha is None:
            self.log_treino.append("<b><font color='red'>ALERTA: Imagens de treino não encontradas. Coloque 'homem.jpg' e 'galinha.png' na raiz.</font></b>")
            return
            
        # Exibindo uma amostra dos dados na tela para o usuário ver a transformação matemática
        self.log_treino.append(f"Vetor Homem [64 atributos]: {np.round(vetor_homem[:8], 2)}...")
        self.log_treino.append(f"Vetor Galinha [64 atributos]: {np.round(vetor_galinha[:8], 2)}...")
        
        # Preparando X (entradas) e y (saídas: 0=Homem, 1=Galinha)
        X = np.array([vetor_homem, vetor_galinha])
        y = np.array([0, 1])
        
        self.log_treino.append("\nIniciando treinamento da Rede Neural (MLP)...")
        QApplication.processEvents()
        
        # Treinamento do modelo
        self.rede_neural = MLPClassifier(hidden_layer_sizes=(10,), max_iter=2000, learning_rate_init=0.05, random_state=42)
        self.rede_neural.fit(X, y)
        
        self.log_treino.append("<b>Treinamento concluído! Modelo pronto para classificação.</b>")
        self.btn_selecionar.setEnabled(True) # Libera o botão para o usuário
        
    def selecionar_imagem(self):
        """Abre o explorador de arquivos e classifica a imagem escolhida"""
        # Abre a janela nativa de seleção de arquivos do sistema operacional
        caminho_arquivo, _ = QFileDialog.getOpenFileName(
            self, 
            "Selecione uma imagem para classificar", 
            "", 
            "Imagens (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if caminho_arquivo:
            # 1. Mostra a imagem na tela
            pixmap = QPixmap(caminho_arquivo)
            self.lbl_imagem.setPixmap(pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation))
            
            # 2. Pré-processa a imagem selecionada
            vetor_teste = self.preprocessar_imagem(caminho_arquivo)
            
            if vetor_teste is not None:
                # 3. Faz a previsão
                previsao = self.rede_neural.predict([vetor_teste])[0]
                probabilidades = self.rede_neural.predict_proba([vetor_teste])[0]
                
                # 4. Atualiza o resultado na tela
                if previsao == 0:
                    classe = "HOMEM"
                    cor = "blue"
                else:
                    classe = "GALINHA"
                    cor = "darkorange"
                    
                confianca = probabilidades[previsao] * 100
                
                self.lbl_resultado.setStyleSheet(f"color: {cor};")
                self.lbl_resultado.setText(f"Classificado como: {classe}\n(Confiança: {confianca:.2f}%)")

if __name__ == '__main__':
    # Inicializa a aplicação PyQt
    app = QApplication(sys.argv)
    janela = ReconhecedorApp()
    janela.show()
    # Executa o loop principal
    sys.exit(app.exec())
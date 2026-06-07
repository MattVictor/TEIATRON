import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QSplitter, QStackedWidget, QLabel, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from config import setup_pyqtgraph, BG_MAIN, ACCENT_COLOR

from view_input import InputCard, InputExpandedPage
from view_charts import ChartsCard, ChartsExpandedPage
from view_algorithm import AlgorithmCard, AlgorithmExpandedPage
from view_accuracy import AccuracyCard, AccuracyExpandedPage

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dashboard ML & Analytics - Final")
        self.resize(1150, 800)
        self.setStyleSheet(f"background-color: {BG_MAIN};")
        
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # --- 1. INSTANCIANDO OS 4 CARDS ---
        self.card_input = InputCard(
            on_expand_callback=lambda: self.stack.setCurrentIndex(1),
            on_classify_callback=self.classify_point # Conecta o botão ao Main
        )     
        self.card_charts = ChartsCard(lambda: self.stack.setCurrentIndex(2))
        
        # Passando o callback de treinamento para o Algoritmo
        self.card_algo = AlgorithmCard(
            on_expand_callback=lambda: self.stack.setCurrentIndex(3),
            on_train_callback=self.train_model
        )
        self.card_accuracy = AccuracyCard(lambda: self.stack.setCurrentIndex(4))

        # --- 2. LAYOUT DO DASHBOARD ---
        dash_widget = QWidget()
        dash_layout = QVBoxLayout(dash_widget)
        dash_layout.setContentsMargins(20, 20, 20, 20)

        lbl_title = QLabel("Dashboard ML & Analytics")
        lbl_title.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        lbl_title.setStyleSheet(f"color: {ACCENT_COLOR}; margin-bottom: 15px;")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        dash_layout.addWidget(lbl_title)
        
        s_horiz1 = QSplitter(Qt.Orientation.Horizontal)
        s_horiz1.addWidget(self.card_input)
        s_horiz1.addWidget(self.card_charts)

        s_horiz2 = QSplitter(Qt.Orientation.Horizontal)
        s_horiz2.addWidget(self.card_algo)
        s_horiz2.addWidget(self.card_accuracy)

        s_vert = QSplitter(Qt.Orientation.Vertical)
        s_vert.addWidget(s_horiz1)
        s_vert.addWidget(s_horiz2)
        
        dash_layout.addWidget(s_vert, stretch=1)
        self.stack.addWidget(dash_widget)

        # --- 3. PÁGINAS EXPANDIDAS ---
        self.page_input = InputExpandedPage(self.card_input.update_preview_text, lambda: self.stack.setCurrentIndex(0))
        self.page_charts = ChartsExpandedPage(self.card_charts.preview_plot, lambda: self.stack.setCurrentIndex(0))
        
        # INJETAMOS O self.train_model AQUI NO FINAL:
        self.page_algo = AlgorithmExpandedPage(self.card_algo.update_preview_text, lambda: self.stack.setCurrentIndex(0), self.train_model)
        
        self.page_accuracy = AccuracyExpandedPage(lambda: self.stack.setCurrentIndex(0))

        # --- 4. ADICIONANDO PÁGINAS AO STACK ---
        self.stack.addWidget(self.page_input)    # Index 1
        self.stack.addWidget(self.page_charts)   # Index 2
        self.stack.addWidget(self.page_algo)     # Index 3
        self.stack.addWidget(self.page_accuracy) # Index 4
    
    def train_model(self):
        # Limpa logs antigos e avisa que começou
        self.page_algo.clear_logs()
        self.page_algo.append_log("[SISTEMA] Iniciando preparação dos dados...")
        
        try:
            dataset, class_data, conjunto_data = self.page_input.get_full_dataset()
            
            if dataset and class_data:
                self.page_algo.append_log(f"[OK] Dataset carregado. Total de pontos: {len(class_data)}")
                self.page_algo.append_log("[SISTEMA] Atualizando gráficos de visualização...")
                
                self.page_charts.clear_charts()
                self.page_charts.set_dataset(dataset, class_data, conjunto_data)
                self.page_charts.checkboxes[0].setChecked(True) 
                self.page_charts.checkboxes[1].setChecked(True) 
                self.page_charts.plot_custom_chart()
                
            self.page_algo.append_log("[OK] Gráficos gerados com sucesso.")
            self.page_algo.append_log("-----------------------------------------")
            self.page_algo.append_log("[SISTEMA] TREINAMENTO CONCLUÍDO COM SUCESSO!")
            
            msg = QMessageBox(self)
            msg.setWindowTitle("Treinamento Concluído")
            msg.setText("Modelo treinado com sucesso!")
            
        except Exception as e:
            self.page_algo.append_log(f"[ERRO CRÍTICO] {e}")
            msg = QMessageBox(self)
            msg.setWindowTitle("Treinamento Interrompido")
            msg.setText(f"{e}")
            
        msg.setStyleSheet("""
                QMessageBox { background-color: #1E1E1E; color: #FFFFFF; }
                QLabel { color: #FFFFFF; font-size: 14px; }
                QPushButton { background-color: #00E5FF; color: #000000; padding: 5px 15px; font-weight: bold; border-radius: 3px; }
        """)
        msg.exec()
        
    def classify_point(self):
        # 1. Pega os valores atuais (Sepal e Petal)
        current_inputs = self.page_input.get_current_inputs()
        
        # 2. Lógica Placeholder (Aqui entrará o Algoritmo no futuro)
        predicted_class = "Iris-setosa" 
        current_inputs["class"] = predicted_class
        
        # 3. Envia o ponto para a página de gráficos plotar
        self.page_charts.set_classified_point(current_inputs)
        
        # 4. Gera o Pop-up visual
        msg = QMessageBox(self)
        msg.setWindowTitle("Resultado da Classificação")
        msg.setText("O algoritmo classificou este ponto como:")
        msg.setInformativeText(f"<b>{predicted_class}</b> (Placeholder)")
        
        msg.setStyleSheet("""
            QMessageBox { background-color: #1E1E1E; color: #FFFFFF; }
            QLabel { color: #FFFFFF; font-size: 14px; }
            QPushButton { background-color: #00E5FF; color: #000000; padding: 5px 15px; font-weight: bold; border-radius: 3px; }
        """)
        msg.exec()
            
if __name__ == "__main__":
    app = QApplication(sys.argv)
    setup_pyqtgraph() 
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
import sys
import re
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QSplitter, QStackedWidget, QLabel, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QAction

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
        
        self.is_light_mode = False 
        
        # --- MENU SUPERIOR ---
        self.create_menu()
        
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # --- INSTANCIANDO OS 4 CARDS ---
        self.card_input = InputCard(lambda: self.stack.setCurrentIndex(1))       
        self.card_charts = ChartsCard(lambda: self.stack.setCurrentIndex(2))
        self.card_algo = AlgorithmCard(
            on_expand_callback=lambda: self.stack.setCurrentIndex(3),
            on_train_callback=self.train_model
        )
        self.card_accuracy = AccuracyCard(lambda: self.stack.setCurrentIndex(4))

        # --- LAYOUT DO DASHBOARD ---
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

        # --- PÁGINAS EXPANDIDAS ---
        self.page_input = InputExpandedPage(self.card_input.update_preview_text, lambda: self.stack.setCurrentIndex(0))
        self.page_charts = ChartsExpandedPage(self.card_charts.preview_plot, lambda: self.stack.setCurrentIndex(0))
        self.page_algo = AlgorithmExpandedPage(self.card_algo.update_preview_text, lambda: self.stack.setCurrentIndex(0))
        self.page_accuracy = AccuracyExpandedPage(lambda: self.stack.setCurrentIndex(0))

        self.stack.addWidget(self.page_input)    # Index 1
        self.stack.addWidget(self.page_charts)   # Index 2
        self.stack.addWidget(self.page_algo)     # Index 3
        self.stack.addWidget(self.page_accuracy) # Index 4

    def create_menu(self):
        menubar = self.menuBar()
        menubar.setStyleSheet("background-color: transparent; font-weight: bold;")
        theme_menu = menubar.addMenu("Visual")

        action_dark = QAction("🌙 Tema Escuro", self)
        action_dark.triggered.connect(lambda: self.switch_theme(is_light=False))
        theme_menu.addAction(action_dark)

        action_light = QAction("☀️ Tema Claro", self)
        action_light.triggered.connect(lambda: self.switch_theme(is_light=True))
        theme_menu.addAction(action_light)

    def switch_theme(self, is_light):
        if self.is_light_mode == is_light:
            return 
            
        self.is_light_mode = is_light

        new_bg = "#FFFFFF" if is_light else "#1E1E1E"
        new_fg = "#333333" if is_light else "#FFFFFF"
        pg.setConfigOption('background', new_bg)
        pg.setConfigOption('foreground', new_fg)

        # Mapeamento exato de cores
        theme_map = {
            "#121212": "#F0F2F5",      # Fundo Main
            "#1E1E1E": "#FFFFFF",      # Fundo Card
            "#1E1E1EE6": "#FFFFFFE6",  # Legenda do Gráfico
            "#FFFFFF": "#121212",      # Texto Branco -> Preto
            "#A0A0A0": "#666666",      # Texto Secundário
            "#00E5FF": "#0078D4",      # Ciano -> Azul
            "#000000": "#FFFFFF",      # Texto Botão
            "#FFEA00": "#D44200",      # Amarelo -> Laranja
            "#333333": "#CCCCCC",      # Bordas longas
            "#444444": "#DDDDDD",      # Linhas longas
            "#2b2b2b": "#F9F9F9",      # Fundo da Tabela
            "#1a1a1a": "#EAEAEA",      # Cabeçalho da Tabela
            "#00B3CC": "#005A9E",      # Hover Button
            "#333": "#CCC",            # Bordas curtas
            "#444": "#DDD",            # Linhas curtas
        }

        mapping = theme_map if is_light else {v: k for k, v in theme_map.items()}

        for widget in QApplication.allWidgets():
            if isinstance(widget, pg.PlotWidget):
                widget.setBackground(new_bg)

            style = widget.styleSheet()
            if style:
                # O segredo está aqui: A Regex garante que a cor só muda se terminar nela mesma
                for src, tgt in mapping.items():
                    pattern = re.compile(src + r'(?![0-9A-Fa-f])', re.IGNORECASE)
                    style = pattern.sub(f"__TMP_{tgt}__", style)
                
                for src, tgt in mapping.items():
                    style = style.replace(f"__TMP_{tgt}__", tgt)
                    
                widget.setStyleSheet(style)

    def train_model(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("Treinamento Concluído")
        msg.setText("Modelo treinado com sucesso!")
        
        bg_color = "#FFFFFF" if self.is_light_mode else "#1E1E1E"
        txt_color = "#121212" if self.is_light_mode else "#FFFFFF"
        btn_color = "#0078D4" if self.is_light_mode else "#00E5FF"
        btn_txt = "#FFFFFF" if self.is_light_mode else "#000000"
        
        msg.setStyleSheet(f"""
            QMessageBox {{ background-color: {bg_color}; color: {txt_color}; }}
            QLabel {{ color: {txt_color}; font-size: 14px; }}
            QPushButton {{ background-color: {btn_color}; color: {btn_txt}; padding: 5px 15px; font-weight: bold; border-radius: 3px; }}
        """)
        msg.exec()

        dataset, class_data = self.page_input.get_full_dataset()
        
        if dataset and class_data:
            self.page_charts.clear_charts()
            self.page_charts.set_dataset(dataset, class_data)
            
            self.page_charts.checkboxes[0].setChecked(True) 
            self.page_charts.checkboxes[1].setChecked(True) 
            self.page_charts.plot_custom_chart()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    setup_pyqtgraph() 
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
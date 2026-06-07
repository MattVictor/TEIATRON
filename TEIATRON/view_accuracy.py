# view_accuracy.py
from PyQt6.QtWidgets import (
    QLabel, QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, 
    QAbstractItemView, QHeaderView, QGridLayout, QGroupBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from config import TEXT_PRIMARY, TEXT_SECONDARY, BG_CARD, ACCENT_COLOR, WARNING_COLOR
from base_components import BaseCard, BaseExpandedPage

class AccuracyCard(BaseCard):
    """Card principal que mostra um resumo das métricas de acurácia."""
    def __init__(self, on_expand_callback):
        super().__init__("Acurácia", on_expand_callback)
        
        # Preview com alguns placeholders
        preview_text = (
            "Acerto Geral: 92.5%\n"
            "Kappa: 0.85\n\n"
            "Clique em '+' para ver a Matriz de Confusão e demais métricas."
        )
        self.preview_label = QLabel(preview_text)
        self.preview_label.setWordWrap(True)
        self.preview_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 14px; line-height: 1.5;")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.add_preview_content(self.preview_label)

    def update_preview(self, text):
        self.preview_label.setText(text)

class AccuracyExpandedPage(BaseExpandedPage):
    """Página expandida contendo a Matriz de Confusão e a lista de coeficientes."""
    def __init__(self, on_back_callback):
        super().__init__("Avaliação do Modelo", on_back_callback)
        
        container = QWidget()
        layout = QVBoxLayout(container)

        # --- 1. TÍTULO E TABELA DA MATRIZ DE CONFUSÃO ---
        lbl_matrix = QLabel("Matriz de Confusão:")
        lbl_matrix.setStyleSheet(f"color: {ACCENT_COLOR}; font-size: 16px; font-weight: bold;")
        layout.addWidget(lbl_matrix)

        self.table = QTableWidget(3, 3) # Exemplo com 3 classes (adapte conforme precisar)
        self.table.setHorizontalHeaderLabels(["Predito A", "Predito B", "Predito C"])
        self.table.setVerticalHeaderLabels(["Real A", "Real B", "Real C"])
        
        # Ativando os scrolls vertical e horizontal
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        
        self.table.setStyleSheet(f"""
            QTableWidget {{ background-color: #2b2b2b; color: {TEXT_PRIMARY}; gridline-color: #555; border: 1px solid #444; }}
            QHeaderView::section {{ background-color: #1a1a1a; color: {ACCENT_COLOR}; font-weight: bold; padding: 4px; border: 1px solid #444; }}
        """)
        
        # Preenchendo a tabela com placeholders
        placeholder_data = [[45, 2, 0], [3, 50, 1], [0, 4, 40]]
        for i in range(3):
            for j in range(3):
                item = QTableWidgetItem(str(placeholder_data[i][j]))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(i, j, item)
                
        # Força o ajuste para que a rolagem funcione corretamente se houver muitas colunas
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        layout.addWidget(self.table, stretch=1)

        # --- 2. PAINEL DE MÉTRICAS (LABELS) ---
        group_box = QGroupBox("Métricas de Desempenho")
        group_box.setStyleSheet(f"""
            QGroupBox {{ color: {ACCENT_COLOR}; font-size: 16px; font-weight: bold; border: 1px solid #444; margin-top: 15px; padding-top: 20px; }}
        """)
        grid = QGridLayout(group_box)
        grid.setVerticalSpacing(15)

        # Lista de métricas solicitadas com valores placeholders
        metrics = [
            ("i) Acerto geral do classificador:", "93.1%"),
            ("ii) Acurácia do produtor (Média):", "92.8%"),
            ("iii) Acurácia do usuário (Média):", "94.0%"),
            ("iv) Coeficiente Kappa:", "0.88"),
            ("v) Coeficiente Tau:", "0.85"),
            ("vi) Coeficiente de Matthews (MCC) para 2 classes:", "0.89"),
            ("vii) Fb Score:", "0.91")
        ]

        # Desenhando as labels dinamicamente na grade
        for row, (name, value) in enumerate(metrics):
            lbl_name = QLabel(name)
            lbl_name.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px;")
            
            lbl_value = QLabel(value)
            lbl_value.setStyleSheet(f"color: {WARNING_COLOR}; font-size: 14px; font-weight: bold;")
            
            grid.addWidget(lbl_name, row, 0)
            grid.addWidget(lbl_value, row, 1)

        layout.addWidget(group_box)
        self.add_main_content(container)
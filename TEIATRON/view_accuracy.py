# view_accuracy.py
from PyQt6.QtWidgets import (
    QLabel, QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, 
    QAbstractItemView, QHeaderView, QGridLayout, QGroupBox, QComboBox, QHBoxLayout
)
from PyQt6.QtCore import Qt
from config import TEXT_PRIMARY, TEXT_SECONDARY, ACCENT_COLOR, WARNING_COLOR
from base_components import BaseCard, BaseExpandedPage

class AccuracyCard(BaseCard):
    def __init__(self, on_expand_callback):
        super().__init__("Acurácia", on_expand_callback)
        self.preview_label = QLabel("Aguardando treinamento para exibir métricas...")
        self.preview_label.setWordWrap(True)
        self.preview_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 14px; line-height: 1.5;")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.add_preview_content(self.preview_label)

    def update_preview(self, text):
        self.preview_label.setText(text)

class AccuracyExpandedPage(BaseExpandedPage):
    def __init__(self, on_back_callback, on_mode_changed_callback):
        super().__init__("Avaliação do Modelo", on_back_callback)
        
        container = QWidget()
        layout = QVBoxLayout(container)

        # --- BARRA DE CONTROLE ---
        control_layout = QHBoxLayout()
        lbl_control = QLabel("Base de Validação:")
        lbl_control.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 16px; font-weight: bold;")
        
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Apenas Teste", "Apenas Treino", "Todo o Dataset"])
        self.combo_mode.setStyleSheet(f"QComboBox {{ background-color: #333; color: {TEXT_PRIMARY}; padding: 5px; font-size: 14px; border-radius: 4px; }}")
        self.combo_mode.currentTextChanged.connect(on_mode_changed_callback)
        
        control_layout.addWidget(lbl_control)
        control_layout.addWidget(self.combo_mode)
        control_layout.addStretch()
        layout.addLayout(control_layout)

        # --- MATRIZ DE CONFUSÃO ---
        lbl_matrix = QLabel("Matriz de Confusão:")
        lbl_matrix.setStyleSheet(f"color: {ACCENT_COLOR}; font-size: 16px; font-weight: bold; margin-top: 10px;")
        layout.addWidget(lbl_matrix)

        self.table = QTableWidget(1, 1)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setStyleSheet(f"""
            QTableWidget {{ background-color: #2b2b2b; color: {TEXT_PRIMARY}; gridline-color: #555; border: 1px solid #444; }}
            QHeaderView::section {{ background-color: #1a1a1a; color: {ACCENT_COLOR}; font-weight: bold; padding: 4px; border: 1px solid #444; }}
        """)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, stretch=1)

        # --- MÉTRICAS ---
        group_box = QGroupBox("Métricas de Desempenho")
        group_box.setStyleSheet(f"QGroupBox {{ color: {ACCENT_COLOR}; font-size: 16px; font-weight: bold; border: 1px solid #444; margin-top: 15px; padding-top: 20px; }}")
        grid = QGridLayout(group_box)
        grid.setVerticalSpacing(15)

        self.labels_metricas = {}
        metrics_names = [
            "Acerto Geral:", "Acurácia Produtor:", "Acurácia Usuário:",
            "Kappa:", "Tau:", "Matthews (Apenas Binário):", "F1 Score:", "F2 Score:"
        ]

        for row, name in enumerate(metrics_names):
            lbl_name = QLabel(name)
            lbl_name.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 20px;")
            lbl_value = QLabel("0.0000")
            lbl_value.setStyleSheet(f"color: {WARNING_COLOR}; font-size: 20px; font-weight: bold; font-family: 'Consolas';")
            
            grid.addWidget(lbl_name, row, 0)
            grid.addWidget(lbl_value, row, 1)
            self.labels_metricas[name] = lbl_value

        layout.addWidget(group_box)
        self.add_main_content(container)

    def update_metrics(self, matriz, classes_names, metrics_obj):
        n = len(matriz)
        self.table.setRowCount(n + 1)
        self.table.setColumnCount(n + 1)
        
        headers = classes_names + ["Total"]
        self.table.setHorizontalHeaderLabels([f"Real:\n{h}" for h in headers])
        self.table.setVerticalHeaderLabels([f"Pred:\n{h}" for h in headers])

        # Preenchimento dinâmico e cálculo de somatórios
        for i in range(n):
            soma_linha = sum(matriz[i])
            self.table.setItem(i, n, QTableWidgetItem(str(soma_linha)))
            for j in range(n):
                item = QTableWidgetItem(str(matriz[i][j]))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(i, j, item)
        
        for j in range(n):
            soma_coluna = sum(matriz[i][j] for i in range(n))
            self.table.setItem(n, j, QTableWidgetItem(str(soma_coluna)))
        
        total_geral = sum(sum(linha) for linha in matriz)
        self.table.setItem(n, n, QTableWidgetItem(str(total_geral)))

        for i in range(n + 1):
            for j in range(n + 1):
                if self.table.item(i, j):
                    self.table.item(i, j).setTextAlignment(Qt.AlignmentFlag.AlignCenter)

        # Atualizando Labels
        self.labels_metricas["Acerto Geral:"].setText(f"{metrics_obj.acerto_geral():.4f}")
        self.labels_metricas["Acurácia Produtor:"].setText(f"{metrics_obj.acuracia_produtor():.4f}")
        self.labels_metricas["Acurácia Usuário:"].setText(f"{metrics_obj.acuracia_usuario():.4f}")
        self.labels_metricas["Kappa:"].setText(f"{metrics_obj.coeficiente_kappa():.4f}")
        self.labels_metricas["Tau:"].setText(f"{metrics_obj.coeficiente_tau():.4f}")
        self.labels_metricas["Matthews (Apenas Binário):"].setText(f"{metrics_obj.coeficiente_matthews():.4f}")
        self.labels_metricas["F1 Score:"].setText(f"{metrics_obj.fb_score(1):.4f}")
        self.labels_metricas["F2 Score:"].setText(f"{metrics_obj.fb_score(2):.4f}")
# view_accuracy.py
from PyQt6.QtWidgets import (
    QLabel, QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, 
    QAbstractItemView, QHeaderView, QComboBox, QHBoxLayout, QSplitter
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
    def __init__(self, on_back_callback, on_update_callback):
        super().__init__("Avaliação e Comparação de Modelos", on_back_callback)
        
        container = QWidget()
        layout = QVBoxLayout(container)

        # --- BARRA DE CONTROLE ---
        control_layout = QHBoxLayout()
        
        lbl_control = QLabel("Base:")
        lbl_control.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 14px; font-weight: bold;")
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Apenas Teste", "Apenas Treino", "Todo o Dataset"])
        self.combo_mode.setStyleSheet(f"QComboBox {{ background-color: #333; color: {TEXT_PRIMARY}; padding: 5px; border-radius: 4px; }}")
        self.combo_mode.currentTextChanged.connect(on_update_callback)
        
        lbl_compare = QLabel("Comparar com:")
        lbl_compare.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 14px; font-weight: bold; margin-left: 20px;")
        self.combo_compare = QComboBox()
        self.combo_compare.addItem("Nenhum")
        self.combo_compare.setStyleSheet(f"QComboBox {{ background-color: #333; color: {TEXT_PRIMARY}; padding: 5px; border-radius: 4px; }}")
        self.combo_compare.currentTextChanged.connect(on_update_callback)
        
        control_layout.addWidget(lbl_control)
        control_layout.addWidget(self.combo_mode)
        control_layout.addWidget(lbl_compare)
        control_layout.addWidget(self.combo_compare)
        control_layout.addStretch()
        layout.addLayout(control_layout)

        # --- SPLITTER VERTICAL ---
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setStyleSheet("""
            QSplitter::handle { background-color: #444444; height: 4px; border-radius: 2px; margin: 5px 0px; }
            QSplitter::handle:hover { background-color: #00E5FF; }
        """)

        # --- PAINEL SUPERIOR: MATRIZ DE CONFUSÃO (MODELO ATUAL) ---
        pane_matrix = QWidget()
        layout_matrix = QVBoxLayout(pane_matrix)
        layout_matrix.setContentsMargins(0, 10, 0, 0)

        lbl_matrix = QLabel("Matriz de Confusão (Modelo Atual):")
        lbl_matrix.setStyleSheet(f"color: {ACCENT_COLOR}; font-size: 16px; font-weight: bold;")
        layout_matrix.addWidget(lbl_matrix)

        self.table_matrix = QTableWidget(1, 1)
        self.table_matrix.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_matrix.setStyleSheet(f"""
            QTableWidget {{ background-color: #2b2b2b; color: {TEXT_PRIMARY}; gridline-color: #555; border: 1px solid #444; }}
            QHeaderView::section {{ background-color: #1a1a1a; color: {ACCENT_COLOR}; font-weight: bold; padding: 4px; border: 1px solid #444; }}
        """)
        self.table_matrix.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout_matrix.addWidget(self.table_matrix, stretch=1)
        splitter.addWidget(pane_matrix)

        # --- PAINEL INFERIOR: TABELA DE MÉTRICAS COMPARATIVAS ---
        pane_metrics = QWidget()
        layout_metrics = QVBoxLayout(pane_metrics)
        layout_metrics.setContentsMargins(0, 10, 0, 0)
        
        lbl_metrics = QLabel("Quadro Comparativo de Desempenho:")
        lbl_metrics.setStyleSheet(f"color: {ACCENT_COLOR}; font-size: 16px; font-weight: bold;")
        layout_metrics.addWidget(lbl_metrics)

        self.metrics_names = [
            "Acerto Geral", "Acurácia Produtor", "Acurácia Usuário",
            "Kappa", "Tau", "Matthews (Apenas Binário)", "F1 Score", "F2 Score"
        ]
        
        self.table_metrics = QTableWidget(len(self.metrics_names), 4)
        self.table_metrics.setHorizontalHeaderLabels(["Métrica", "Modelo Atual", "Modelo Comparado", "Ganho / Perda"])
        self.table_metrics.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        
        # Cor amarela solicitada para o texto da tabela de métricas
        self.table_metrics.setStyleSheet(f"""
            QTableWidget {{ background-color: #1E1E1E; color: #FFFF00; gridline-color: #555; border: 1px solid #444; font-family: 'Consolas'; font-size: 13px; font-weight: bold; }}
            QHeaderView::section {{ background-color: #1a1a1a; color: {ACCENT_COLOR}; font-weight: bold; padding: 4px; border: 1px solid #444; }}
        """)
        self.table_metrics.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        for i, nome in enumerate(self.metrics_names):
            item = QTableWidgetItem(nome)
            item.setForeground(Qt.GlobalColor.white) # A primeira coluna (nome da métrica) fica branca para contraste
            self.table_metrics.setItem(i, 0, item)
            
        layout_metrics.addWidget(self.table_metrics, stretch=1)
        
        self.lbl_mcnemar = QLabel("")
        self.lbl_mcnemar.setStyleSheet(f"color: {WARNING_COLOR}; font-size: 15px; font-weight: bold; margin-top: 8px;")
        self.lbl_mcnemar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout_metrics.addWidget(self.lbl_mcnemar)
        
        splitter.addWidget(pane_metrics)
        
        splitter.setSizes([300, 300]) 
        layout.addWidget(splitter, stretch=1)
        self.add_main_content(container)

    def update_metrics(self, matriz, classes_names, metrics_current, metrics_compare=None, mcnemar_text=""):
        # 1. Atualizar Matriz de Confusão (Mesma lógica de antes)
        n = len(matriz)
        self.table_matrix.setRowCount(n + 1)
        self.table_matrix.setColumnCount(n + 1)
        headers = classes_names + ["Total"]
        self.table_matrix.setHorizontalHeaderLabels([f"Real:\n{h}" for h in headers])
        self.table_matrix.setVerticalHeaderLabels([f"Pred: {h}" for h in headers])
        
        # Correção do corte do texto no cabeçalho vertical
        self.table_matrix.verticalHeader().setMinimumWidth(150)
        self.table_matrix.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        for i in range(n):
            soma_linha = sum(matriz[i])
            self.table_matrix.setItem(i, n, QTableWidgetItem(str(soma_linha)))
            for j in range(n):
                self.table_matrix.setItem(i, j, QTableWidgetItem(str(matriz[i][j])))
        
        for j in range(n):
            self.table_matrix.setItem(n, j, QTableWidgetItem(str(sum(matriz[i][j] for i in range(n)))))
        
        self.table_matrix.setItem(n, n, QTableWidgetItem(str(sum(sum(linha) for linha in matriz))))
        
        for i in range(n + 1):
            for j in range(n + 1):
                if self.table_matrix.item(i, j):
                    self.table_matrix.item(i, j).setTextAlignment(Qt.AlignmentFlag.AlignCenter)

        # 2. Atualizar Tabela de Métricas Comparativa
        def extrair_valores(m):
            if not m: return [0]*8
            return [
                m.acerto_geral(), m.acuracia_produtor(), m.acuracia_usuario(),
                m.coeficiente_kappa(), m.coeficiente_tau(), m.coeficiente_matthews(),
                m.fb_score(1), m.fb_score(2)
            ]

        vals_current = extrair_valores(metrics_current)
        vals_compare = extrair_valores(metrics_compare) if metrics_compare else None

        for i in range(len(self.metrics_names)):
            # Atual
            item_atual = QTableWidgetItem(f"{vals_current[i]:.4f}")
            item_atual.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table_metrics.setItem(i, 1, item_atual)
            
            # Comparado e Ganho/Perda
            if vals_compare:
                item_comp = QTableWidgetItem(f"{vals_compare[i]:.4f}")
                
                diff = vals_current[i] - vals_compare[i]
                sinal = "+" if diff > 0 else ""
                item_diff = QTableWidgetItem(f"{sinal}{diff:.4f}")
                
                # Pinta verde se ganhou, vermelho se perdeu, cinza se igual
                if diff > 0:
                    item_diff.setForeground(Qt.GlobalColor.green)
                elif diff < 0:
                    item_diff.setForeground(Qt.GlobalColor.red)
                else:
                    item_diff.setForeground(Qt.GlobalColor.gray)
                    
            else:
                item_comp = QTableWidgetItem("-")
                item_diff = QTableWidgetItem("-")
                item_diff.setForeground(Qt.GlobalColor.gray)
                
            item_comp.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_diff.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table_metrics.setItem(i, 2, item_comp)
            self.table_metrics.setItem(i, 3, item_diff)
            
        self.lbl_mcnemar.setText(mcnemar_text)
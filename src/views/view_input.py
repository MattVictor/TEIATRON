# view_input.py
import os
from PyQt6.QtWidgets import (
    QCheckBox, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QDoubleSpinBox, 
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, 
    QFileDialog, QGridLayout, QMessageBox, QSpinBox
)
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QColor
from views.config import TEXT_PRIMARY, ACCENT_COLOR, ACCENT_TEXT, WARNING_COLOR
from views.base_components import BaseCard, BaseExpandedPage

class InputCard(BaseCard):
    # Alteramos o init para receber o on_classify_callback
    def __init__(self, on_expand_callback, on_classify_callback):
        super().__init__("Entrada (Iris)", on_expand_callback)
        
        self.preview_label = QLabel("Nenhum dado carregado.")
        self.preview_label.setWordWrap(True)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter) # Mantendo centralizado!
        self.preview_label.setStyleSheet(f"color: {WARNING_COLOR}; font-size: 15px; font-weight: bold; font-family: 'Consolas';")
        self.add_preview_content(self.preview_label)

        self.btn_classify = QPushButton("Classificar")
        self.btn_classify.setStyleSheet(f"""
            QPushButton {{ 
                background-color: {ACCENT_COLOR}; 
                color: {ACCENT_TEXT}; 
                font-weight: bold; 
                font-size: 14px; 
                padding: 8px; 
                border-radius: 5px; 
            }}
            QPushButton:hover {{ background-color: #00B3CC; }}
        """)
        # Agora o botão chama a função injetada pelo main.py
        self.btn_classify.clicked.connect(on_classify_callback)
        self.layout.addWidget(self.btn_classify)

    def update_preview_text(self, text):
        self.preview_label.setText(text)

    def show_classification_popup(self):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Resultado da Classificação")
        msg_box.setText("O algoritmo classificou este ponto como:")
        msg_box.setInformativeText("<b>Iris-setosa</b> (Placeholder)")
        msg_box.setStyleSheet("""
            QMessageBox { background-color: #1E1E1E; color: #FFFFFF; }
            QLabel { color: #FFFFFF; font-size: 14px; }
            QPushButton { background-color: #00E5FF; color: #000000; padding: 5px 15px; font-weight: bold; border-radius: 3px; }
        """)
        msg_box.exec()

class InputExpandedPage(BaseExpandedPage):
    def __init__(self, update_card_callback, on_back_callback, on_import_callback, on_split_callback):
        super().__init__("Entrada de Dados", on_back_callback)
        self.update_card_callback = update_card_callback
        self.on_import_callback = on_import_callback
        self.on_split_callback = on_split_callback
        self.current_class = None
        
        container = QWidget()
        layout = QVBoxLayout(container)

        # --- PAINEL DE INPUTS MANUAIS ---
        input_panel = QWidget()
        grid = QGridLayout(input_panel)
        self.inputs = {}
        labels = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
        
        for i, name in enumerate(labels):
            lbl = QLabel(name)
            lbl.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px;")
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 20.0)
            spin.setSingleStep(0.1)
            spin.setDecimals(1)
            spin.setStyleSheet(f"background-color: #333; color: {TEXT_PRIMARY}; padding: 6px; font-size: 14px;")
            spin.valueChanged.connect(self.on_manual_input_change)
            
            self.inputs[name] = spin
            row, col = i // 2, (i % 2) * 2
            grid.addWidget(lbl, row, col)
            grid.addWidget(spin, row, col + 1)
            
        layout.addWidget(input_panel)

        # --- BOTÃO IMPORTAR ---
        btn_import = QPushButton("📂 Importar Dataset (.csv)")
        btn_import.setStyleSheet(f"""
            QPushButton {{ background-color: #4CAF50; color: white; font-weight: bold; font-size: 14px; padding: 10px; border-radius: 5px; margin-top: 10px; }}
            QPushButton:hover {{ background-color: #45a049; }}
        """)
        btn_import.clicked.connect(self.load_csv)
        layout.addWidget(btn_import)

        # --- TABELA DE DADOS ---
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setStyleSheet(f"""
            QTableWidget {{ background-color: #2b2b2b; color: {TEXT_PRIMARY}; gridline-color: #555; border: 1px solid #444; }}
            QHeaderView::section {{ background-color: #1a1a1a; color: {ACCENT_COLOR}; font-weight: bold; padding: 4px; border: 1px solid #444; }}
        """)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.cellDoubleClicked.connect(self.on_table_double_click)
        layout.addWidget(self.table)

        # --- NOVOS CONTROLES: ALEATORIZAÇÃO E DIVISÃO (TRAIN/TEST) ---
        split_panel = QWidget()
        split_layout = QHBoxLayout(split_panel)
        split_layout.setContentsMargins(0, 10, 0, 0)

        lbl_pct = QLabel("Porcentagem de Treino (%):")
        lbl_pct.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px; font-weight: bold;")
        
        self.spin_pct = QSpinBox()
        self.spin_pct.setRange(1, 99)
        self.spin_pct.setValue(70) # Padrão 80% Treino / 20% Teste
        self.spin_pct.setStyleSheet(f"background-color: #333; color: {TEXT_PRIMARY}; padding: 6px; font-size: 14px; font-weight: bold;")

        btn_stratified = QPushButton("Aleatorizar por Classe")
        btn_stratified.setStyleSheet(f"QPushButton {{ background-color: {ACCENT_COLOR}; color: {ACCENT_TEXT}; font-weight: bold; padding: 8px; border-radius: 4px; }} QPushButton:hover {{ background-color: #00B3CC; }}")
        btn_stratified.clicked.connect(lambda: self.apply_split(stratified=True))

        btn_global = QPushButton("Aleatorizar Tudo")
        btn_global.setStyleSheet(f"QPushButton {{ background-color: #FF9800; color: {ACCENT_TEXT}; font-weight: bold; padding: 8px; border-radius: 4px; }} QPushButton:hover {{ background-color: #F57C00; }}")
        btn_global.clicked.connect(lambda: self.apply_split(stratified=False))

        split_layout.addWidget(lbl_pct)
        split_layout.addWidget(self.spin_pct)
        split_layout.addWidget(btn_stratified)
        split_layout.addWidget(btn_global)
        split_layout.addStretch()

        layout.addWidget(split_panel)

        # --- SELEÇÃO DE FEATURES ---
        feature_panel = QWidget()
        feature_layout = QHBoxLayout(feature_panel)
        feature_layout.setContentsMargins(0, 10, 0, 0)
        
        lbl_feat = QLabel("Usar Características:")
        lbl_feat.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px; font-weight: bold;")
        feature_layout.addWidget(lbl_feat)
        
        self.feature_checkboxes = {}
        for name in labels:
            chk = QCheckBox(name)
            chk.setChecked(True)
            chk.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px;")
            self.feature_checkboxes[name] = chk
            feature_layout.addWidget(chk)
            
        feature_layout.addStretch()
        layout.addWidget(feature_panel)

        self.add_main_content(container)
        self.sync_to_card()

    def get_selected_features(self):
        return [name for name, chk in self.feature_checkboxes.items() if chk.isChecked()]

    def get_current_inputs(self):
        """Coleta os 4 valores que estão atualmente digitados nas caixas de input."""
        return {
            "Sepal Length": self.inputs["Sepal Length"].value(),
            "Sepal Width": self.inputs["Sepal Width"].value(),
            "Petal Length": self.inputs["Petal Length"].value(),
            "Petal Width": self.inputs["Petal Width"].value()
        }

    # ==========================================
    # LÓGICA DE DADOS
    # ==========================================
    def on_manual_input_change(self):
        if not hasattr(self, '_ignore_manual'):
            self.current_class = None
        self.sync_to_card()

    def load_csv(self):
        settings = QSettings("TEIATRON", "MachineLearningApp")
        last_path = settings.value("last_dataset_path")

        if last_path and os.path.exists(last_path):
            reply = QMessageBox.question(
                self, 
                "Reutilizar Dataset", 
                f"Deseja utilizar o último dataset carregado?\n\n{last_path}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._do_load_csv(last_path)
                return

        file_path, _ = QFileDialog.getOpenFileName(self, "Importar CSV", "", "Arquivos CSV (*.csv)")
        if not file_path:
            return
            
        settings.setValue("last_dataset_path", file_path)
        self._do_load_csv(file_path)

    def _do_load_csv(self, file_path):
        try:
            headers, data = self.on_import_callback(file_path)
            
            # Aleatorização Automática solicitada pelo usuário
            train_ratio = self.spin_pct.value() / 100.0
            headers, data = self.on_split_callback(True, train_ratio) # True = Stratified
            
            self.render_table(headers, data)
        except Exception as e:
            QMessageBox.critical(self, "Erro de Importação", str(e))

    def apply_split(self, stratified):
        train_ratio = self.spin_pct.value() / 100.0
        headers, data = self.on_split_callback(stratified, train_ratio)
        self.render_table(headers, data)

    def render_table(self, headers, all_data):
        """Apenas renderiza a matriz de dados na tabela (Visão Passiva)."""
        if not all_data:
            return
            
        self.table.clearContents()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(all_data))

        for i, row in enumerate(all_data):
            for j, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                
                # Destacar a coluna "Conjunto" com cores para visualização rápida
                if j == len(headers) - 1:
                    if val == "Treino":
                        item.setForeground(QColor("#00E5FF")) # Ciano brilhante
                    else:
                        item.setForeground(QColor("#FFEA00")) # Amarelo brilhante
                        
                self.table.setItem(i, j, item)

    def on_table_double_click(self, row, column):
        col_count = self.table.columnCount()
        self._ignore_manual = True

        try:
            keys = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
            for i, key in enumerate(keys):
                if i < col_count:
                    raw_text = self.table.item(row, i).text()
                    val = float(raw_text.replace(',', '.'))
                    self.inputs[key].setValue(val)

            # Para capturar a classe, consideramos a 5ª coluna ou a penúltima caso 'Conjunto' tenha sido criado
            if col_count >= 6:
                self.current_class = self.table.item(row, 4).text()
            elif col_count >= 5:
                self.current_class = self.table.item(row, 4).text()
            else:
                self.current_class = None
                
        except ValueError:
            pass
        finally:
            del self._ignore_manual
            self.sync_to_card()

    def sync_to_card(self):
        sl = self.inputs["Sepal Length"].value()
        sw = self.inputs["Sepal Width"].value()
        pl = self.inputs["Petal Length"].value()
        pw = self.inputs["Petal Width"].value()

        preview = f"SL: {sl:.1f}  |  SW: {sw:.1f}\nPL: {pl:.1f}  |  PW: {pw:.1f}"
        
        if self.current_class:
            preview += f"\n\nClasse Real: {self.current_class}"
        else:
            preview += f"\n\nClasse Real: Desconhecida"

        self.update_card_callback(preview)
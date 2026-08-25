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
    def __init__(self, update_card_callback, on_back_callback, on_import_callback, on_split_callback, on_dataset_ready=None):
        super().__init__("Entrada de Dados", on_back_callback)
        self.update_card_callback = update_card_callback
        self.on_import_callback = on_import_callback
        self.on_split_callback = on_split_callback
        self.on_dataset_ready = on_dataset_ready
        self.current_class = None
        
        container = QWidget()
        layout = QVBoxLayout(container)

        # --- PAINEL DE INPUTS MANUAIS ---
        self.input_panel = QWidget()
        self.input_grid = QGridLayout(self.input_panel)
        self.inputs = {}
        layout.addWidget(self.input_panel)

        self.generate_inputs_panel(["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"])

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
        self.spin_pct.setRange(1, 100)
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
        lbl_feat = QLabel("Features Ativas:")
        lbl_feat.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px; font-weight: bold;")
        feature_layout.addWidget(lbl_feat)
        
        self.feature_layout = feature_layout
        self.feature_checkboxes = {}
        # Inicialmente cria com Iris para compatibilidade caso abra sem carregar CSV
        self.generate_feature_checkboxes(["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"])
        
        feature_layout.addStretch()
        layout.addWidget(feature_panel)

        self.add_main_content(container)
        self.sync_to_card()

    def get_selected_features(self):
        return [name for name, chk in self.feature_checkboxes.items() if chk.isChecked()]


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
            import pandas as pd
            df = pd.read_csv(file_path, nrows=0) # Ler apenas os cabeçalhos
            cols = df.columns.tolist()
            
            if not cols:
                raise Exception("O CSV parece estar vazio ou não possui cabeçalhos.")
                
            from PyQt6.QtWidgets import QInputDialog
            target_col, ok = QInputDialog.getItem(
                self, "Selecionar Variável Alvo", 
                "Qual coluna representa a CLASSE (Target) do dataset?", 
                cols, len(cols)-1, False
            )
            
            if not ok or not target_col:
                return # Usuário cancelou
                
            headers, data = self.on_import_callback(file_path, target_col)
            
            # Aleatorização Automática solicitada pelo usuário
            train_ratio = self.spin_pct.value() / 100.0
            headers, data = self.on_split_callback(True, train_ratio) # True = Stratified
            
            self.render_table(headers, data)
            
            # Ocultar painel de input manual pois os campos antigos não valem mais
            if hasattr(self, 'input_panel'):
                self.input_panel.setVisible(False)
                
            if hasattr(self, 'on_dataset_ready') and self.on_dataset_ready:
                self.on_dataset_ready()
                
            QMessageBox.information(self, "Sucesso", f"Dataset carregado e separado com alvo: {target_col}")
        except Exception as e:
            QMessageBox.critical(self, "Erro de Importação", str(e))

    def apply_split(self, stratified):
        train_ratio = self.spin_pct.value() / 100.0
        headers, data = self.on_split_callback(stratified, train_ratio)
        self.render_table(headers, data)

    def generate_feature_checkboxes(self, feature_names):
        # Clear old checkboxes
        for name, chk in self.feature_checkboxes.items():
            self.feature_layout.removeWidget(chk)
            chk.deleteLater()
            
        self.feature_checkboxes = {}
        # Insert them before the stretch at the end
        # Remove the stretch temporarily
        item = self.feature_layout.takeAt(self.feature_layout.count() - 1)
        
        for name in feature_names:
            chk = QCheckBox(name)
            chk.setChecked(True)
            chk.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px;")
            self.feature_checkboxes[name] = chk
            self.feature_layout.addWidget(chk)
            
        if item:
            self.feature_layout.addItem(item)

    def generate_inputs_panel(self, feature_names):
        # Clear existing layout
        while self.input_grid.count():
            item = self.input_grid.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
                
        self.inputs = {}
        for i, name in enumerate(feature_names):
            lbl = QLabel(name)
            lbl.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px;")
            spin = QDoubleSpinBox()
            spin.setRange(-100000.0, 100000.0)
            spin.setSingleStep(0.1)
            spin.setDecimals(4)
            spin.setStyleSheet(f"background-color: #333; color: {TEXT_PRIMARY}; padding: 6px; font-size: 14px;")
            spin.valueChanged.connect(self.on_manual_input_change)
            
            self.inputs[name] = spin
            row, col = i // 2, (i % 2) * 2
            self.input_grid.addWidget(lbl, row, col)
            self.input_grid.addWidget(spin, row, col + 1)
            
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

    def get_current_inputs(self):
        return {name: spin.value() for name, spin in self.inputs.items()}

    def on_table_double_click(self, row, column):
        self._ignore_manual = True

        try:
            keys = list(self.inputs.keys())
            for i, key in enumerate(keys):
                raw_text = self.table.item(row, i).text()
                try:
                    val = float(raw_text.replace(',', '.'))
                    self.inputs[key].setValue(val)
                except ValueError:
                    pass

            target_idx = len(keys)
            if self.table.columnCount() > target_idx:
                self.current_class = self.table.item(row, target_idx).text()
            else:
                self.current_class = None
                
        except Exception:
            pass
        finally:
            del self._ignore_manual
            self.sync_to_card()

    def sync_to_card(self):
        preview = ""
        for i, (name, spin) in enumerate(self.inputs.items()):
            short_name = "".join([word[0] for word in name.split()[:2]]).upper()
            if not short_name: short_name = name[:2].upper()
            
            preview += f"{short_name}: {spin.value():.1f} | "
            if (i + 1) % 2 == 0:
                preview = preview[:-2] + "\n"
                
        preview = preview.strip()
        if preview.endswith("|"): preview = preview[:-1].strip()
        
        if self.current_class:
            preview += f"\n\nClasse Real: {self.current_class}"
        else:
            preview += f"\n\nClasse Real: Desconhecida"

        self.update_card_callback(preview)
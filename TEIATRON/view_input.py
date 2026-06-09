# view_input.py
import csv
import random
from PyQt6.QtWidgets import (
    QLabel, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QDoubleSpinBox, 
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, 
    QFileDialog, QGridLayout, QMessageBox, QSpinBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from config import TEXT_PRIMARY, ACCENT_COLOR, ACCENT_TEXT, WARNING_COLOR
from base_components import BaseCard, BaseExpandedPage

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
    def __init__(self, update_card_callback, on_back_callback):
        super().__init__("Entrada de Dados", on_back_callback)
        self.update_card_callback = update_card_callback
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

        self.add_main_content(container)
        self.sync_to_card()

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
        file_path, _ = QFileDialog.getOpenFileName(self, "Importar CSV", "", "Arquivos CSV (*.csv)")
        if not file_path:
            return

        with open(file_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            data = list(reader)

        if not data or len(data) < 2:
            return

        headers = data[0]
        rows = data[1:]

        self.table.setRowCount(len(rows))
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)

        for i, row_data in enumerate(rows):
            for j, val in enumerate(row_data):
                self.table.setItem(i, j, QTableWidgetItem(val.strip()))

    def apply_split(self, stratified):
        """Aplica a aleatorização e divide entre Treino e Teste com base na SpinBox."""
        rows = self.table.rowCount()
        cols = self.table.columnCount()
        
        if rows == 0:
            return

        # 1. Extrair os cabeçalhos atuais
        headers = []
        for j in range(cols):
            item = self.table.horizontalHeaderItem(j)
            headers.append(item.text() if item else f"Col {j}")

        # Identificar onde está a classe e verificar se já existe a coluna "Conjunto"
        has_conjunto = "Conjunto" in headers
        class_col_idx = cols - 2 if has_conjunto else cols - 1

        if not has_conjunto:
            headers.append("Conjunto")
        
        # 2. Extrair os dados de todas as linhas
        all_data = []
        for i in range(rows):
            row_data = []
            for j in range(cols):
                row_data.append(self.table.item(i, j).text())
            if not has_conjunto:
                row_data.append("") # Espaço reservado para a nova coluna "Conjunto"
            all_data.append(row_data)

        train_ratio = self.spin_pct.value() / 100.0

        # 3. Aplicar Lógica de Separação
        if stratified:
            # Agrupar por classe
            groups = {}
            for row in all_data:
                c = row[class_col_idx]
                if c not in groups:
                    groups[c] = []
                groups[c].append(row)
            
            final_data = []
            for c, group in groups.items():
                random.shuffle(group) # Embaralha DENTRO da classe
                split_idx = int(len(group) * train_ratio)
                for i, row in enumerate(group):
                    row[-1] = "Treino" if i < split_idx else "Teste"
                    final_data.append(row)
            
            # Reatribui all_data mantendo as classes agrupadas visualmente na tabela
            all_data = final_data 
        else:
            random.shuffle(all_data) # Embaralha GLOBALMENTE ignorando classes
            split_idx = int(len(all_data) * train_ratio)
            for i, row in enumerate(all_data):
                row[-1] = "Treino" if i < split_idx else "Teste"

        # 4. Reescrever a tabela com os novos dados
        self.table.clearContents()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(all_data))

        for i, row in enumerate(all_data):
            for j, val in enumerate(row):
                item = QTableWidgetItem(val)
                
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

    def get_full_dataset(self):
        """Retorna o dataset completo, as classes e as marcações de Treino/Teste."""
        rows = self.table.rowCount()
        cols = self.table.columnCount()
        
        if rows == 0:
            raise Exception("Dataset não carregado")
            
        keys = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
        dataset = {key: [] for key in keys}
        class_data = []
        conjunto_data = [] # Nova lista para Treino/Teste

        # Verifica se a coluna Conjunto existe
        headers = []
        for j in range(cols):
            item = self.table.horizontalHeaderItem(j)
            headers.append(item.text() if item else "")
        has_conjunto = "Conjunto" in headers
        conj_idx = headers.index("Conjunto") if has_conjunto else -1

        for i in range(rows):
            try:
                for j, key in enumerate(keys):
                    if j < cols:
                        val = float(self.table.item(i, j).text().replace(',', '.'))
                        dataset[key].append(val)
                    else:
                        dataset[key].append(0.0)
                
                # Coleta a Classe
                if cols >= 6:
                    c = self.table.item(i, 4).text().strip()
                elif cols >= 5:
                    c = self.table.item(i, 4).text().strip()
                elif cols > 2:
                    c = self.table.item(i, cols-1).text().strip()
                else:
                    c = "Desconhecida"
                    
                # Coleta o Conjunto (Se não foi dividido, assume tudo como Treino)
                conj = self.table.item(i, conj_idx).text().strip() if has_conjunto else "Treino"
                    
                class_data.append(c)
                conjunto_data.append(conj)
            except Exception:
                pass
                
        return dataset, class_data, conjunto_data
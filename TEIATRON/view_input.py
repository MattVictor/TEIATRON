import csv
from PyQt6.QtWidgets import (
    QLabel, QWidget, QVBoxLayout, QPushButton, QDoubleSpinBox, 
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, 
    QFileDialog, QGridLayout, QMessageBox
)
from PyQt6.QtCore import Qt
from config import TEXT_PRIMARY, ACCENT_COLOR, ACCENT_TEXT, WARNING_COLOR
from base_components import BaseCard, BaseExpandedPage

class InputCard(BaseCard):
    def __init__(self, on_expand_callback):
        super().__init__("Entrada (Iris)", on_expand_callback)
        
        self.preview_label = QLabel("Nenhum dado carregado.")
        self.preview_label.setWordWrap(True)
        self.preview_label.setStyleSheet(f"color: {WARNING_COLOR}; font-size: 15px; font-weight: bold; font-family: 'Consolas';")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignTop)
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
        self.btn_classify.clicked.connect(self.show_classification_popup)
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

        btn_import = QPushButton("📂 Importar Dataset (.csv)")
        btn_import.setStyleSheet(f"""
            QPushButton {{ background-color: #4CAF50; color: white; font-weight: bold; font-size: 14px; padding: 10px; border-radius: 5px; margin-top: 10px; }}
            QPushButton:hover {{ background-color: #45a049; }}
        """)
        btn_import.clicked.connect(self.load_csv)
        layout.addWidget(btn_import)

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

        self.add_main_content(container)
        self.sync_to_card()

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

            if col_count >= 5:
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
        """Retorna o dataset completo (Dicionário com as 4 colunas) e as classes."""
        rows = self.table.rowCount()
        cols = self.table.columnCount()
        if rows == 0:
            return None, None
            
        keys = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
        dataset = {key: [] for key in keys}
        class_data = []

        for i in range(rows):
            try:
                for j, key in enumerate(keys):
                    if j < cols:
                        val = float(self.table.item(i, j).text().replace(',', '.'))
                        dataset[key].append(val)
                    else:
                        dataset[key].append(0.0) # Fallback seguro
                
                # A classe geralmente fica na 5ª coluna
                if cols >= 5:
                    c = self.table.item(i, 4).text().strip()
                elif cols > 2:
                    c = self.table.item(i, cols-1).text().strip()
                else:
                    c = "Desconhecida"
                    
                class_data.append(c)
            except Exception:
                pass
                
        return dataset, class_data
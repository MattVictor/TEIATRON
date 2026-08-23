# view_algorithm.py
import random
from PyQt6.QtWidgets import (
    QLabel, QWidget, QVBoxLayout, QComboBox, QFormLayout, 
    QSpinBox, QDoubleSpinBox, QGroupBox, QPushButton, QTextEdit, 
    QLineEdit, QCheckBox, QHBoxLayout, QSplitter, QScrollArea
)
from PyQt6.QtCore import Qt
from views.config import TEXT_PRIMARY, TEXT_SECONDARY, ACCENT_COLOR, WARNING_COLOR, ACCENT_TEXT
from views.base_components import BaseCard, BaseExpandedPage

class AlgorithmCard(BaseCard):
    def __init__(self, on_expand_callback, on_train_callback):
        super().__init__("Algoritmo", on_expand_callback)
        self.preview_label = QLabel("Nenhum algoritmo selecionado.")
        self.preview_label.setWordWrap(True)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet(f"color: {WARNING_COLOR}; font-size: 14px; font-weight: bold;")
        self.add_preview_content(self.preview_label)

        self.btn_train = QPushButton("Treinar Modelo")
        self.btn_train.setStyleSheet(f"""
            QPushButton {{ background-color: {ACCENT_COLOR}; color: {ACCENT_TEXT}; font-weight: bold; font-size: 14px; padding: 8px; border-radius: 5px; }}
            QPushButton:hover {{ background-color: #00B3CC; }}
        """)
        self.btn_train.clicked.connect(on_train_callback)
        self.layout.addWidget(self.btn_train)

    def update_preview_text(self, text):
        self.preview_label.setText(text)

class AlgorithmExpandedPage(BaseExpandedPage):
    def __init__(self, update_card_callback, on_back_callback, on_train_callback, get_metadata_callback=None):
        super().__init__("Configuração de Machine Learning", on_back_callback)
        self.update_card_callback = update_card_callback
        self.get_metadata_callback = get_metadata_callback
        self.dynamic_widgets = {}
        self.log_font_size = 13  
        
        container = QWidget()
        main_layout = QVBoxLayout(container)
        
        # ==========================================================
        # SPLITTER VERTICAL: Permite redimensionar Entradas vs Logs
        # ==========================================================
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setStyleSheet("""
            QSplitter::handle { 
                background-color: #555; 
                height: 3px; 
                margin: 5px 0px; 
            }
            QSplitter::handle:hover {
                background-color: #00E5FF;
            }
        """)

        # --- METADE SUPERIOR DO SPLITTER: INPUTS (AGORA COM SCROLL) ---
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 10, 0) # Margem direita para acomodar a barra de rolagem
        
        self.combo_algo = QComboBox()
        self.combo_algo.addItems(["Distância Mínima", "Distância Máxima", "Perceptron", "Rede Neural (MLP)", "Problema do XOR", "Bayes Ótimo", "Naive Bayes","Máquina de Vetores de Suporte (SVM)"])
        self.combo_algo.setStyleSheet(f"QComboBox {{ background-color: #333; color: {TEXT_PRIMARY}; padding: 8px; font-size: 16px; border-radius: 4px; }}")
        self.combo_algo.currentIndexChanged.connect(self.build_dynamic_form)
        
        top_layout.addWidget(QLabel("Selecione o Algoritmo:", styleSheet=f"color:{TEXT_SECONDARY}; font-size:16px;"))
        top_layout.addWidget(self.combo_algo)
        
        self.group_box = QGroupBox("Hiperparâmetros e Filtros")
        self.group_box.setStyleSheet(f"QGroupBox {{ color: {ACCENT_COLOR}; font-size: 16px; font-weight: bold; border: 1px solid #444; margin-top: 10px; padding-top: 15px; }}")
        self.form_layout = QFormLayout(self.group_box)
        top_layout.addWidget(self.group_box)
        top_layout.addStretch() 
        
        # Cria a Área de Rolagem e coloca o top_widget dentro dela
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(top_widget)
        scroll_area.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")

        # --- METADE INFERIOR DO SPLITTER: LOGS ---
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        
        log_header_layout = QHBoxLayout()
        lbl_logs = QLabel("Logs de Treinamento:")
        lbl_logs.setStyleSheet(f"color: {WARNING_COLOR}; font-weight: bold; font-size: 14px;")
        log_header_layout.addWidget(lbl_logs)
        log_header_layout.addStretch()
        
        btn_zoom_in = QPushButton("A+")
        btn_zoom_in.setFixedSize(35, 25)
        btn_zoom_in.setStyleSheet("QPushButton { background-color: #333; color: white; font-weight: bold; border: 1px solid #555; border-radius: 3px; } QPushButton:hover { background-color: #444; }")
        btn_zoom_in.clicked.connect(self.zoom_in_logs)
        
        btn_zoom_out = QPushButton("A-")
        btn_zoom_out.setFixedSize(35, 25)
        btn_zoom_out.setStyleSheet("QPushButton { background-color: #333; color: white; font-weight: bold; border: 1px solid #555; border-radius: 3px; } QPushButton:hover { background-color: #444; }")
        btn_zoom_out.clicked.connect(self.zoom_out_logs)
        
        log_header_layout.addWidget(btn_zoom_in)
        log_header_layout.addWidget(btn_zoom_out)
        bottom_layout.addLayout(log_header_layout)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.update_log_font()
        bottom_layout.addWidget(self.log_box)
        
        # Adiciona a Área de Rolagem e o Log no Splitter
        splitter.addWidget(scroll_area)
        splitter.addWidget(bottom_widget)
        splitter.setSizes([350, 450]) 
        
        main_layout.addWidget(splitter, stretch=1)

        # --- BOTÃO TREINAR FIXO NO RODAPÉ ---
        self.btn_train_expanded = QPushButton("▶ INICIAR TREINAMENTO")
        self.btn_train_expanded.setStyleSheet(f"""
            QPushButton {{ background-color: #4CAF50; color: white; font-weight: bold; font-size: 16px; padding: 12px; border-radius: 5px; margin-top: 10px; }}
            QPushButton:hover {{ background-color: #45a049; }}
        """)
        self.btn_train_expanded.clicked.connect(on_train_callback)
        main_layout.addWidget(self.btn_train_expanded)
        
        self.add_main_content(container)

    def build_dynamic_form(self):
        while self.form_layout.count():
            child = self.form_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
        algo = self.combo_algo.currentText()
        metadata = self.get_metadata_callback(algo) if self.get_metadata_callback else []
        
        self.dynamic_widgets = {}
        self.widget_metadata = {}
        classes_iris = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
        
        for param in metadata:
            name = param["name"]
            p_type = param["type"]
            default = param.get("default")
            
            self.widget_metadata[name] = param
            
            if p_type == "bool":
                widget = QCheckBox(name)
                widget.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px; padding: 4px;")
                if default is not None: widget.setChecked(default)
                widget.stateChanged.connect(self.sync_to_card)
                self.form_layout.addRow(widget)
                self.dynamic_widgets[name] = widget
                
            elif p_type == "int":
                widget = QSpinBox()
                widget.setRange(param.get("min", 0), param.get("max", 100000))
                if default is not None: widget.setValue(default)
                widget.setStyleSheet(f"background-color: #333; color: {TEXT_PRIMARY}; padding: 4px;")
                widget.valueChanged.connect(self.sync_to_card)
                self.form_layout.addRow(f"{name}:", widget)
                self.dynamic_widgets[name] = widget
                
            elif p_type == "float":
                widget = QDoubleSpinBox()
                widget.setDecimals(4)
                widget.setRange(param.get("min", 0.0), param.get("max", 1000.0))
                widget.setSingleStep(0.01)
                if default is not None: widget.setValue(default)
                widget.setStyleSheet(f"background-color: #333; color: {TEXT_PRIMARY}; padding: 4px;")
                widget.valueChanged.connect(self.sync_to_card)
                self.form_layout.addRow(f"{name}:", widget)
                self.dynamic_widgets[name] = widget
                
            elif p_type == "options":
                widget = QComboBox()
                widget.addItems(param.get("choices", []))
                widget.setStyleSheet(f"background-color: #333; color: {TEXT_PRIMARY}; padding: 3px;")
                if default is not None: widget.setCurrentText(default)
                widget.currentIndexChanged.connect(self.sync_to_card)
                self.form_layout.addRow(f"{name}:", widget)
                self.dynamic_widgets[name] = widget
                
            elif p_type == "class_selector":
                widget = QComboBox()
                widget.addItems(classes_iris)
                widget.setStyleSheet(f"background-color: #333; color: {TEXT_PRIMARY}; padding: 3px;")
                if default is not None: widget.setCurrentText(default)
                widget.currentIndexChanged.connect(self.sync_to_card)
                self.form_layout.addRow(f"{name}:", widget)
                self.dynamic_widgets[name] = widget
                
            elif p_type == "string":
                widget = QLineEdit()
                if default is not None: widget.setText(default)
                widget.setStyleSheet(f"background-color: #333; color: {TEXT_PRIMARY}; padding: 5px;")
                widget.textChanged.connect(self.sync_to_card)
                self.form_layout.addRow(f"{name}:", widget)
                self.dynamic_widgets[name] = widget

        self.evaluate_rules()
        self.sync_to_card()

    def _get_widget_value(self, widget):
        if isinstance(widget, QCheckBox): return widget.isChecked()
        if isinstance(widget, QComboBox): return widget.currentText()
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)): return widget.value()
        if isinstance(widget, QLineEdit): return widget.text()
        return None

    def evaluate_rules(self):
        for name, widget in self.dynamic_widgets.items():
            meta = self.widget_metadata[name]
            
            # 1. Evaluate Visibility
            if "depends_on" in meta:
                dep = meta["depends_on"]
                dep_name = dep["field"]
                dep_val = dep["value"]
                
                dep_widget = self.dynamic_widgets.get(dep_name)
                if dep_widget:
                    curr_val = self._get_widget_value(dep_widget)
                    is_visible = (curr_val == dep_val)
                    
                    widget.setVisible(is_visible)
                    label = self.form_layout.labelForField(widget)
                    if label:
                        label.setVisible(is_visible)
                        
            # 2. Evaluate Mutual Exclusion
            if "prevent_same_as" in meta:
                other_name = meta["prevent_same_as"]
                other_widget = self.dynamic_widgets.get(other_name)
                if other_widget and widget.isVisible():
                    val_self = self._get_widget_value(widget)
                    val_other = self._get_widget_value(other_widget)
                    if val_self == val_other:
                        widget.blockSignals(True)
                        for i in range(widget.count()):
                            if widget.itemText(i) != val_other:
                                widget.setCurrentIndex(i)
                                break
                        widget.blockSignals(False)

    def sync_to_card(self):
        self.evaluate_rules()
        algo = self.combo_algo.currentText()
        lines = [f"Modelo: {algo}"]
        
        count = 0
        for name, widget in self.dynamic_widgets.items():
            if not widget.isVisible():
                continue
                
            if count >= 3: break # Limita o texto do Card para não estourar a tela
            
            val = self._get_widget_value(widget)
            if isinstance(widget, QCheckBox):
                val = "Sim" if val else "Não"
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                val = str(val)
                
            lines.append(f"{name}: {val}")
            count += 1
            
        self.update_card_callback("\n".join(lines))

    def get_current_params(self):
        algo = self.combo_algo.currentText()
        params = {"Algoritmo": algo}
        
        for name, widget in self.dynamic_widgets.items():
            if isinstance(widget, QCheckBox):
                params[name] = widget.isChecked()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                params[name] = widget.value()
            elif isinstance(widget, QComboBox):
                params[name] = widget.currentText()
            elif isinstance(widget, QLineEdit):
                params[name] = widget.text()
                
        return params

    def zoom_in_logs(self):
        if self.log_font_size < 30:
            self.log_font_size += 1
            self.update_log_font()

    def zoom_out_logs(self):
        if self.log_font_size > 8:
            self.log_font_size -= 1
            self.update_log_font()

    def update_log_font(self):
        self.log_box.setStyleSheet(f"background-color: #1a1a1a; color: #00FF00; font-family: 'Consolas'; font-size: {self.log_font_size}px; border: 1px solid #444; border-radius: 4px; padding: 8px;")

    def append_log(self, text):
        self.log_box.append(text)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()
        
    def clear_logs(self):
        self.log_box.clear()
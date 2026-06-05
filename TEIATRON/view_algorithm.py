from PyQt6.QtWidgets import QLabel, QWidget, QVBoxLayout, QComboBox, QFormLayout, QSpinBox, QDoubleSpinBox, QGroupBox, QPushButton
from PyQt6.QtCore import Qt
from config import TEXT_PRIMARY, TEXT_SECONDARY, ACCENT_COLOR, WARNING_COLOR, ACCENT_TEXT
from base_components import BaseCard, BaseExpandedPage

class AlgorithmCard(BaseCard):
    def __init__(self, on_expand_callback, on_train_callback):
        super().__init__("Algoritmo", on_expand_callback)
        self.preview_label = QLabel("Nenhum algoritmo selecionado.")
        self.preview_label.setWordWrap(True)
        self.preview_label.setStyleSheet(f"color: {WARNING_COLOR}; font-size: 14px; font-weight: bold;")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.add_preview_content(self.preview_label)

        # Novo botão de Treinar
        self.btn_train = QPushButton("Treinar Modelo")
        self.btn_train.setStyleSheet(f"""
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
        self.btn_train.clicked.connect(on_train_callback)
        self.layout.addWidget(self.btn_train)

    def update_preview_text(self, text):
        self.preview_label.setText(text)

class AlgorithmExpandedPage(BaseExpandedPage):
    def __init__(self, update_card_callback, on_back_callback):
        super().__init__("Configuração de Machine Learning", on_back_callback)
        self.update_card_callback = update_card_callback
        
        self.algorithms_data = {
            "Classificador Mínimo": [],
            "Classificador Máximo": [],
            "Perceptron": [("Épocas", int, 100), ("Learning Rate", float, 0.01)],
            "Perceptron Um contra todos": [("Épocas", int, 500), ("Learning Rate", float, 0.05)]
        }
        
        container = QWidget()
        layout = QVBoxLayout(container)
        
        self.combo_algo = QComboBox()
        self.combo_algo.addItems(self.algorithms_data.keys())
        self.combo_algo.setStyleSheet(f"QComboBox {{ background-color: #333; color: {TEXT_PRIMARY}; padding: 8px; font-size: 16px; border-radius: 4px; }}")
        self.combo_algo.currentIndexChanged.connect(self.build_dynamic_form)
        
        layout.addWidget(QLabel("Selecione o Algoritmo:", styleSheet=f"color:{TEXT_SECONDARY}; font-size:16px;"))
        layout.addWidget(self.combo_algo)
        
        self.group_box = QGroupBox("Hiperparâmetros")
        self.group_box.setStyleSheet(f"QGroupBox {{ color: {ACCENT_COLOR}; font-size: 16px; font-weight: bold; border: 1px solid #444; margin-top: 10px; padding-top: 15px; }}")
        self.form_layout = QFormLayout(self.group_box)
        layout.addWidget(self.group_box)
        layout.addStretch()
        
        self.add_main_content(container)
        self.current_inputs = {}
        self.build_dynamic_form()

    def build_dynamic_form(self):
        while self.form_layout.count():
            child = self.form_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
        self.current_inputs.clear()
        algo_name = self.combo_algo.currentText()
        params = self.algorithms_data[algo_name]
        
        if not params:
            lbl = QLabel("Este algoritmo não requer parâmetros adicionais.")
            lbl.setStyleSheet(f"color: {TEXT_SECONDARY};")
            self.form_layout.addRow(lbl)
        else:
            for param_name, param_type, default_val in params:
                if param_type == int:
                    spin = QSpinBox()
                    spin.setRange(1, 10000)
                elif param_type == float:
                    spin = QDoubleSpinBox()
                    spin.setRange(0.0001, 10.0)
                    spin.setDecimals(4)
                    spin.setSingleStep(0.01)
                    
                spin.setValue(default_val)
                spin.valueChanged.connect(self.sync_to_card)
                self.current_inputs[param_name] = spin
                spin.setStyleSheet(f"background-color: #333; color: {TEXT_PRIMARY}; padding: 5px;")
                
                lbl = QLabel(param_name)
                lbl.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px;")
                self.form_layout.addRow(lbl, spin)
                
        self.sync_to_card()

    def sync_to_card(self):
        algo_name = self.combo_algo.currentText()
        preview_lines = [f"Modelo: {algo_name}"]
        for param_name, widget in self.current_inputs.items():
            preview_lines.append(f"{param_name}: {widget.value()}")
        self.update_card_callback("\n".join(preview_lines))
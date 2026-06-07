# view_algorithm.py
from PyQt6.QtWidgets import (
    QLabel, QWidget, QVBoxLayout, QComboBox, QFormLayout, 
    QSpinBox, QDoubleSpinBox, QGroupBox, QPushButton, QTextEdit, QLineEdit
)
from PyQt6.QtCore import Qt
from config import TEXT_PRIMARY, TEXT_SECONDARY, ACCENT_COLOR, WARNING_COLOR, ACCENT_TEXT
from base_components import BaseCard, BaseExpandedPage

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
    def __init__(self, update_card_callback, on_back_callback, on_train_callback):
        super().__init__("Configuração de Machine Learning", on_back_callback)
        self.update_card_callback = update_card_callback
        
        # --- DICIONÁRIO DE ALGORITMOS COMPLETOS ---
        # Formato: ("Nome do Parâmetro", tipo_do_dado, valor_padrão)
        self.algorithms_data = {
            "Classificador Mínimo": [],
            "Classificador Máximo": [],
            "Perceptron": [
                ("Épocas", int, 100), 
                ("Learning Rate", float, 0.01),
                ("Bias Inicial", float, 0.0),
                ("Pesos Iniciais (w1,w2,w3,w4)", str, "0.0, 0.0, 0.0, 0.0")
            ],
            "Perceptron Um contra todos": [
                ("Épocas", int, 500), 
                ("Learning Rate", float, 0.05),
                ("Bias Inicial", float, 0.0),
                ("Pesos Iniciais (w1,w2,w3,w4)", str, "0.0, 0.0, 0.0, 0.0"),
                ("Classe Alvo", list, ["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
            ],
            "Perceptron Regra Delta": [
                ("Épocas", int, 100), 
                ("Learning Rate", float, 0.01),
                ("Bias Inicial", float, 0.0),
                ("Pesos Iniciais (w1,w2,w3,w4)", str, "0.0, 0.0, 0.0, 0.0")
            ]
        }
        
        container = QWidget()
        layout = QVBoxLayout(container)
        
        # --- 1. SELEÇÃO DE ALGORITMO ---
        self.combo_algo = QComboBox()
        self.combo_algo.addItems(self.algorithms_data.keys())
        self.combo_algo.setStyleSheet(f"QComboBox {{ background-color: #333; color: {TEXT_PRIMARY}; padding: 8px; font-size: 16px; border-radius: 4px; }}")
        self.combo_algo.currentIndexChanged.connect(self.build_dynamic_form)
        
        layout.addWidget(QLabel("Selecione o Algoritmo:", styleSheet=f"color:{TEXT_SECONDARY}; font-size:16px;"))
        layout.addWidget(self.combo_algo)
        
        # --- 2. HIPERPARÂMETROS ---
        self.group_box = QGroupBox("Hiperparâmetros")
        self.group_box.setStyleSheet(f"QGroupBox {{ color: {ACCENT_COLOR}; font-size: 16px; font-weight: bold; border: 1px solid #444; margin-top: 10px; padding-top: 15px; }}")
        self.form_layout = QFormLayout(self.group_box)
        layout.addWidget(self.group_box)
        
        # --- 3. TERMINAL DE LOGS ---
        lbl_logs = QLabel("Logs de Treinamento:")
        lbl_logs.setStyleSheet(f"color: {WARNING_COLOR}; font-weight: bold; font-size: 14px; margin-top: 15px;")
        layout.addWidget(lbl_logs)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet(f"background-color: #1a1a1a; color: #00FF00; font-family: 'Consolas'; font-size: 13px; border: 1px solid #444; border-radius: 4px; padding: 8px;")
        layout.addWidget(self.log_box, stretch=1) 

        # --- 4. BOTÃO TREINAR INFERIOR ---
        self.btn_train_expanded = QPushButton("▶ INICIAR TREINAMENTO")
        self.btn_train_expanded.setStyleSheet(f"""
            QPushButton {{ background-color: #4CAF50; color: white; font-weight: bold; font-size: 16px; padding: 12px; border-radius: 5px; margin-top: 10px; }}
            QPushButton:hover {{ background-color: #45a049; }}
        """)
        self.btn_train_expanded.clicked.connect(on_train_callback)
        layout.addWidget(self.btn_train_expanded)
        
        self.add_main_content(container)
        self.current_inputs = {}
        self.build_dynamic_form()

    def build_dynamic_form(self):
        # Limpa o formulário anterior
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
                
                # Renderiza SpinBox de Inteiros (Ex: Épocas)
                if param_type == int:
                    widget = QSpinBox()
                    widget.setRange(1, 100000)
                    widget.setValue(default_val)
                    widget.valueChanged.connect(self.sync_to_card)
                    
                # Renderiza SpinBox de Decimais (Ex: Learning Rate ou Bias)
                elif param_type == float:
                    widget = QDoubleSpinBox()
                    widget.setDecimals(4)
                    widget.setSingleStep(0.01)
                    if "Learning Rate" in param_name:
                        widget.setRange(0.0001, 10.0)
                    else:
                        widget.setRange(-100.0, 100.0) # Bias pode ser negativo
                    widget.setValue(default_val)
                    widget.valueChanged.connect(self.sync_to_card)
                    
                # Renderiza Caixa de Texto (Ex: Pesos separados por vírgula)
                elif param_type == str:
                    widget = QLineEdit()
                    widget.setText(default_val)
                    widget.textChanged.connect(self.sync_to_card)
                    
                # Renderiza Menu Suspenso (Ex: Seleção de Classe no OvA)
                elif param_type == list:
                    widget = QComboBox()
                    widget.addItems(default_val)
                    widget.currentIndexChanged.connect(self.sync_to_card)

                widget.setStyleSheet(f"background-color: #333; color: {TEXT_PRIMARY}; padding: 5px; font-size: 14px;")
                self.current_inputs[param_name] = widget
                
                lbl = QLabel(param_name)
                lbl.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px;")
                self.form_layout.addRow(lbl, widget)
                
        self.sync_to_card()

    def sync_to_card(self):
        """Lê os valores atuais dos widgets e envia para a frente do Card."""
        algo_name = self.combo_algo.currentText()
        preview_lines = [f"Modelo: {algo_name}"]
        
        for param_name, widget in self.current_inputs.items():
            # Precisamos descobrir qual o tipo do widget para ler o valor corretamente
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                val = widget.value()
            elif isinstance(widget, QLineEdit):
                val = widget.text()
            elif isinstance(widget, QComboBox):
                val = widget.currentText()
                
            preview_lines.append(f"{param_name}: {val}")
            
        self.update_card_callback("\n".join(preview_lines))

    def append_log(self, text):
        self.log_box.append(text)
        
    def clear_logs(self):
        self.log_box.clear()
# view_algorithm.py
import random
from PyQt6.QtWidgets import (
    QLabel, QWidget, QVBoxLayout, QComboBox, QFormLayout, 
    QSpinBox, QDoubleSpinBox, QGroupBox, QPushButton, QTextEdit, 
    QLineEdit, QCheckBox, QHBoxLayout, QSplitter, QScrollArea
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
        self.combo_algo.addItems(["Distância Mínima", "Distância Máxima", "Perceptron", "Problema do XOR", "Bayes Ótimo", "Naive Bayes"])
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
        self.build_dynamic_form()

    def build_dynamic_form(self):
        while self.form_layout.count():
            child = self.form_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
        algo = self.combo_algo.currentText()
        classes_iris = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
        
        if algo in ["Distância Mínima", "Distância Máxima"]:
            self.chk_multiclasse = QCheckBox("Multiclasse (Considerar as 3 classes)")
            self.chk_duas_classes = QCheckBox("Duas classes")
            
            self.chk_multiclasse.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px; padding: 4px;")
            self.chk_duas_classes.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px; padding: 4px;")
            
            self.form_layout.addRow(self.chk_multiclasse)
            self.form_layout.addRow(self.chk_duas_classes)
            
            self.dist_classes_widget = QWidget()
            dist_form = QFormLayout(self.dist_classes_widget)
            dist_form.setContentsMargins(15, 0, 0, 0)
            
            self.combo_dist_c1 = QComboBox()
            self.combo_dist_c2 = QComboBox()
            self.combo_dist_c1.addItems(classes_iris)
            self.combo_dist_c2.addItems(classes_iris)
            self.combo_dist_c2.setCurrentIndex(1)
            
            self.combo_dist_c1.setStyleSheet(f"background-color: #333; color: {TEXT_PRIMARY}; padding: 3px;")
            self.combo_dist_c2.setStyleSheet(f"background-color: #333; color: {TEXT_PRIMARY}; padding: 3px;")
            
            dist_form.addRow("Classe 1:", self.combo_dist_c1)
            dist_form.addRow("Classe 2:", self.combo_dist_c2)
            self.form_layout.addRow(self.dist_classes_widget)
            
            self.chk_multiclasse.stateChanged.connect(self.on_dist_multi_changed)
            self.chk_duas_classes.stateChanged.connect(self.on_dist_two_changed)
            
            self.combo_dist_c1.currentIndexChanged.connect(lambda: self.prevent_same_class(self.combo_dist_c1, self.combo_dist_c2))
            self.combo_dist_c2.currentIndexChanged.connect(lambda: self.prevent_same_class(self.combo_dist_c2, self.combo_dist_c1))
            
            self.chk_multiclasse.setChecked(True)
            self.dist_classes_widget.setVisible(False)
            
        elif algo == "Perceptron":
            self.chk_delta = QCheckBox("Aplicar correção via Regra Delta (Adaline)")
            self.chk_delta.setStyleSheet(f"color: {WARNING_COLOR}; font-size: 14px; font-weight: bold; margin-bottom: 10px;")
            self.chk_delta.stateChanged.connect(self.sync_to_card)
            self.form_layout.addRow(self.chk_delta)
            
            self.combo_strat = QComboBox()
            self.combo_strat.addItems(["Clássico", "Um contra todos"])
            self.combo_strat.setStyleSheet(f"QComboBox {{ background-color: #333; color: {TEXT_PRIMARY}; padding: 5px; }}")
            self.combo_strat.currentIndexChanged.connect(self.on_perceptron_strat_changed)
            self.form_layout.addRow("Estratégia:", self.combo_strat)
            
            self.strat_widget = QWidget()
            self.strat_form = QFormLayout(self.strat_widget)
            self.strat_form.setContentsMargins(0, 5, 0, 5)
            self.form_layout.addRow(self.strat_widget)
            
            self.spin_epocas = QSpinBox()
            self.spin_epocas.setRange(1, 100000)
            self.spin_epocas.setValue(100)
            self.spin_epocas.setStyleSheet(f"background-color: #333; color: {TEXT_PRIMARY}; padding: 4px;")
            self.spin_epocas.valueChanged.connect(self.sync_to_card)
            self.form_layout.addRow("Épocas:", self.spin_epocas)
            
            self.spin_lr = QDoubleSpinBox()
            self.spin_lr.setDecimals(4)
            self.spin_lr.setRange(0.0001, 10.0)
            self.spin_lr.setSingleStep(0.01)
            self.spin_lr.setValue(0.01)
            self.spin_lr.setStyleSheet(f"background-color: #333; color: {TEXT_PRIMARY}; padding: 4px;")
            self.spin_lr.valueChanged.connect(self.sync_to_card)
            self.form_layout.addRow("Learning Rate (η):", self.spin_lr)
            
            self.spin_bias = QDoubleSpinBox()
            self.spin_bias.setDecimals(4)
            self.spin_bias.setRange(-100.0, 100.0)
            self.spin_bias.setValue(0.0)
            self.spin_bias.setStyleSheet(f"background-color: #333; color: {TEXT_PRIMARY}; padding: 4px;")
            self.spin_bias.valueChanged.connect(self.sync_to_card)
            self.form_layout.addRow("Bias Inicial (θ):", self.spin_bias)
            
            pesos_container = QWidget()
            pesos_layout = QHBoxLayout(pesos_container)
            pesos_layout.setContentsMargins(0, 0, 0, 0)
            
            self.txt_pesos = QLineEdit("0.0, 0.0, 0.0, 0.0")
            self.txt_pesos.setStyleSheet(f"background-color: #333; color: {TEXT_PRIMARY}; padding: 5px;")
            self.txt_pesos.textChanged.connect(self.sync_to_card)
            
            btn_rand_weights = QPushButton("🎲 Aleatórios")
            btn_rand_weights.setStyleSheet(f"""
                QPushButton {{ 
                    background-color: {ACCENT_COLOR}; 
                    color: {ACCENT_TEXT}; 
                    font-weight: bold; 
                    padding: 5px 10px; 
                    border-radius: 3px; 
                }} 
                QPushButton:hover {{ background-color: #00B3CC; }}
            """)
            btn_rand_weights.clicked.connect(self.generate_random_weights)
            
            pesos_layout.addWidget(self.txt_pesos, stretch=1)
            pesos_layout.addWidget(btn_rand_weights)
            
            self.form_layout.addRow("Pesos Iniciais (W):", pesos_container)
            
            self.on_perceptron_strat_changed()
            
        self.sync_to_card()

    def generate_random_weights(self):
        random_w = [round(random.uniform(0.0, 5.0), 2) for _ in range(4)]
        str_w = ", ".join(map(str, random_w))
        self.txt_pesos.setText(str_w)

    def zoom_in_logs(self):
        if self.log_font_size < 30:
            self.log_font_size += 1
            self.update_log_font()

    def zoom_out_logs(self):
        if self.log_font_size > 8:
            self.log_font_size -= 1
            self.update_log_font()

    def update_log_font(self):
        self.log_box.setStyleSheet(f"""
            background-color: #1a1a1a; 
            color: #00FF00; 
            font-family: 'Consolas'; 
            font-size: {self.log_font_size}px; 
            border: 1px solid #444; 
            border-radius: 4px; 
            padding: 8px;
        """)

    def on_dist_multi_changed(self, state):
        if state == 2: 
            self.chk_duas_classes.blockSignals(True)
            self.chk_duas_classes.setChecked(False)
            self.chk_duas_classes.blockSignals(False)
            self.dist_classes_widget.setVisible(False)
        else:
            if not self.chk_duas_classes.isChecked():
                self.chk_multiclasse.blockSignals(True)
                self.chk_multiclasse.setChecked(True)
                self.chk_multiclasse.blockSignals(False)
        self.sync_to_card()

    def on_dist_two_changed(self, state):
        if state == 2: 
            self.chk_multiclasse.blockSignals(True)
            self.chk_multiclasse.setChecked(False)
            self.chk_multiclasse.blockSignals(False)
            self.dist_classes_widget.setVisible(True)
        else:
            if not self.chk_multiclasse.isChecked():
                self.chk_duas_classes.blockSignals(True)
                self.chk_duas_classes.setChecked(True)
                self.chk_duas_classes.blockSignals(False)
            self.dist_classes_widget.setVisible(True)
        self.sync_to_card()

    def prevent_same_class(self, changed_combo, other_combo):
        if changed_combo.currentText() == other_combo.currentText():
            other_combo.blockSignals(True)
            for i in range(other_combo.count()):
                if other_combo.itemText(i) != changed_combo.currentText():
                    other_combo.setCurrentIndex(i)
                    break
            other_combo.blockSignals(False)
        self.sync_to_card()

    def on_perceptron_strat_changed(self):
        while self.strat_form.count():
            child = self.strat_form.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
        strat = self.combo_strat.currentText()
        classes_iris = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
        
        if strat == "Clássico":
            self.combo_perp_c1 = QComboBox()
            self.combo_perp_c2 = QComboBox()
            self.combo_perp_c1.addItems(classes_iris)
            self.combo_perp_c2.addItems(classes_iris)
            self.combo_perp_c2.setCurrentIndex(1)
            
            self.combo_perp_c1.setStyleSheet(f"background-color: #333; color: {TEXT_PRIMARY}; padding: 3px;")
            self.combo_perp_c2.setStyleSheet(f"background-color: #333; color: {TEXT_PRIMARY}; padding: 3px;")
            
            self.combo_perp_c1.currentIndexChanged.connect(lambda: self.prevent_same_class(self.combo_perp_c1, self.combo_perp_c2))
            self.combo_perp_c2.currentIndexChanged.connect(lambda: self.prevent_same_class(self.combo_perp_c2, self.combo_perp_c1))
            
            self.strat_form.addRow("Classe 1 (+1):", self.combo_perp_c1)
            self.strat_form.addRow("Classe 2 (-1):", self.combo_perp_c2)
            
        elif strat == "Um contra todos":
            self.combo_perp_target = QComboBox()
            self.combo_perp_target.addItems(classes_iris)
            self.combo_perp_target.setStyleSheet(f"background-color: #333; color: {TEXT_PRIMARY}; padding: 3px;")
            self.combo_perp_target.currentIndexChanged.connect(self.sync_to_card)
            
            self.strat_form.addRow("Classe Isolada (+1):", self.combo_perp_target)
            
        self.sync_to_card()

    def sync_to_card(self):
        algo = self.combo_algo.currentText()
        lines = [f"Modelo: {algo}"]
        
        if algo in ["Distância Mínima", "Distância Máxima"]:
            if hasattr(self, 'chk_multiclasse') and self.chk_multiclasse.isChecked():
                lines.append("Escopo: Multiclasse")
            elif hasattr(self, 'chk_duas_classes') and self.chk_duas_classes.isChecked():
                lines.append("Escopo: Binário (2 Classes)")
                if hasattr(self, 'combo_dist_c1'):
                    lines.append(f"C1: {self.combo_dist_c1.currentText()} | C2: {self.combo_dist_c2.currentText()}")
                    
        elif algo == "Perceptron":
            if hasattr(self, 'chk_delta'):
                lines.append("Algoritmo: Adaline (Regra Delta)" if self.chk_delta.isChecked() else "Algoritmo: Perceptron Tradicional")
            if hasattr(self, 'combo_strat'):
                strat = self.combo_strat.currentText()
                lines.append(f"Estratégia: {strat}")
                if strat == "Clássico" and hasattr(self, 'combo_perp_c1'):
                    lines.append(f"Classes: {self.combo_perp_c1.currentText()} x {self.combo_perp_c2.currentText()}")
                elif strat == "Um contra todos" and hasattr(self, 'combo_perp_target'):
                    lines.append(f"Alvo: {self.combo_perp_target.currentText()} vs Resto")
            if hasattr(self, 'spin_epocas'):
                lines.append(f"Épocas: {self.spin_epocas.value()} | LR: {self.spin_lr.value()}")
                
        self.update_card_callback("\n".join(lines))

    def get_current_params(self):
        algo = self.combo_algo.currentText()
        params = {"Algoritmo": algo}
        
        if algo in ["Distância Mínima", "Distância Máxima"]:
            params["Multiclasse"] = self.chk_multiclasse.isChecked()
            if self.chk_duas_classes.isChecked():
                params["Classe 1"] = self.combo_dist_c1.currentText()
                params["Classe 2"] = self.combo_dist_c2.currentText()
                
        elif algo == "Perceptron":
            params["Regra Delta"] = self.chk_delta.isChecked()
            strat = self.combo_strat.currentText()
            params["Estratégia"] = strat
            if strat == "Clássico":
                params["Classe 1"] = self.combo_perp_c1.currentText()
                params["Classe 2"] = self.combo_perp_c2.currentText()
            elif strat == "Um contra todos":
                params["Classe Alvo"] = self.combo_perp_target.currentText()
                
            params["Épocas"] = self.spin_epocas.value()
            params["Learning Rate"] = self.spin_lr.value()
            params["Bias Inicial"] = self.spin_bias.value()
            params["Pesos Iniciais"] = self.txt_pesos.text()
            
        return params

    def append_log(self, text):
        self.log_box.append(text)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())
        
    def clear_logs(self):
        self.log_box.clear()
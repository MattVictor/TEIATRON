# view_charts.py
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QListWidget, QSplitter, QCheckBox, QPushButton, QMessageBox
)
from PyQt6.QtCore import Qt
from config import ACCENT_COLOR, TEXT_PRIMARY, WARNING_COLOR, BG_CARD
from base_components import BaseCard, BaseExpandedPage

CLASS_COLORS = ["#FF5252", "#4CAF50", "#448AFF", "#FFEB3B", "#E040FB", "#00BCD4"]

class ChartsCard(BaseCard):
    def __init__(self, on_expand_callback):
        super().__init__("Gráficos", on_expand_callback)
        self.preview_plot = pg.PlotWidget()
        self.preview_plot.setMouseEnabled(x=False, y=False)
        self.preview_plot.hideButtons()
        self.preview_plot.setMenuEnabled(False)
        self.preview_plot.showGrid(x=True, y=True, alpha=0.1)
        self.preview_plot.setStyleSheet("border: none;")
        self.add_preview_content(self.preview_plot)
        self.preview_plot.clear()

class ChartsExpandedPage(BaseExpandedPage):
    def __init__(self, preview_widget, on_back_callback):
        super().__init__("Análise de Gráficos", on_back_callback)
        self.preview_widget = preview_widget
        self.generated_charts = {}
        
        self.current_dataset = None
        self.current_classes = None
        self.current_conjuntos = None
        
        container = QWidget()
        layout = QHBoxLayout(container)
        
        # --- PAINEL ESQUERDO ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        lbl_list = QLabel("Gráficos Gerados:")
        lbl_list.setStyleSheet(f"color: {TEXT_PRIMARY}; font-weight: bold; font-size: 14px;")
        left_layout.addWidget(lbl_list)
        
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(f"""
            QListWidget {{ background-color: #2b2b2b; color: {TEXT_PRIMARY}; border: 1px solid #444; border-radius: 5px; font-size: 14px; padding: 5px; }}
            QListWidget::item {{ padding: 8px; }}
            QListWidget::item:selected {{ background-color: {ACCENT_COLOR}; color: #000; font-weight: bold; border-radius: 3px;}}
        """)
        self.list_widget.currentRowChanged.connect(self.display_chart)
        left_layout.addWidget(self.list_widget)
        
        # Filtros de Treino/Teste
        lbl_filters = QLabel("Filtros de Exibição:")
        lbl_filters.setStyleSheet(f"color: {WARNING_COLOR}; font-weight: bold; margin-top: 10px;")
        left_layout.addWidget(lbl_filters)

        self.chk_train = QCheckBox("Mostrar Treino (Círculos)")
        self.chk_train.setChecked(True)
        self.chk_train.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px;")
        self.chk_train.stateChanged.connect(self.refresh_current_chart)
        left_layout.addWidget(self.chk_train)

        self.chk_test = QCheckBox("Mostrar Teste (Triângulos)")
        self.chk_test.setChecked(True)
        self.chk_test.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px;")
        self.chk_test.stateChanged.connect(self.refresh_current_chart)
        left_layout.addWidget(self.chk_test)

        # Variáveis
        lbl_vars = QLabel("Variáveis (Selecione 2):")
        lbl_vars.setStyleSheet(f"color: {ACCENT_COLOR}; font-weight: bold; margin-top: 10px;")
        left_layout.addWidget(lbl_vars)

        self.checkboxes = []
        for var_name in ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]:
            chk = QCheckBox(var_name)
            chk.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px; padding: 2px;")
            self.checkboxes.append(chk)
            left_layout.addWidget(chk)

        self.btn_plot = QPushButton("Plotar gráfico")
        self.btn_plot.setStyleSheet(f"""
            QPushButton {{ background-color: #4CAF50; color: white; font-weight: bold; font-size: 14px; padding: 10px; border-radius: 5px; margin-top: 10px; }}
            QPushButton:hover {{ background-color: #45a049; }}
        """)
        self.btn_plot.clicked.connect(self.plot_custom_chart)
        left_layout.addWidget(self.btn_plot)
        
        # --- PAINEL DIREITO: Gráfico ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.coord_label = QLabel("Mouse: X=0.00, Y=0.00")
        self.coord_label.setStyleSheet(f"color: {WARNING_COLOR}; font-weight: bold; font-family: 'Consolas';")
        self.coord_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        right_layout.addWidget(self.coord_label)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.scene().sigMouseMoved.connect(self.mouse_moved)
        self.legend = self.plot_widget.addLegend(offset=(10, 10), brush='#1E1E1EE6')
        right_layout.addWidget(self.plot_widget)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([250, 800])
        splitter.setStyleSheet("QSplitter::handle { background-color: #444; width: 2px; }")
        
        layout.addWidget(splitter)
        self.add_main_content(container)

    # ==========================================
    # LÓGICA DE DADOS E PLOTAGEM
    # ==========================================
    def set_dataset(self, dataset, classes, conjuntos):
        self.current_dataset = dataset
        self.current_classes = classes
        self.current_conjuntos = conjuntos

    def set_classified_point(self, point_dict):
        """Recebe o ponto classificado, salva na memória e atualiza o gráfico atual."""
        self.classified_point = point_dict
        self.refresh_current_chart()

    def plot_custom_chart(self):
        if not self.current_dataset:
            self.show_error_popup("Nenhum dado disponível. Importe os dados e treine o modelo primeiro.")
            return
            
        selected_chks = [chk for chk in self.checkboxes if chk.isChecked()]
        if len(selected_chks) != 2:
            self.show_error_popup("Por favor, selecione EXATAMENTE DUAS variáveis para plotar o gráfico.")
            return
            
        x_key, y_key = selected_chks[0].text(), selected_chks[1].text()
        x_data = self.current_dataset[x_key]
        y_data = self.current_dataset[y_key]
        
        chart_name = f"Dispersão ({x_key} x {y_key})"
        # Passamos as chaves x_key e y_key
        self.add_chart(chart_name, "Dispersão", x_data, y_data, self.current_classes, self.current_conjuntos, x_key, y_key)
        
        self.list_widget.setCurrentRow(self.list_widget.count() - 1)

    def show_error_popup(self, message):
        msg = QMessageBox(self)
        msg.setWindowTitle("Erro de Seleção")
        msg.setText(message)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setStyleSheet("""
            QMessageBox { background-color: #1E1E1E; color: #FFFFFF; }
            QLabel { color: #FFFFFF; font-size: 14px; }
            QPushButton { background-color: #FF5252; color: #FFFFFF; padding: 5px 15px; font-weight: bold; border-radius: 3px; }
        """)
        msg.exec()

    def add_chart(self, name, chart_type, x_data, y_data, classes=None, conjuntos=None, x_key=None, y_key=None):
        # Agora salvamos as chaves x_key e y_key para saber onde encaixar o novo ponto
        self.generated_charts[name] = {
            "type": chart_type, "x": x_data, "y": y_data, 
            "classes": classes, "conjuntos": conjuntos,
            "x_key": x_key, "y_key": y_key
        }
        self.list_widget.addItem(name)

    def clear_charts(self):
        self.generated_charts.clear()
        self.list_widget.clear()
        self.plot_widget.clear()
        self.preview_widget.clear()
        if self.legend is not None:
            self.legend.clear()

    def refresh_current_chart(self):
        """Redesenha o gráfico atual quando as checkboxes são alteradas."""
        current_row = self.list_widget.currentRow()
        if current_row >= 0:
            self.display_chart(current_row)

    def display_chart(self, index):
        if index < 0 or index >= self.list_widget.count():
            return
            
        chart_name = self.list_widget.item(index).text()
        chart_info = self.generated_charts.get(chart_name)
        
        self.plot_widget.clear()
        self.preview_widget.clear()
        if self.legend is not None:
            self.legend.clear() 
            
        if not chart_info:
            return
            
        c_type = chart_info["type"]
        x = np.array(chart_info["x"])
        y = np.array(chart_info["y"])
        classes = chart_info["classes"]
        conjuntos = chart_info.get("conjuntos", ["Treino"] * len(x))
        x_key = chart_info.get("x_key")
        y_key = chart_info.get("y_key")
        
        show_train = self.chk_train.isChecked()
        show_test = self.chk_test.isChecked()
        
        if c_type == "Dispersão" and classes:
            unique_classes = list(sorted(set(classes)))
            for i, cls_name in enumerate(unique_classes):
                color = CLASS_COLORS[i % len(CLASS_COLORS)]
                brush = pg.mkBrush(color=color)
                
                # PLOT DE TREINO
                if show_train:
                    idx_tr = [j for j, (c, conj) in enumerate(zip(classes, conjuntos)) if c == cls_name and conj == "Treino"]
                    if idx_tr:
                        self.plot_widget.plot(x[idx_tr], y[idx_tr], pen=None, symbol='o', symbolBrush=brush, symbolSize=8, name=f"{cls_name} (Treino)")
                        self.preview_widget.plot(x[idx_tr], y[idx_tr], pen=None, symbol='o', symbolBrush=brush, symbolSize=4)
                
                # PLOT DE TESTE
                if show_test:
                    idx_ts = [j for j, (c, conj) in enumerate(zip(classes, conjuntos)) if c == cls_name and conj == "Teste"]
                    if idx_ts:
                        self.plot_widget.plot(x[idx_ts], y[idx_ts], pen=None, symbol='t', symbolBrush=brush, symbolSize=9, name=f"{cls_name} (Teste)")
                        self.preview_widget.plot(x[idx_ts], y[idx_ts], pen=None, symbol='t', symbolBrush=brush, symbolSize=4)
        else:
            pen = pg.mkPen(color=ACCENT_COLOR, width=3)
            self.plot_widget.plot(x, y, pen=pen)
            self.preview_widget.plot(x, y, pen=pen)

        # =========================================================
        # NOVO: SOBREPOSIÇÃO DO PONTO CLASSIFICADO (SEMPRE NO TOPO)
        # =========================================================
        if c_type == "Dispersão" and hasattr(self, 'classified_point') and self.classified_point:
            cx = self.classified_point.get(x_key)
            cy = self.classified_point.get(y_key)
            
            if cx is not None and cy is not None:
                # Estrela Amarela Gigante
                item = self.plot_widget.plot(
                    [cx], [cy], 
                    pen=pg.mkPen(color="#000000", width=1.5), # Borda preta para destacar
                    symbol='star', 
                    symbolBrush=pg.mkBrush(color="#FFFF00"),  # Amarelo vivo
                    symbolSize=22, 
                    name="Ponto Classificado"
                )
                item.setZValue(10) # <-- O MÁGICO: Força o ponto a ser desenhado sobre os outros
                
                item_prev = self.preview_widget.plot(
                    [cx], [cy], 
                    pen=pg.mkPen(color="#000000", width=1), 
                    symbol='star', 
                    symbolBrush=pg.mkBrush(color="#FFFF00"), 
                    symbolSize=12
                )
                item_prev.setZValue(10)

    def mouse_moved(self, pos):
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            self.coord_label.setText(f"Mouse: X={mouse_point.x():.2f}, Y={mouse_point.y():.2f}")
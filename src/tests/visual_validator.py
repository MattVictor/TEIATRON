import numpy as np
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel
import pyqtgraph as pg

# Matplotlib integration
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import LabelEncoder

# TEIATRON PlotEngine
from ..core.plot_engine import PlotEngine

class VisualValidator(QWidget):
    def __init__(self, teiatron_model, sklearn_model, X, y, class_names, feature_names):
        super().__init__()
        self.setWindowTitle("Validação Visual Lado-a-Lado (PyQtGraph vs Matplotlib)")
        self.resize(1000, 600)
        
        self.teiatron_model = teiatron_model
        self.sklearn_model = sklearn_model
        
        self.X = X
        self.y = y
        self.class_names = class_names
        self.feature_names = feature_names
        
        # Assume first 2 features for 2D plot for simplicity, or we slice
        self.x_idx = 0
        self.y_idx = 1
        
        self.init_ui()
        self.draw_plots()
        
    def init_ui(self):
        layout = QHBoxLayout(self)
        
        # Left Side (PyQtGraph)
        left_layout = QVBoxLayout()
        lbl_left = QLabel("TEIATRON (PyQtGraph / Matemática Pura)")
        lbl_left.setStyleSheet("font-weight: bold; font-size: 14px; color: #00E5FF;")
        self.pg_plot = pg.PlotWidget(background='#1E1E1E')
        left_layout.addWidget(lbl_left)
        left_layout.addWidget(self.pg_plot)
        
        # Right Side (Matplotlib)
        right_layout = QVBoxLayout()
        lbl_right = QLabel("Scikit-Learn (Matplotlib / Gabarito)")
        lbl_right.setStyleSheet("font-weight: bold; font-size: 14px; color: #FF5252;")
        self.mpl_canvas = FigureCanvas(Figure(figsize=(5, 5)))
        self.ax = self.mpl_canvas.figure.subplots()
        right_layout.addWidget(lbl_right)
        right_layout.addWidget(self.mpl_canvas)
        
        layout.addLayout(left_layout)
        layout.addLayout(right_layout)

    def draw_plots(self):
        x_data = self.X[:, self.x_idx]
        y_data = self.X[:, self.y_idx]
        x_key = self.feature_names[self.x_idx]
        y_key = self.feature_names[self.y_idx]
        
        # 1. Plot Left (PyQtGraph)
        self.pg_plot.clear()
        self.pg_plot.getAxis('bottom').setLabel(x_key)
        self.pg_plot.getAxis('left').setLabel(y_key)
        
        # Mock dataset structure needed by PlotEngine (fixes other features to their mean)
        dataset_mock = {name: self.X[:, i] for i, name in enumerate(self.feature_names)}
        
        unique_classes = list(sorted(set(self.y)))
        colors = ['#FF5252', '#00E5FF', '#69F0AE', '#FFD740']
        
        # Draw Points
        for i, cls in enumerate(unique_classes):
            idx = np.where(np.array(self.y) == cls)[0]
            brush = pg.mkBrush(color=colors[i % len(colors)])
            self.pg_plot.plot(x_data[idx], y_data[idx], pen=None, symbol='o', symbolBrush=brush, name=cls)
            
        kwargs = {
            'keys': self.feature_names,
            'x_key': x_key,
            'y_key': y_key,
            'dataset': dataset_mock,
            'x_data': x_data,
            'y_data': y_data
        }
        
        # Draw Mathematics
        plot_data = PlotEngine.get_plot_data(self.teiatron_model, **kwargs)
        for pt in plot_data.get("points", []):
            self.pg_plot.plot([pt["x"]], [pt["y"]], pen=None, symbol=pt["symbol"], symbolPen=pg.mkPen(color=pt["color"], width=3), symbolSize=pt["size"])
            
        for ln in plot_data.get("lines", []):
            pen_reta = pg.mkPen(color='#FFD740', width=2, style=pg.QtCore.Qt.PenStyle.DashLine)
            reta = pg.InfiniteLine(pos=ln["pos"], angle=ln["angle"], pen=pen_reta)
            self.pg_plot.addItem(reta)
            
        for ct in plot_data.get("contours", []):
            contour = pg.IsocurveItem(data=ct["Z"], level=ct["level"], pen=pg.mkPen(color='#FFD740', width=2, style=pg.QtCore.Qt.PenStyle.DashLine))
            tr = pg.QtGui.QTransform()
            tr.translate(ct["x_min"], ct["y_min"])
            tr.scale((ct["x_max"] - ct["x_min"]) / ct["res"], (ct["y_max"] - ct["y_min"]) / ct["res"])
            contour.setTransform(tr)
            self.pg_plot.addItem(contour)
            
        # 2. Plot Right (Matplotlib)
        self.ax.clear()
        self.ax.set_xlabel(x_key)
        self.ax.set_ylabel(y_key)
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(self.y)
        
        scatter = self.ax.scatter(x_data, y_data, c=y_encoded, cmap='coolwarm', edgecolors='k')
        
        # Se for um modelo que possui 'predict', o SKLearn suporta desenhar as fronteiras
        if hasattr(self.sklearn_model, "predict"):
            try:
                # O problema é que o sklearn espera N dimensões se treinado com N. 
                # Precisamos de um wrapper que fixa as dimensões nulas na média para plotar a fronteira 2D.
                if self.X.shape[1] > 2:
                    class Wrapper:
                        def __init__(self, model, X_full, x_idx, y_idx):
                            self.model = model
                            self.means = np.mean(X_full, axis=0)
                            self.x_idx = x_idx
                            self.y_idx = y_idx
                            self.classes_ = model.classes_
                        def predict(self, X_2d):
                            X_full = np.tile(self.means, (X_2d.shape[0], 1))
                            X_full[:, self.x_idx] = X_2d[:, 0]
                            X_full[:, self.y_idx] = X_2d[:, 1]
                            return self.model.predict(X_full)
                    model_to_plot = Wrapper(self.sklearn_model, self.X, self.x_idx, self.y_idx)
                else:
                    model_to_plot = self.sklearn_model
                    
                DecisionBoundaryDisplay.from_estimator(
                    model_to_plot, 
                    np.column_stack((x_data, y_data)), 
                    response_method="predict", 
                    alpha=0.4, 
                    ax=self.ax,
                    cmap='coolwarm'
                )
            except Exception as e:
                print(f"[Matplotlib] Não foi possível desenhar a fronteira: {e}")
                
        self.mpl_canvas.draw()

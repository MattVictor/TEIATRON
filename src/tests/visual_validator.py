import numpy as np
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QApplication
import pyqtgraph as pg

# Matplotlib integration
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.plot_engine import PlotEngine

from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

class VisualValidator(QWidget):
    def __init__(self, teiatron_model, sklearn_model, X, y, class_names, feature_names):
        super().__init__()
        self.setWindowTitle("Validação Visual Lado-a-Lado (PyQtGraph vs Matplotlib)")
        self.resize(1100, 600)
        
        self.teiatron_model = teiatron_model
        self.sklearn_model = sklearn_model
        
        self.X = X
        self.y = y
        self.class_names = class_names
        self.feature_names = feature_names
        self.le = LabelEncoder().fit(self.y)
        
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
        
        # Right Side (Matplotlib NATIVE)
        right_layout = QVBoxLayout()
        lbl_right = QLabel("Scikit-Learn (Matplotlib / Gabarito)")
        lbl_right.setStyleSheet("font-weight: bold; font-size: 14px; color: #FF5252;")
        self.mpl_canvas = FigureCanvas(Figure(figsize=(5, 5)))
        self.ax = self.mpl_canvas.figure.subplots()
        self.toolbar = NavigationToolbar(self.mpl_canvas, self) # Toolbar nativa para zoom/pan!
        
        right_layout.addWidget(lbl_right)
        right_layout.addWidget(self.mpl_canvas)
        right_layout.addWidget(self.toolbar)
        
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
        
        dataset_mock = {name: self.X[:, i] for i, name in enumerate(self.feature_names)}
        unique_classes = list(sorted(set(self.y)))
        colors = ['#FF5252', '#00E5FF', '#69F0AE', '#FFD740']
        
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
        
        y_encoded = self.le.transform(self.y)
        scatter = self.ax.scatter(x_data, y_data, c=y_encoded, cmap='coolwarm', edgecolors='k')
        
        if hasattr(self.sklearn_model, "predict"):
            try:
                # 1. Create a mesh grid
                x_min, x_max = x_data.min() - 1, x_data.max() + 1
                y_min, y_max = y_data.min() - 1, y_data.max() + 1
                # Resolution matches pyqtgraph default or typical grid
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                                     np.linspace(y_min, y_max, 200))
                grid_2d = np.c_[xx.ravel(), yy.ravel()]
                
                # 2. Fill missing dimensions with means
                if self.X.shape[1] > 2:
                    means = np.mean(self.X, axis=0)
                    X_full = np.tile(means, (grid_2d.shape[0], 1))
                    X_full[:, self.x_idx] = grid_2d[:, 0]
                    X_full[:, self.y_idx] = grid_2d[:, 1]
                else:
                    X_full = grid_2d
                
                # 3. Predict and encode
                preds = self.sklearn_model.predict(X_full)
                Z = self.le.transform(preds).reshape(xx.shape)
                
                # 4. Draw Background
                self.ax.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
                
                # 5. Draw Lines
                num_classes = len(np.unique(y_encoded))
                c_levels = np.arange(num_classes) + 0.5
                self.ax.contour(xx, yy, Z, levels=c_levels, linewidths=1.5, linestyles='dashed', colors='k')
                
            except Exception as e:
                print(f"[Matplotlib] Não foi possível desenhar a fronteira: {e}")
                import traceback
                traceback.print_exc()
                
        self.mpl_canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    setup_pyqtgraph() 
    window = VisualValidator()
    window.show()
    sys.exit(app.exec())
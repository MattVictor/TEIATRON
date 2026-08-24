import random
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsTextItem, QToolTip
from PyQt6.QtGui import QPen, QBrush, QColor, QFont
from PyQt6.QtCore import Qt

COLOR_BORDER = QColor("#008B7D")  
COLOR_INPUT = QColor("#F0F0F0")   
COLOR_HIDDEN = QColor("#FCE4E4")  
COLOR_OUTPUT = QColor("#FFF2CC")  
COLOR_EDGE = QColor("#BDBDBD")    
COLOR_EDGE_HOVER = QColor("#FF0000") 

class Edge(QGraphicsLineItem):
    def __init__(self, source_node, target_node, weight):
        super().__init__()
        self.source_node = source_node
        self.target_node = target_node
        self.weight = weight
        
        self.setAcceptHoverEvents(True)
        self.setZValue(-1) 
        
        self.text_item = QGraphicsTextItem("", self)
        self.text_item.setDefaultTextColor(QColor("#CCCCCC"))
        self.text_item.setFont(QFont("Arial", 8, QFont.Weight.Bold))
        
        self.update_position()
        self.update_appearance()

    def update_position(self):
        self.setLine(self.source_node.x(), self.source_node.y(), 
                     self.target_node.x(), self.target_node.y())
        center_x = (self.source_node.x() + self.target_node.x()) / 2
        center_y = (self.source_node.y() + self.target_node.y()) / 2
        self.text_item.setPos(center_x - 15, center_y - 10)

    def update_appearance(self):
        thickness = 1 + abs(self.weight) * 3
        if thickness > 6: thickness = 6
        self.setPen(QPen(COLOR_EDGE, int(thickness)))
        self.text_item.setPlainText(f"{self.weight:.2f}")

    def hoverEnterEvent(self, event):
        self.setPen(QPen(COLOR_EDGE_HOVER, 4))
        QToolTip.showText(event.screenPos(), f"Peso exato: {self.weight:.6f}")
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.update_appearance()
        QToolTip.hideText() 
        super().hoverLeaveEvent(event)


class Node(QGraphicsEllipseItem):
    def __init__(self, name, layer_type, x, y, activation=0.0, bias=0.0):
        super().__init__(-35, -35, 70, 70)
        self.name = name
        self.layer_type = layer_type
        self.activation = activation
        self.bias = bias
        
        self.setPos(x, y)
        self.setAcceptHoverEvents(True)
        self.setZValue(1) 

        self.setPen(QPen(COLOR_BORDER, 2))
        if layer_type == "Entrada":
            self.setBrush(QBrush(COLOR_INPUT))
        elif layer_type == "Oculta":
            self.setBrush(QBrush(COLOR_HIDDEN))
        else:
            self.setBrush(QBrush(COLOR_OUTPUT))

        self.text_item = QGraphicsTextItem("", self)
        self.text_item.setDefaultTextColor(QColor("#333333"))
        self.text_item.setFont(QFont("Arial", 8, QFont.Weight.Bold))
        self.update_display()

    def update_display(self):
        label = f"{self.name}\nA: {self.activation:.2f}"
        if self.layer_type != "Entrada":
            label += f"\nB: {self.bias:.2f}"
            
        self.text_item.setPlainText(label)
        text_rect = self.text_item.boundingRect()
        self.text_item.setPos(-text_rect.width() / 2, -text_rect.height() / 2)

    def hoverEnterEvent(self, event):
        self.setPen(QPen(COLOR_BORDER, 4)) 
        bias_text = f"\nBias: {self.bias:.6f}" if self.layer_type != "Entrada" else ""
        QToolTip.showText(event.screenPos(), f"Nó: {self.name}\nAtivação: {self.activation:.6f}{bias_text}")
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setPen(QPen(COLOR_BORDER, 2)) 
        QToolTip.hideText()
        super().hoverLeaveEvent(event)


class NNDiagramWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(self.view.renderHints() | self.view.renderHints().Antialiasing) 
        self.view.setStyleSheet("background-color: #1a1a1a; border: none;")
        layout.addWidget(self.view)

    def add_layer_label(self, text, x, y):
        label = QGraphicsTextItem(text)
        label.setDefaultTextColor(COLOR_BORDER)
        label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        rect = label.boundingRect()
        label.setPos(x - rect.width() / 2, y)
        self.scene.addItem(label)

    def build_network(self, model):
        self.scene.clear()
        
        if not hasattr(model, 'weights') or not model.weights:
            return
            
        weights = model.weights
        biases = model.biases
        last_pass = getattr(model, 'last_forward_pass', {})
        activations = last_pass.get("activations", [])
        
        num_layers = len(weights) + 1
        if num_layers < 2: return
        
        # Calculate horizontal spacing
        x_start = -400
        x_end = 400
        x_step = (x_end - x_start) / (num_layers - 1) if num_layers > 1 else 0
        
        y_gap = 90
        def get_y_coords(count):
            total_height = (count - 1) * y_gap
            start_y = -total_height / 2
            return [start_y + i * y_gap for i in range(count)]
            
        layer_nodes = []
        
        for L in range(num_layers):
            if L == 0:
                n_nodes = len(weights[0][0])
                layer_type = "Entrada"
                label_text = "Entrada"
            elif L == num_layers - 1:
                n_nodes = len(biases[-1])
                layer_type = "Saída"
                label_text = "Saída"
            else:
                n_nodes = len(biases[L-1])
                layer_type = "Oculta"
                label_text = f"Oculta {L}"
                
            x_pos = x_start + L * x_step
            y_coords = get_y_coords(n_nodes)
            
            top_y = min(y_coords) - 80 if y_coords else -80
            self.add_layer_label(label_text, x_pos, top_y)
            
            nodes_in_layer = []
            for i in range(n_nodes):
                act = activations[L][i] if L < len(activations) and i < len(activations[L]) else 0.0
                bias = biases[L-1][i] if L > 0 else 0.0
                
                if layer_type == "Entrada":
                    name = f"In {i+1}"
                elif layer_type == "Saída":
                    name = f"Out {i+1}"
                    if hasattr(model, 'reverse_map') and i in model.reverse_map:
                        name = str(model.reverse_map[i])[:8]
                else:
                    name = f"H{L} {i+1}"
                    
                node = Node(name, layer_type, x_pos, y_coords[i], activation=act, bias=bias)
                nodes_in_layer.append(node)
                self.scene.addItem(node)
                
            layer_nodes.append(nodes_in_layer)
            
        # Draw edges
        for L in range(len(weights)):
            w_layer = weights[L]
            source_nodes = layer_nodes[L]
            target_nodes = layer_nodes[L+1]
            
            for j in range(len(target_nodes)):
                for k in range(len(source_nodes)):
                    w = w_layer[j][k]
                    edge = Edge(source_nodes[k], target_nodes[j], w)
                    self.scene.addItem(edge)

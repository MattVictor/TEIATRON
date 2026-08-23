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
        
        if not hasattr(model, 'W_ih') or not model.W_ih:
            return
            
        W_ih = model.W_ih
        b_h = model.b_h
        W_ho = model.W_ho
        b_o = model.b_o
        
        last_pass = getattr(model, 'last_forward_pass', {})
        in_acts = last_pass.get("inputs", [])
        h_acts = last_pass.get("hidden", [])
        o_acts = last_pass.get("output", [])
        
        n_inputs = len(W_ih[0]) if len(W_ih) > 0 else 0
        n_hidden = len(b_h)
        n_outputs = len(b_o)
        
        if n_inputs == 0: return

        x_in, x_hid, x_out = -300, 0, 300
        y_gap = 90
        
        def get_y_coords(count):
            total_height = (count - 1) * y_gap
            start_y = -total_height / 2
            return [start_y + i * y_gap for i in range(count)]
            
        y_in = get_y_coords(n_inputs)
        y_hid = get_y_coords(n_hidden)
        y_out = get_y_coords(n_outputs)
        
        top_y = min(min(y_in) if y_in else [0], min(y_hid) if y_hid else [0], min(y_out) if y_out else [0]) - 80
        self.add_layer_label("Entrada", x_in, top_y)
        self.add_layer_label("Oculta", x_hid, top_y)
        self.add_layer_label("Saída", x_out, top_y)
        
        input_nodes = []
        for i in range(n_inputs):
            act = in_acts[i] if i < len(in_acts) else 0.0
            node = Node(f"In {i+1}", "Entrada", x_in, y_in[i], activation=act)
            input_nodes.append(node)
            self.scene.addItem(node)
            
        hidden_nodes = []
        for j in range(n_hidden):
            act = h_acts[j] if j < len(h_acts) else 0.0
            bias = b_h[j]
            node = Node(f"H {j+1}", "Oculta", x_hid, y_hid[j], activation=act, bias=bias)
            hidden_nodes.append(node)
            self.scene.addItem(node)
            
        output_nodes = []
        for k in range(n_outputs):
            act = o_acts[k] if k < len(o_acts) else 0.0
            bias = b_o[k]
            name = f"Out {k+1}"
            if hasattr(model, 'reverse_map') and k in model.reverse_map:
                name = str(model.reverse_map[k])[:8] 
            node = Node(name, "Saída", x_out, y_out[k], activation=act, bias=bias)
            output_nodes.append(node)
            self.scene.addItem(node)
            
        for j in range(n_hidden):
            for i in range(n_inputs):
                w = W_ih[j][i]
                edge = Edge(input_nodes[i], hidden_nodes[j], w)
                self.scene.addItem(edge)
                
        for k in range(n_outputs):
            for j in range(n_hidden):
                w = W_ho[k][j]
                edge = Edge(hidden_nodes[j], output_nodes[k], w)
                self.scene.addItem(edge)

# import random
# import math
# import matplotlib.pyplot as plt

# # 1. Funções de Ativação e Auxiliares
# def sigmoid(x):
#     # Prevenção de overflow no math.exp
#     if x < -700:
#         return 0.0
#     return 1 / (1 + math.exp(-x))

# def sigmoid_derivative(out):
#     # A derivada da sigmoide em relação à sua própria saída
#     return out * (1 - out)

# # 2. Configuração Inicial da Rede Neural (2 entradas, 2 ocultos, 2 saídas)
# # Pesos inicializados aleatoriamente entre -1 e 1
# W_ih = [[random.uniform(-1, 1) for _ in range(2)] for _ in range(2)] # Pesos Entrada -> Oculta
# b_h = [random.uniform(-1, 1) for _ in range(2)]                      # Bias da Oculta

# W_ho = [[random.uniform(-1, 1) for _ in range(2)] for _ in range(2)] # Pesos Oculta -> Saída
# b_o = [random.uniform(-1, 1) for _ in range(2)]                      # Bias da Saída

# # 3. Base de Dados do Problema XOR
# X = [[0, 0], [0, 1], [1, 0], [1, 1]]

# # Como a rede tem 2 neurônios de saída (o1, o2), replicamos o alvo do XOR para ambos.
# # Alvos: [A XOR B, A XOR B]
# Y = [[0, 0], [1, 1], [1, 1], [0, 0]]

# # 4. Parâmetros de Treinamento
# epocas = 10000
# historico_erro = []

# # Solicitação do Learning Rate ao usuário
# try:
#     # Tratamento caso o script seja rodado em ambientes sem console interativo
#     learning_rate = 0.5
#     print(f"Console interativo indisponível. Usando Learning Rate padrão: {learning_rate}")
# except ValueError:
#     learning_rate = 0.5
#     print(f"Valor inválido. Usando Learning Rate padrão: {learning_rate}")

# print(f"\nIniciando treinamento com Learning Rate = {learning_rate} por {epocas} épocas...")

# # 5. Loop de Treinamento
# for epoca in range(epocas):
#     erro_total_epoca = 0
    
#     for i in range(len(X)):
#         entrada = X[i]
#         alvo = Y[i]
        
#         # --- FORWARD PASS (Ativação) ---
        
#         # Camada Oculta
#         net_h = [0.0, 0.0]
#         out_h = [0.0, 0.0]
#         for j in range(2): # Para cada neurônio oculto
#             net_h[j] = W_ih[j][0] * entrada[0] + W_ih[j][1] * entrada[1] + b_h[j]
#             out_h[j] = sigmoid(net_h[j])
            
#         # Camada de Saída
#         net_o = [0.0, 0.0]
#         out_o = [0.0, 0.0]
#         for j in range(2): # Para cada neurônio de saída
#             net_o[j] = W_ho[j][0] * out_h[0] + W_ho[j][1] * out_h[1] + b_o[j]
#             out_o[j] = sigmoid(net_o[j])
            
#         # Cálculo do Erro Quadrático Médio (MSE) para o padrão atual
#         erro_padrao = 0.5 * sum((alvo[k] - out_o[k])**2 for k in range(2))
#         erro_total_epoca += erro_padrao
        
#         # --- BACKWARD PASS (Retropropagação) ---
        
#         # 1. Calcular o Termo de Erro (Delta) da Camada de Saída
#         delta_o = [0.0, 0.0]
#         for j in range(2):
#             erro = alvo[j] - out_o[j]
#             delta_o[j] = erro * sigmoid_derivative(out_o[j])
            
#         # 2. Calcular o Termo de Erro (Delta) da Camada Oculta
#         delta_h = [0.0, 0.0]
#         for j in range(2):
#             # Soma os erros retropropagados dos neurônios de saída
#             erro_retropropagado = delta_o[0] * W_ho[0][j] + delta_o[1] * W_ho[1][j]
#             delta_h[j] = erro_retropropagado * sigmoid_derivative(out_h[j])
            
#         # 3. Atualizar Pesos e Bias (Oculta -> Saída)
#         for j in range(2): # Neurônio de saída
#             for k in range(2): # Conexão vinda do neurônio oculto
#                 W_ho[j][k] += learning_rate * delta_o[j] * out_h[k]
#             b_o[j] += learning_rate * delta_o[j]
            
#         # 4. Atualizar Pesos e Bias (Entrada -> Oculta)
#         for j in range(2): # Neurônio oculto
#             for k in range(2): # Conexão vinda da entrada
#                 W_ih[j][k] += learning_rate * delta_h[j] * entrada[k]
#             b_h[j] += learning_rate * delta_h[j]
            
#     # Guardar erro médio da época
#     historico_erro.append(erro_total_epoca / len(X))

# print("Treinamento finalizado!")

# # 6. Teste da Rede após o treinamento
# print("\n--- Resultados após o Treinamento ---")
# for i in range(len(X)):
#     entrada = X[i]
#     # Forward Pass rápido para teste
#     out_h = [sigmoid(W_ih[j][0]*entrada[0] + W_ih[j][1]*entrada[1] + b_h[j]) for j in range(2)]
#     out_o = [sigmoid(W_ho[j][0]*out_h[0] + W_ho[j][1]*out_h[1] + b_o[j]) for j in range(2)]
    
#     print(f"Entrada {entrada} -> Saídas: o1={out_o[0]:.4f}, o2={out_o[1]:.4f} (Alvo: {Y[i][0]})")

# # 7. Plotar a Curva de Erro
# plt.figure(figsize=(10, 6))
# plt.plot(historico_erro, color='blue', linewidth=2)
# plt.title(f'Curva de Aprendizado - Problema XOR (Learning Rate: {learning_rate})')
# plt.xlabel('Épocas')
# plt.ylabel('Erro Quadrático Médio (MSE)')
# plt.grid(True)
# plt.show()

import sys
import random
import math
from PyQt6.QtWidgets import (QApplication, QMainWindow, QGraphicsView, 
                             QGraphicsScene, QGraphicsEllipseItem, QGraphicsLineItem, 
                             QGraphicsTextItem, QVBoxLayout, QHBoxLayout, QWidget, 
                             QPushButton, QToolTip, QTextEdit, QSplitter)
from PyQt6.QtGui import QPen, QBrush, QColor, QFont
from PyQt6.QtCore import Qt

# Importações do Matplotlib para integração com PyQt6
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- Configuração de Cores ---
COLOR_BORDER = QColor("#008B7D")  
COLOR_INPUT = QColor("#F0F0F0")   
COLOR_HIDDEN = QColor("#FCE4E4")  
COLOR_OUTPUT = QColor("#FFF2CC")  
COLOR_EDGE = QColor("#BDBDBD")    
COLOR_EDGE_HOVER = QColor("#FF0000") 

class Edge(QGraphicsLineItem):
    def __init__(self, source_node, target_node):
        super().__init__()
        self.source_node = source_node
        self.target_node = target_node
        self.weight = random.uniform(-1.0, 1.0)
        
        self.setAcceptHoverEvents(True)
        self.setZValue(-1) 
        
        self.text_item = QGraphicsTextItem("", self)
        self.text_item.setDefaultTextColor(QColor("#333333"))
        self.text_item.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        
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
        self.setPen(QPen(COLOR_EDGE, thickness))
        self.text_item.setPlainText(f"W: {self.weight:.2f}")

    def randomize_weight(self):
        self.weight = random.uniform(-1.0, 1.0)
        self.update_appearance()

    def hoverEnterEvent(self, event):
        self.setPen(QPen(COLOR_EDGE_HOVER, 3))
        QToolTip.showText(event.screenPos(), f"Peso exato: {self.weight:.6f}")
        super().hoverEnterEvent(event)

    def hoverMoveEvent(self, event):
        QToolTip.showText(event.screenPos(), f"Peso exato: {self.weight:.6f}")
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event):
        self.update_appearance()
        QToolTip.hideText() 
        super().hoverLeaveEvent(event)


class Node(QGraphicsEllipseItem):
    def __init__(self, name, layer_type, x, y):
        super().__init__(-45, -45, 90, 90)
        self.name = name
        self.layer_type = layer_type
        
        self.activation = random.uniform(0.0, 1.0)
        self.bias = random.uniform(-1.0, 1.0)
        
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
        self.text_item.setDefaultTextColor(COLOR_BORDER)
        self.text_item.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        self.update_display()

    def update_display(self):
        label = f"{self.name}\nA: {self.activation:.2f}\nB: {self.bias:.2f}"
        if self.layer_type == "Entrada":
            label = f"{self.name}\nA: {self.activation:.0f}"
            
        self.text_item.setPlainText(label)
        text_rect = self.text_item.boundingRect()
        self.text_item.setPos(-text_rect.width() / 2, -text_rect.height() / 2)

    def randomize_values(self):
        self.activation = random.uniform(0.0, 1.0)
        if self.layer_type != "Entrada":
            self.bias = random.uniform(-1.0, 1.0)
        self.update_display()

    def hoverEnterEvent(self, event):
        self.setPen(QPen(COLOR_BORDER, 4)) 
        QToolTip.showText(event.screenPos(), f"Nó: {self.name}\nAtivação: {self.activation:.6f}\nBias: {self.bias:.6f}")
        super().hoverEnterEvent(event)

    def hoverMoveEvent(self, event):
        QToolTip.showText(event.screenPos(), f"Nó: {self.name}\nAtivação: {self.activation:.6f}\nBias: {self.bias:.6f}")
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event):
        self.setPen(QPen(COLOR_BORDER, 2)) 
        QToolTip.hideText()
        super().hoverLeaveEvent(event)


class NeuralNetworkApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rede Neural Interativa (XOR) - PyQt6 com Dashboard")
        self.resize(1200, 700) # Janela redimensionada para acomodar o painel

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Layout Principal (Esquerda: Rede, Direita: Controles e Gráficos)
        main_layout = QHBoxLayout(main_widget)

        # --- Esquerda: Visão da Rede Neural ---
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(self.view.renderHints() | self.view.renderHints().Antialiasing) 
        main_layout.addWidget(self.view, stretch=2)

        # --- Direita: Painel de Controle ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        main_layout.addWidget(right_panel, stretch=1)

        # Botões
        self.btn_randomize = QPushButton("Aleatorizar Valores (Reset)")
        self.btn_randomize.clicked.connect(self.randomize_network)
        self.btn_randomize.setStyleSheet("padding: 10px; font-size: 14px;")
        right_layout.addWidget(self.btn_randomize)

        self.btn_train = QPushButton("Treinar Modelo (10.000 Épocas)")
        self.btn_train.clicked.connect(self.train_model)
        self.btn_train.setStyleSheet("padding: 10px; font-size: 14px; font-weight: bold;")
        right_layout.addWidget(self.btn_train)

        # Log de Texto
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet("background-color: #000000; font-family: monospace; font-size: 12px;")
        right_layout.addWidget(self.log_console, stretch=1)

        # Canvas do Matplotlib para o Gráfico
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        right_layout.addWidget(self.canvas, stretch=2)

        # Inicialização dos dicionários e construção da rede
        self.nodes = []
        self.edges = []
        self.node_dict = {}
        self.edge_dict = {}
        self.build_network()

        # Log Inicial
        self.log_console.append("Pronto para iniciar.\nClique em 'Treinar Modelo' para resolver o XOR.")

    def build_network(self):
        x_input, x_hidden, x_output = -200, 0, 200
        y_top, y_bottom = -120, 120

        self.add_layer_label("Entrada", x_input, -210)
        self.add_layer_label("Oculta", x_hidden, -210)
        self.add_layer_label("Saída", x_output, -210)

        i1 = Node("i1", "Entrada", x_input, y_top)
        i2 = Node("i2", "Entrada", x_input, y_bottom)
        
        h1 = Node("h1", "Oculta", x_hidden, y_top)
        h2 = Node("h2", "Oculta", x_hidden, y_bottom)
        
        o1 = Node("o1", "Saída", x_output, y_top)
        o2 = Node("o2", "Saída", x_output, y_bottom)

        self.nodes.extend([i1, i2, h1, h2, o1, o2])
        self.node_dict = {n.name: n for n in self.nodes}
        
        for node in self.nodes:
            self.scene.addItem(node)

        connections = [
            (i1, h1), (i1, h2),
            (i2, h1), (i2, h2),
            (h1, o1), (h1, o2),
            (h2, o1), (h2, o2)
        ]

        for src, tgt in connections:
            edge = Edge(src, tgt)
            self.edges.append(edge)
            self.edge_dict[(src.name, tgt.name)] = edge
            self.scene.addItem(edge)

    def add_layer_label(self, text, x, y):
        label = QGraphicsTextItem(text)
        label.setDefaultTextColor(COLOR_BORDER)
        label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        rect = label.boundingRect()
        label.setPos(x - rect.width() / 2, y)
        self.scene.addItem(label)

    def randomize_network(self):
        for node in self.nodes:
            node.randomize_values()
        for edge in self.edges:
            edge.randomize_weight()
        self.scene.update()
        self.log_console.append("\nValores da rede foram resetados (aleatorizados).")

    def sigmoid(self, x):
        if x < -700: return 0.0
        return 1 / (1 + math.exp(-x))

    def train_model(self):
        # Base do XOR
        X = [[0, 0], [0, 1], [1, 0], [1, 1]]
        Y = [[0, 0], [1, 1], [1, 1], [0, 0]]
        
        learning_rate = 0.5
        epocas = 10000
        historico_erro = []
        
        nd = self.node_dict
        ed = self.edge_dict
        
        # Reinicia os pesos antes de iniciar um novo treinamento de 10.000 épocas
        self.randomize_network()
        
        self.log_console.clear()
        self.log_console.append(f"Iniciando treinamento...\nLearning Rate: {learning_rate}\nÉpocas: {epocas}\n")
        
        # Atualiza a interface para mostrar que o processamento começou (força o PyQt a desenhar)
        QApplication.processEvents()
        
        for epoca in range(epocas):
            erro_epoca = 0
            
            for i in range(len(X)):
                entrada = X[i]
                alvo = Y[i]
                
                # --- FORWARD PASS ---
                # A ativação dos nós de entrada são os próprios valores da base (0 ou 1)
                nd['i1'].activation = entrada[0]
                nd['i2'].activation = entrada[1]
                
                net_h1 = nd['i1'].activation * ed[('i1', 'h1')].weight + nd['i2'].activation * ed[('i2', 'h1')].weight + nd['h1'].bias
                nd['h1'].activation = self.sigmoid(net_h1)
                
                net_h2 = nd['i1'].activation * ed[('i1', 'h2')].weight + nd['i2'].activation * ed[('i2', 'h2')].weight + nd['h2'].bias
                nd['h2'].activation = self.sigmoid(net_h2)
                
                net_o1 = nd['h1'].activation * ed[('h1', 'o1')].weight + nd['h2'].activation * ed[('h2', 'o1')].weight + nd['o1'].bias
                nd['o1'].activation = self.sigmoid(net_o1)
                
                net_o2 = nd['h1'].activation * ed[('h1', 'o2')].weight + nd['h2'].activation * ed[('h2', 'o2')].weight + nd['o2'].bias
                nd['o2'].activation = self.sigmoid(net_o2)
                
                # Calculo do erro para o gráfico (MSE)
                erro_padrao = 0.5 * ((alvo[0] - nd['o1'].activation)**2 + (alvo[1] - nd['o2'].activation)**2)
                erro_epoca += erro_padrao
                
                # --- BACKWARD PASS ---
                delta_o1 = (alvo[0] - nd['o1'].activation) * nd['o1'].activation * (1 - nd['o1'].activation)
                delta_o2 = (alvo[1] - nd['o2'].activation) * nd['o2'].activation * (1 - nd['o2'].activation)
                
                delta_h1 = (delta_o1 * ed[('h1', 'o1')].weight + delta_o2 * ed[('h1', 'o2')].weight) * nd['h1'].activation * (1 - nd['h1'].activation)
                delta_h2 = (delta_o1 * ed[('h2', 'o1')].weight + delta_o2 * ed[('h2', 'o2')].weight) * nd['h2'].activation * (1 - nd['h2'].activation)
                
                # Atualizando Pesos
                ed[('h1', 'o1')].weight += learning_rate * delta_o1 * nd['h1'].activation
                ed[('h2', 'o1')].weight += learning_rate * delta_o1 * nd['h2'].activation
                ed[('h1', 'o2')].weight += learning_rate * delta_o2 * nd['h1'].activation
                ed[('h2', 'o2')].weight += learning_rate * delta_o2 * nd['h2'].activation
                
                ed[('i1', 'h1')].weight += learning_rate * delta_h1 * nd['i1'].activation
                ed[('i2', 'h1')].weight += learning_rate * delta_h1 * nd['i2'].activation
                ed[('i1', 'h2')].weight += learning_rate * delta_h2 * nd['i1'].activation
                ed[('i2', 'h2')].weight += learning_rate * delta_h2 * nd['i2'].activation
                
                # Atualizando Bias
                nd['o1'].bias += learning_rate * delta_o1
                nd['o2'].bias += learning_rate * delta_o2
                nd['h1'].bias += learning_rate * delta_h1
                nd['h2'].bias += learning_rate * delta_h2

            # Média do erro da época (dividido pelos 4 padrões do XOR)
            historico_erro.append(erro_epoca / 4)

        # Plotar o gráfico no Canvas do Matplotlib
        self.ax.clear()
        self.ax.plot(historico_erro, color='blue', linewidth=2)
        self.ax.set_title('Curva de Aprendizado (Erro vs Época)')
        self.ax.set_xlabel('Épocas')
        self.ax.set_ylabel('MSE')
        self.ax.grid(True)
        self.canvas.draw()

        # Adicionar os testes finais no console de Log
        self.log_console.append("Treinamento finalizado!\n")
        self.log_console.append("--- Resultados (Forward Pass Test) ---")
        
        for i in range(len(X)):
            entrada = X[i]
            
            # Forward Pass de teste com a rede já treinada
            out_h1 = self.sigmoid(entrada[0] * ed[('i1', 'h1')].weight + entrada[1] * ed[('i2', 'h1')].weight + nd['h1'].bias)
            out_h2 = self.sigmoid(entrada[0] * ed[('i1', 'h2')].weight + entrada[1] * ed[('i2', 'h2')].weight + nd['h2'].bias)
            out_o1 = self.sigmoid(out_h1 * ed[('h1', 'o1')].weight + out_h2 * ed[('h2', 'o1')].weight + nd['o1'].bias)
            out_o2 = self.sigmoid(out_h1 * ed[('h1', 'o2')].weight + out_h2 * ed[('h2', 'o2')].weight + nd['o2'].bias)
            
            self.log_console.append(f"In {entrada} -> Saídas: o1={out_o1:.4f}, o2={out_o2:.4f} (Alvo: {Y[i][0]})")

        # Mantém na interface visual a última passagem feita para demonstração (geralmente a entrada 1, 1 do XOR)
        for node in self.nodes:
            node.update_display()
        for edge in self.edges:
            edge.update_appearance()
        self.scene.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NeuralNetworkApp()
    window.show()
    sys.exit(app.exec())
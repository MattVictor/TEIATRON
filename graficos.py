import pyqtgraph as pg

# O pyqtgraph utiliza o Qt (PyQt5, PyQt6 ou PySide) por baixo dos panos.
# A importação das funções de matemática do seu módulo continua igual:
from classificadores import calcular_centroide, produto_escalar

class GraficosClassificadores:
    _app = None
    _janelas = [] # Guarda as referências das janelas para que não sumam (garbage collection)

    @staticmethod
    def plotar_superficie_2d(modelo_superficie, X_treino, y_treino, X_teste, y_teste, titulo, idx_f1=0, idx_f2=2):
        """
        Plota a superfície de decisão usando PyQtGraph.
        """
        # Inicializa a aplicação Qt apenas uma vez
        if GraficosClassificadores._app is None:
            GraficosClassificadores._app = pg.mkQApp("Sistema de Reconhecimento de Padrões")
            
        # Cria a janela principal para este gráfico
        win = pg.GraphicsLayoutWidget(show=True, title=titulo)
        win.resize(800, 600)
        GraficosClassificadores._janelas.append(win)
        
        # Adiciona o Plot à janela
        plot = win.addPlot(title=titulo)
        plot.addLegend(offset=(20, 20)) # Adiciona a legenda no canto
        plot.showGrid(x=True, y=True, alpha=0.5)
        
        # GARANTE A PERPENDICULARIDADE: Equivalente ao plt.axis('equal')
        plot.setAspectLocked(True) 
        
        plot.setLabel('bottom', 'Comprimento da Sépala (Atributo 1)')
        plot.setLabel('left', 'Comprimento da Pétala (Atributo 3)')
        
        classe_A = modelo_superficie.classe_A
        classe_B = modelo_superficie.classe_B
        
        # --- 1. SEPARAR E PROJETAR DADOS EM 2D ---
        X_tr_A = [[x[idx_f1], x[idx_f2]] for x, rotulo in zip(X_treino, y_treino) if rotulo == classe_A]
        X_tr_B = [[x[idx_f1], x[idx_f2]] for x, rotulo in zip(X_treino, y_treino) if rotulo == classe_B]
        
        X_te_A = [[x[idx_f1], x[idx_f2]] for x, rotulo in zip(X_teste, y_teste) if rotulo == classe_A]
        X_te_B = [[x[idx_f1], x[idx_f2]] for x, rotulo in zip(X_teste, y_teste) if rotulo == classe_B]
        
        # --- 2. PLOTAR DADOS DE TREINO (Bolinhas 'o') ---
        if X_tr_A:
            scatter_tr_A = pg.ScatterPlotItem(
                x=[x[0] for x in X_tr_A], y=[x[1] for x in X_tr_A],
                size=10, pen=pg.mkPen(None), brush=pg.mkBrush(0, 100, 255, 120), # Azul transparente
                symbol='o', name=f'Treino: {classe_A}'
            )
            plot.addItem(scatter_tr_A)
            
        if X_tr_B:
            scatter_tr_B = pg.ScatterPlotItem(
                x=[x[0] for x in X_tr_B], y=[x[1] for x in X_tr_B],
                size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 50, 50, 120), # Vermelho transparente
                symbol='o', name=f'Treino: {classe_B}'
            )
            plot.addItem(scatter_tr_B)
            
        # --- 3. PLOTAR DADOS DE TESTE (Triângulos 't') ---
        if X_te_A:
            scatter_te_A = pg.ScatterPlotItem(
                x=[x[0] for x in X_te_A], y=[x[1] for x in X_te_A],
                size=12, pen=pg.mkPen('w', width=1), brush=pg.mkBrush(0, 100, 255, 255), # Azul sólido
                symbol='t', name=f'Teste: {classe_A}'
            )
            plot.addItem(scatter_te_A)
            
        if X_te_B:
            scatter_te_B = pg.ScatterPlotItem(
                x=[x[0] for x in X_te_B], y=[x[1] for x in X_te_B],
                size=12, pen=pg.mkPen('w', width=1), brush=pg.mkBrush(255, 50, 50, 255), # Vermelho sólido
                symbol='t', name=f'Teste: {classe_B}'
            )
            plot.addItem(scatter_te_B)
            
        # --- 4. CALCULAR E PLOTAR CENTRÓIDES (Marcador 'x') ---
        mA = calcular_centroide(X_tr_A)
        mB = calcular_centroide(X_tr_B)
        
        scatter_cen_A = pg.ScatterPlotItem(
            x=[mA[0]], y=[mA[1]],
            size=20, pen=pg.mkPen('w', width=3), brush=pg.mkBrush(0, 255, 255, 255), # Ciano
            symbol='x', name=f'Centróide: {classe_A}'
        )
        plot.addItem(scatter_cen_A)
        
        scatter_cen_B = pg.ScatterPlotItem(
            x=[mB[0]], y=[mB[1]],
            size=20, pen=pg.mkPen('w', width=3), brush=pg.mkBrush(255, 255, 0, 255), # Amarelo
            symbol='x', name=f'Centróide: {classe_B}'
        )
        plot.addItem(scatter_cen_B)
        
        # --- 5. CALCULAR E PLOTAR SUPERFÍCIE DE DECISÃO ---
        w = [a - b for a, b in zip(mA, mB)]
        bias = -0.5 * (produto_escalar(mA, mA) - produto_escalar(mB, mB))
        
        todos_x1 = [x[0] for x in X_tr_A + X_tr_B + X_te_A + X_te_B]
        x1_min, x1_max = min(todos_x1) - 0.5, max(todos_x1) + 0.5
        x_vals = [x1_min, x1_max]
        
        if w[1] != 0:
            y_vals = [-(w[0] * x + bias) / w[1] for x in x_vals]
            
            # Formatação da Equação Matemática exata (Ex: 0.52*x1 - 1.20*x2 + 0.35 = 0)
            sinal_w2 = "+" if w[1] >= 0 else "-"
            sinal_bias = "+" if bias >= 0 else "-"
            # Criando a string com formatação limpa de duas casas decimais
            equacao_str = f"Decisão: {w[0]:.2f}*x1 {sinal_w2} {abs(w[1]):.2f}*x2 {sinal_bias} {abs(bias):.2f} = 0"
            
            linha_decisao = pg.PlotDataItem(
                x_vals, y_vals, 
                pen=pg.mkPen('y', width=2, dash=[4, 4]), # Linha amarela tracejada (Universal)
                name=equacao_str # Atribui a equação gerada como título da linha na legenda!
            )
            plot.addItem(linha_decisao)

    @staticmethod
    def mostrar_todos():
        """
        Função obrigatória no PyQtGraph para iniciar o loop de execução.
        Isso impede que as janelas fechem assim que o script terminar.
        """
        if GraficosClassificadores._app is not None:
            pg.exec()
            
    @staticmethod
    def plotar_nuvem_todas_classes(dataset_completo, titulo="Item A: Dispersão de Todas as Classes (X1 e X2)", idx_f1=0, idx_f2=1):
        if GraficosClassificadores._app is None:
            GraficosClassificadores._app = pg.mkQApp("Sistema de Reconhecimento")
            
        win = pg.GraphicsLayoutWidget(show=True, title=titulo)
        win.resize(800, 600)
        GraficosClassificadores._janelas.append(win)
        
        plot = win.addPlot(title=titulo)
        plot.addLegend(offset=(20, 20))
        plot.showGrid(x=True, y=True, alpha=0.5)
        plot.setAspectLocked(True)
        plot.setLabel('bottom', 'X1 (Comprimento da Sépala)')
        plot.setLabel('left', 'X2 (Largura da Sépala)')
        
        cores = {'setosa': (0, 255, 255), 'versicolor': (255, 255, 0), 'virginica': (255, 0, 255)}
        
        for classe_alvo, cor in cores.items():
            X1 = [linha[0][idx_f1] for linha in dataset_completo if linha[1] == classe_alvo]
            X2 = [linha[0][idx_f2] for linha in dataset_completo if linha[1] == classe_alvo]
            
            scatter = pg.ScatterPlotItem(
                x=X1, y=X2, size=12, pen=pg.mkPen('k', width=1), 
                brush=pg.mkBrush(*cor, 200), symbol='o', name=classe_alvo.capitalize()
            )
            plot.addItem(scatter)

    @staticmethod
    def plotar_superficie_perceptron(modelo, X_treino, y_treino, titulo, idx_f1=0, idx_f2=1):
        # Semelhante à função anterior, mas adaptada para o Perceptron
        if GraficosClassificadores._app is None:
            GraficosClassificadores._app = pg.mkQApp("Sistema de Reconhecimento")
            
        win = pg.GraphicsLayoutWidget(show=True, title=titulo)
        win.resize(800, 600)
        GraficosClassificadores._janelas.append(win)
        
        plot = win.addPlot(title=titulo)
        plot.addLegend()
        plot.showGrid(x=True, y=True, alpha=0.5)
        plot.setAspectLocked(True)
        plot.setLabel('bottom', 'X1')
        plot.setLabel('left', 'X2')
        
        # Filtra os dados de treino pelas classes originais 1 e 0
        X_pos = [[x[idx_f1], x[idx_f2]] for x, y in zip(X_treino, y_treino) if y == 1]
        X_neg = [[x[idx_f1], x[idx_f2]] for x, y in zip(X_treino, y_treino) if y == 0]
        
        if X_pos:
            plot.addItem(pg.ScatterPlotItem(x=[x[0] for x in X_pos], y=[x[1] for x in X_pos],
                size=12, brush=pg.mkBrush(0, 100, 255, 200), symbol='o', name=modelo.classe_positiva))
        if X_neg:
            plot.addItem(pg.ScatterPlotItem(x=[x[0] for x in X_neg], y=[x[1] for x in X_neg],
                size=12, brush=pg.mkBrush(255, 50, 50, 200), symbol='t', name=modelo.classe_negativa))
                
        # Treina um perceptron 2D rápido para traçar a reta exata no gráfico 2D
        from classificadores import PerceptronClassico
        perc_2d = PerceptronClassico(n_atributos=2, taxa_aprendizado=modelo.taxa_aprendizado, max_epocas=modelo.max_epocas)
        X_2d = X_pos + X_neg
        y_2d = [1]*len(X_pos) + [0]*len(X_neg)
        perc_2d.treinar(X_2d, y_2d, modelo.classe_positiva, modelo.classe_negativa)
        
        todos_x1 = [x[0] for x in X_2d]
        x_vals = [min(todos_x1) - 0.5, max(todos_x1) + 0.5]
        
        if perc_2d.pesos[1] != 0:
            y_vals = [-(perc_2d.pesos[0] * x + perc_2d.bias) / perc_2d.pesos[1] for x in x_vals]
            equacao = f"Reta: {perc_2d.pesos[0]:.2f}*x1 + {perc_2d.pesos[1]:.2f}*x2 + ({perc_2d.bias:.2f}) = 0"
            plot.addItem(pg.PlotDataItem(x_vals, y_vals, pen=pg.mkPen('y', width=2, dash=[4, 4]), name=equacao))
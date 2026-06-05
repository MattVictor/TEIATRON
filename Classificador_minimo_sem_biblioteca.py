import random
import matplotlib.pyplot as plt
import numpy as np


def produto_escalar(vetor):
    resultado = 0
    
    for i in vetor:
        resultado+=(i**2)
        
    return resultado

def definir_limites(ax, base, idx_x, idx_y, margem=0.5):
    xs = [ponto[idx_x] for ponto, _ in base]
    ys = [ponto[idx_y] for ponto, _ in base]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    ax.set_xlim(min_x - margem, max_x + margem)
    ax.set_ylim(min_y - margem, max_y + margem)

def plotar_reta(ax, c1, c2, idx_x=0, idx_y=1):
    # Calcula vetor normal (nx, ny) e ponto médio (mx, my)
    nx, ny, mx, my = calcular_reta_decisao(c1, c2, idx_x, idx_y)

    # Captura os limites atuais do gráfico
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Caso normal (reta inclinada ou horizontal)
    if ny != 0:
        # A equação é y = -(nx/ny)*x + my + (nx/ny)*mx
        coef_angular = -(nx / ny)
        coef_linear = my - (coef_angular * mx)
        
        x_vals = [x_min, x_max]
        y_vals = [coef_angular * x + coef_linear for x in x_vals]
        
        # Gera o label idêntico ao que calculamos antes: "x2 = -2.80x1 + 8.92"
        label_reta = rf'Superfície: $g(x) = {coef_angular:.2f}x_1 + {coef_linear:.2f}$'
        
        ax.plot(x_vals, y_vals, '-', color='black', label=label_reta, linewidth=2)

    # Caso vertical (ny == 0)
    else:
        # A reta sobe e desce reta no ponto X do ponto médio (mx)
        x_vals = [mx, mx]
        y_vals = [y_min, y_max]
        
        # A equação de uma reta vertical tem o formato "x = constante"
        label_reta = rf'Superfície: $x_1 = {mx:.2f}$'
        
        ax.plot(x_vals, y_vals, '--', color='red', label=label_reta, linewidth=2)
        
    # Restaura os limites do gráfico para a reta não esticar os eixos ao infinito
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)
    
    # ADICIONE ESTA LINHA:
    ax.legend(loc='best')


def calcular_reta_decisao(c1, c2, idx_x, idx_y):
    # Vetor normal
    nx = c2[idx_x] - c1[idx_x]
    ny = c2[idx_y] - c1[idx_y]

    # Ponto médio
    mx = (c1[idx_x] + c2[idx_x]) / 2
    my = (c1[idx_y] + c2[idx_y]) / 2

    return nx, ny, mx, my


def filtrar_classes(base, classe1, classe2):
    return [
        (ponto, classe)
        for ponto, classe in base
        if classe == classe1 or classe == classe2
    ]

# Estratificar dados
def dividir_base(base_dados, proporcao_treino=0.7):
    # 1. Agrupar por classe
    grupos = {}
    for ponto, classe in base_dados:
        if classe not in grupos:
            grupos[classe] = []
        grupos[classe].append((ponto, classe))

    treino = []
    teste = []

    # 2. Dividir cada classe separadamente
    for classe, itens in grupos.items():
        itens_copia = itens[:]
        random.shuffle(itens_copia)

        corte = int(len(itens_copia) * proporcao_treino)

        treino.extend(itens_copia[:corte])
        teste.extend(itens_copia[corte:])

    # 3. Embaralhar os conjuntos finais
    random.shuffle(treino)
    random.shuffle(teste)

    return treino, teste


# Função de importação dos dados
def carregar_dados(caminho_arquivo="TEIATRON\\Iris_data.csv"):
    base_dados = []

    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        linhas = f.readlines()

    for linha in linhas:
        if linha[0] == 'S':
            continue
        
        if not linha:
            continue
        
        linha = linha.replace("\n","")
        partes = linha.split(',')

        for i in range(len(partes)):
            partes[i] = partes[i].strip('"')
        
        # 4 atributos + 1 classe
        atributos = [float(x) for x in partes[:-1]]
        classe = partes[-1]

        base_dados.append((atributos, classe))
        

    for i in base_dados:
        print(i)
    
    return base_dados

# Distância euclidiana
def calcular_distancia(p1, p2):
    soma = 0
    for i in range(len(p1)):
        soma += (p1[i] - p2[i]) ** 2
    return soma ** 0.5

# 1. Calcular centróides
def calcular_centroides(base_dados):
    grupos = {}

    # Agrupar por classe
    for ponto, classe in base_dados:
        if classe not in grupos:
            grupos[classe] = []
        grupos[classe].append(ponto)

    centroides = {}

    # Calcular média
    for classe, pontos in grupos.items():
        n = len(pontos)
        dimensao = len(pontos[0])

        media = [0] * dimensao

        for ponto in pontos:
            for i in range(dimensao):
                print(media)
                print(ponto)
                media[i] += ponto[i]

        for i in range(dimensao):
            media[i] /= n

        centroides[classe] = media

    return centroides

# 2. Classificar novo ponto
def classificador_minimo(centroides, novo_ponto):
    melhor_classe = None
    menor_distancia = float('inf')

    for classe, centroide in centroides.items():
        d = calcular_distancia(novo_ponto, centroide)

        if d < menor_distancia:
            menor_distancia = d
            melhor_classe = classe

    return melhor_classe

# 2. Classificar novo ponto
def classificador_maximo(dataset, novo_ponto):
    melhor_classe = None
    maior_distancia = float('inf')

    distancia_maxima = {
        "setosa": 0,
        "versicolor": 0,
        "virginica": 0
    }
    
    for ponto in dataset:
        d = calcular_distancia(novo_ponto, ponto[0])

        if distancia_maxima[ponto[1]] < d:
            distancia_maxima[ponto[1]] = d

    print(distancia_maxima)

    melhor_classe = min(distancia_maxima, key=distancia_maxima.get)
    
    return melhor_classe

def plotar_classes(ax, base, centroides, classe1, classe2, idx_x=0, idx_y=1):
    parametros = ["Sepal length","Sepal width","Petal length","Petal width"]
    
    cores = {
        classe1: 'blue',
        classe2: 'red'
    }

    # Plotar pontos
    for ponto, classe in base:
        x = ponto[idx_x]
        y = ponto[idx_y]
        ax.scatter(x, y, color=cores[classe],s=1)

    # definir zoom baseado nos dados
    definir_limites(ax, base, idx_x, idx_y)

    # Plotar centróides
    for classe, centroide in centroides.items():
        x = centroide[idx_x]
        y = centroide[idx_y]
        ax.scatter(x, y, color='black', marker='X', s=50)

    classes = list(centroides.keys())
    c1 = centroides[classes[0]]
    c2 = centroides[classes[1]]

    plotar_reta(ax, c1, c2, idx_x, idx_y)

    ax.set_title(f"{classe1} vs {classe2}")
    ax.set_xlabel(f"{parametros[idx_x]}")
    ax.set_ylabel(f"{parametros[idx_y]}")
    
    ax.set_aspect('equal', adjustable='datalim')



def plotar_comparacoes(comparacoes, parametros=[2, 3]):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for i, (c1, c2) in enumerate(comparacoes):
        base_filtrada = filtrar_classes(carregar_dados(), c1, c2)
        base_treino, base_teste = dividir_base(base_filtrada)
        centroides = calcular_centroides(base_treino)
        
        plotar_classes(
            axs[i],
            base_filtrada,
            centroides,
            c1,
            c2,
            parametros[0],
            parametros[1]
        )

    plt.tight_layout()
    plt.show()

#-----------------------#
# i) Classificador de distância mínima para as três classes.  
#-----------------------#
class_min = classificador_minimo(calcular_centroides(carregar_dados()),[4.4, 3.5, 1.0, 0.2])
print(f"CLASSIFICADOR MINIMO")
print(f"O ponto foi classificado como: {class_min}")

#-----------------------#
# ii) Classificador de distância mínima para as três classes.  
#-----------------------#
class_max = classificador_maximo(carregar_dados(),[4.4, 3.5, 1.0, 0.2])
print(f"CLASSIFICADOR MAXIMO")
print(f"O ponto foi classificado como: {class_max}")

#-----------------------------------------------#
# iii) Superfície de decisão para  duas classes. 
#-----------------------------------------------#
comparacoes = [
    ("virginica", "setosa"),
    ("setosa", "versicolor"),
    ("versicolor", "virginica")
]
plotar_comparacoes(comparacoes)

# import tkinter as tk
# from matplotlib.figure import Figure
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# # 1. Configuração da Janela Principal
# root = tk.Tk()
# root.title("Matplotlib dentro do Tkinter")
# root.geometry("800x600")

# # 2. Criando o Frame que vai receber o gráfico
# frame_grafico = tk.Frame(root, bg="white")
# frame_grafico.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

# # 3. Criando a Figura do Matplotlib
# fig = Figure(figsize=(6, 4), dpi=100)
# ax = fig.add_subplot(111)

# # Gerando alguns dados de exemplo
# x = [1, 2, 3, 4, 5]
# y = [10, 25, 15, 30, 20]
# ax.plot(x, y, marker='o', linestyle='-', color='b', label='Vendas')
# ax.set_title("Meu Gráfico Interativo")
# ax.set_xlabel("Tempo")
# ax.set_ylabel("Valor")
# ax.grid(True)
# ax.legend()

# # 4. Integrando o Gráfico ao Tkinter (Canvas)
# canvas = FigureCanvasTkAgg(fig, master=frame_grafico)
# canvas.draw()
# # Empacotando o canvas no frame
# canvas_widget = canvas.get_tk_widget()
# canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# # 5. Adicionando a Barra de Ferramentas (Toolbar)
# toolbar = NavigationToolbar2Tk(canvas, frame_grafico)
# toolbar.update()
# # Empacotando a barra (o pack do próprio widget da toolbar)
# canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# # Iniciando o loop da interface
# root.mainloop()

# class Perceptron:
#     def __init__(self, num_entradas, taxa_aprendizado=0.1, epocas=10):
#         # Inicializa os pesos e o bias com zero
#         self.pesos = [0.0] * num_entradas
#         self.bias = 0.0
#         self.taxa_aprendizado = taxa_aprendizado
#         self.epocas = epocas

#     # Função de Ativação (Degrau)
#     def funcao_ativacao(self, valor):
#         return 1 if valor >= 0 else 0

#     # Função para realizar a previsão (Passo de Feedforward)
#     def prever(self, entradas):
#         # Calcula a soma ponderada: somatório de (x * w) + bias
#         soma = sum(x * w for x, w in zip(entradas, self.pesos)) + self.bias
#         return self.funcao_ativacao(soma)

#     # Função de Treinamento
#     def treinar(self, dados_treino, rotulos):
#         for epoca in range(self.epocas):
#             total_erros = 0
#             for entradas, alvo in zip(dados_treino, rotulos):
#                 # 1. Faz a previsão
#                 previsao = self.prever(entradas)
                
#                 # 2. Calcula o erro
#                 erro = alvo - previsao
                
#                 # 3. Se houver erro, atualiza pesos e bias
#                 if erro != 0:
#                     total_erros += abs(erro)
#                     # Atualiza cada peso
#                     for i in range(len(self.pesos)):
#                         self.pesos[i] += self.taxa_aprendizado * erro * entradas[i]
#                     # Atualiza o bias
#                     self.bias += self.taxa_aprendizado * erro
            
#             print(f"Época {epoca + 1}: Erros cometidos = {total_erros}")
#             # Se não houver mais erros, o modelo convergiu e podemos parar
#             if total_erros == 0:
#                 print("O modelo convergiu antecipadamente!")
#                 break


# # --- TESTANDO O PERCEPTRON (Porta Lógica AND) ---

# todos_dados = carregar_dados()

# # Dados de entrada (X1, X2)
# dados_treino = []
# rotulos_treino = []

# flor_eliminada = "versicolor"
# flor_1 = "setosa"
# flor_0 = "virginica"

# for vetor,rotulo in todos_dados:
#     if(rotulo != flor_eliminada):
#         dados_treino.append(vetor)
#         rotulos_treino.append(1 if rotulo == flor_1 else 0)
        
# # Criando o modelo para 2 entradas
# modelo = Perceptron(num_entradas=4, taxa_aprendizado=0.1, epocas=1000)

# print("--- Iniciando o Treinamento ---")
# modelo.treinar(dados_treino, rotulos_treino)

# print("\n--- Pesos Finais Aprendidos ---")
# print(f"Pesos: {modelo.pesos}")
# print(f"Bias: {modelo.bias}")

# print("\n--- Testando as Previsões Finais ---")
# for i in range(len(dados_treino)):
#     resultado = modelo.prever(dados_treino[i])
#     print(f"Entrada: {dados_treino[i]}({flor_1 if rotulos_treino[i] == 1 else flor_0}) -> Previsão do Perceptron: {flor_1 if resultado == 1 else flor_0}")
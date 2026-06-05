from dataset import Iris_Data
from classificadores import PerceptronClassico
from graficos import GraficosClassificadores

def filtrar_duas_classes(treino, teste, classe_pos, classe_neg):
    """Filtra as listas para conter apenas as duas classes do embate"""
    tr_filtrado = [d for d in treino if d[1] in [classe_pos, classe_neg]]
    te_filtrado = [d for d in teste if d[1] in [classe_pos, classe_neg]]
    return tr_filtrado, te_filtrado

def calcular_acuracia_binaria(modelo, X_teste, y_teste):
    acertos = sum(1 for x, y in zip(X_teste, y_teste) if modelo.prever(x) == y)
    return (acertos / len(X_teste)) * 100

def main():
    print("Iniciando Perceptron Clássico...")
    iris = Iris_Data()
    
    # ITEM A: Separação 70/30
    treino, teste = iris.get_treino_teste_balanceado(proporcao=0.70)
    
    # ITEM A (Gráfico): Plotar nuvem inteira usando X1 e X2
    GraficosClassificadores.plotar_nuvem_todas_classes(iris.dados_tratados)
    
    # ITEM C: Definindo os pares para classificação binária
    pares = [
        ("setosa", "versicolor"),
        ("versicolor", "virginica"),
        ("setosa", "virginica")
    ]
    
    relatorio = []
    modelos_treinados = []
    
    # ITEM D: Fluxo de Classificação Binária
    for classe_pos, classe_neg in pares:
        print(f"Treinando {classe_pos} vs {classe_neg}...")
        
        # 1. Filtra os dados apenas para as duas classes do par
        tr_par, te_par = filtrar_duas_classes(treino, teste, classe_pos, classe_neg)
        
        # 2. Converte as strings para 1 (pos) e 0 (neg)
        tr_bin = iris.mapear_classes_binarias(tr_par, classe_pos, rotulo_pos=1, rotulo_neg=0)
        te_bin = iris.mapear_classes_binarias(te_par, classe_pos, rotulo_pos=1, rotulo_neg=0)
        
        # 3. Separa em Atributos (X) e Alvos (y)
        X_tr, y_tr = iris.separar_x_y(tr_bin)
        X_te, y_te = iris.separar_x_y(te_bin)
        
        # ITEM B: Inicializa Perceptron com taxa = 0.03 e w(1) = 0
        perc = PerceptronClassico(n_atributos=4, taxa_aprendizado=0.03, max_epocas=100)
        
        # Treina
        epocas_rodadas = perc.treinar(X_tr, y_tr, classe_pos, classe_neg)
        
        # Testa
        acc = calcular_acuracia_binaria(perc, X_te, y_te)
        
        # Guarda para plotar e salvar
        modelos_treinados.append((perc, X_tr, y_tr))
        
        pesos_str = ", ".join([f"{w:.4f}" for w in perc.pesos])
        relatorio.append(f"--- {classe_pos.upper()} vs {classe_neg.upper()} ---")
        relatorio.append(f"Épocas Treinadas: {epocas_rodadas} / 100")
        relatorio.append(f"Vetor de Pesos Final W: [{pesos_str}]")
        relatorio.append(f"Bias final: {perc.bias:.4f}")
        relatorio.append(f"Acurácia no Teste: {acc:.2f}%\n")
        
        # ITEM E: Observação sobre Versicolor vs Virginica
        if classe_pos == "versicolor" and classe_neg == "virginica":
            if epocas_rodadas == 100:
                relatorio.append("OBSERVACAO ITEM E: O algoritmo atingiu o limite máximo de 100 épocas. "
                                 "Isso ocorreu porque Versicolor e Virginica não são linearmente separáveis, "
                                 "provando que o Perceptron Clássico entra em loop infinito sem uma trava.\n")

    # Salva o arquivo de texto
    with open("relatorio_perceptron.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(relatorio))
    print("Relatório salvo em 'relatorio_perceptron.txt'.")

    # Plota os resultados
    for modelo, X_tr, y_tr in modelos_treinados:
        GraficosClassificadores.plotar_superficie_perceptron(
            modelo, X_tr, y_tr, 
            titulo=f"Superfície: {modelo.classe_positiva} vs {modelo.classe_negativa}"
        )

    GraficosClassificadores.mostrar_todos()

if __name__ == "__main__":
    main()
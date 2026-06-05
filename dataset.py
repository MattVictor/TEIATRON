import csv
from random import shuffle

class Iris_Data():
    def __init__(self, caminho_arquivo="TEIATRON\\Iris_data.csv"):
        self.dados_tratados = self.carregar_dados(caminho_arquivo)
        
        self.dados_dict = {
            "setosa": [i for i in self.dados_tratados if i[1] == "setosa"],
            "virginica": [i for i in self.dados_tratados if i[1] == "virginica"],
            "versicolor": [i for i in self.dados_tratados if i[1] == "versicolor"]
        }
    
    # Função de importação dos dados SIMPLIFICADA
    def carregar_dados(self, caminho_arquivo):
        base_dados = []

        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            leitor = csv.reader(f)
            
            for linha in leitor:
                # Ignora cabeçalho (que começa com 'S' de Sepal) ou linhas vazias
                if not linha or linha[0].startswith('S'):
                    continue
                
                # Graças ao seu novo tratamento, basta converter direto para float!
                atributos = [float(x) for x in linha[:-1]]
                classe = linha[-1].strip() # strip() por segurança, para remover espaços na classe

                base_dados.append((atributos, classe))
        
        return base_dados
    
    def get_treino_teste_balanceado(self, proporcao=0.70, classe_excluida=None):
        dados = self.dados_dict.copy()
        
        if classe_excluida and classe_excluida in dados:
            dados.pop(classe_excluida)
        
        bases = dados.values()
        
        treino = []
        teste = []
        
        for classe_dados in bases:
            copia_dados = classe_dados.copy()
            shuffle(copia_dados)
            
            razao = int(proporcao * len(copia_dados))
            
            treino += copia_dados[:razao]
            teste += copia_dados[razao:]
            
        shuffle(treino)
        shuffle(teste)
        
        return treino, teste
    
    def get_treino_teste_desbalanceado(self, proporcao=0.70, classe_excluida=None):
        dados = self.dados_dict.copy()
        
        if classe_excluida and classe_excluida in dados:
            dados.pop(classe_excluida)
        
        todos_dados = []
        for classe_dados in dados.values():
            todos_dados.extend(classe_dados)
        
        shuffle(todos_dados)
        
        razao = int(proporcao * len(todos_dados))
        
        treino = todos_dados[:razao]
        teste = todos_dados[razao:]
        
        return treino, teste

    # MÉTODOS AUXILIARES PARA OS CLASSIFICADORES
    def mapear_classes_binarias(self, dataset, classe_positiva, rotulo_pos=1, rotulo_neg=-1):
        dataset_mapeado = []
        for atributos, classe in dataset:
            alvo = rotulo_pos if classe == classe_positiva else rotulo_neg
            dataset_mapeado.append((atributos, alvo))
        return dataset_mapeado

    def separar_x_y(self, dataset):
        X = [linha[0] for linha in dataset]
        y = [linha[1] for linha in dataset]
        return X, y
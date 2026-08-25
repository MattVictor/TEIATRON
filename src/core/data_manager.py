import random
import pandas as pd
import numpy as np

class DataManager:
    def __init__(self):
        self.feature_names = []
        self.class_names = []
        self.target_column = None
        self.raw_data = None  # DataFrame do pandas
        self.headers = []
        self.conjunto_col = "Conjunto"

    def load_csv(self, file_path, target_col):
        """Lê o CSV usando pandas, valida tipos e define as colunas."""
        df = pd.read_csv(file_path)
        
        if df.empty:
            raise Exception("CSV inválido ou vazio.")
            
        if target_col not in df.columns:
            raise Exception(f"Coluna alvo '{target_col}' não encontrada no dataset.")
            
        self.target_column = target_col
        
        self.feature_names = []
        invalid_cols = []
        
        for c in df.columns:
            if c == target_col:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                self.feature_names.append(c)
            else:
                invalid_cols.append(c)
                
        if not self.feature_names:
            raise Exception("Erro: O dataset não possui nenhuma coluna numérica para ser usada como feature!")
            
        if invalid_cols:
            df = df.drop(columns=invalid_cols)
            
        self.class_names = df[target_col].dropna().astype(str).unique().tolist()
        
        if self.conjunto_col not in df.columns:
            df[self.conjunto_col] = "Treino"
            
        # Garante a ordem exata das colunas: TODAS as features, depois o ALVO, e por último o CONJUNTO
        self.headers = self.feature_names + [self.target_column, self.conjunto_col]
        
        self.raw_data = df[self.headers]
        data_matrix = self.raw_data.astype(str).values.tolist()
        return self.headers, data_matrix

    def apply_split(self, stratified, train_ratio):
        """Aplica a aleatorização (Treino/Teste) usando pandas."""
        if self.raw_data is None or self.raw_data.empty:
            return [], []
            
        df = self.raw_data.copy()
        df[self.conjunto_col] = "Treino"
        
        if stratified:
            # Embaralha e divide mantendo a proporção de cada classe
            for name, group in df.groupby(self.target_column):
                shuffled = group.sample(frac=1)
                split_idx = int(len(shuffled) * train_ratio)
                test_idx = shuffled.index[split_idx:]
                df.loc[test_idx, self.conjunto_col] = "Teste"
        else:
            # Divisão simples
            shuffled = df.sample(frac=1)
            split_idx = int(len(shuffled) * train_ratio)
            test_idx = shuffled.index[split_idx:]
            df.loc[test_idx, self.conjunto_col] = "Teste"
            
        # Reordenar para embaralhar a exibição (opcional, mas bom)
        df = df.sample(frac=1).reset_index(drop=True)
        
        self.raw_data = df[self.headers]
        data_matrix = self.raw_data.astype(str).values.tolist()
        return self.headers, data_matrix

    def get_full_dataset(self):
        """Extrai os dados organizados em dicionário para o formato exigido pelos modelos."""
        if self.raw_data is None or self.raw_data.empty:
            raise Exception("Dataset não carregado")
            
        dataset = {key: self.raw_data[key].astype(float).tolist() for key in self.feature_names}
        class_data = self.raw_data[self.target_column].astype(str).tolist()
        conjunto_data = self.raw_data[self.conjunto_col].astype(str).tolist()
        
        return dataset, class_data, conjunto_data

    def prepare_data(self, dataset, class_data, conjunto_data, params):
        # Em vez de Iris-setosa, pega as classes diretamente dos parâmetros ou do dataset
        c1 = params.get("Classe 1", "")
        c2 = params.get("Classe 2", "")
        alvo = params.get("Classe Alvo", "")

        is_perceptron_or_svm = params.get('Algoritmo') in ["Perceptron", "Máquina de Vetores de Suporte (SVM)"]
        is_ova = is_perceptron_or_svm and params.get("Estratégia") == "Um contra todos"

        if is_perceptron_or_svm:
            is_multiclass = True if is_ova else False
        else:
            is_multiclass = params.get("Multiclasse", True)

        filtered_dataset = {k: [] for k in self.feature_names}
        filtered_class_data = []
        filtered_conjunto_data = []
        X_train = []
        y_train = []
        
        selected_features = params.get("selected_features", self.feature_names)

        for i in range(len(class_data)):
            classe_atual = class_data[i]
            classe_exibicao = classe_atual

            if not is_multiclass:
                if classe_atual not in [c1, c2]:
                    continue

            if is_ova:
                classe_exibicao = alvo if classe_atual == alvo else "Resto"
                if classe_atual != alvo:
                    if len([c for c in filtered_class_data if c == "Resto"]) >= len([c for c in filtered_class_data if c == alvo]):
                        if random.random() > 0.5:
                            continue

            for k in self.feature_names:
                filtered_dataset[k].append(dataset[k][i])
                
            filtered_class_data.append(classe_exibicao)
            filtered_conjunto_data.append(conjunto_data[i])

            if conjunto_data[i] == "Treino":
                ponto = [dataset[k][i] for k in selected_features]
                X_train.append(ponto)
                y_train.append(classe_exibicao)

        return {
            'filtered_dataset': filtered_dataset,
            'filtered_class_data': filtered_class_data,
            'filtered_conjunto_data': filtered_conjunto_data,
            'X_train': X_train,
            'y_train': y_train,
            'is_ova': is_ova,
            'alvo': alvo
        }

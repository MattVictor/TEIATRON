import csv
import random

class DataManager:
    def __init__(self):
        self.keys = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
        self.headers = []
        self.raw_data = [] # Matriz bruta (strings)

    def load_csv(self, file_path):
        """Lê o CSV do disco e guarda na memória."""
        with open(file_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            data = list(reader)

        if not data or len(data) < 2:
            raise Exception("CSV inválido ou vazio.")

        self.headers = data[0]
        self.raw_data = data[1:]
        
        # Se não tiver a coluna "Conjunto", adicione espaço para ela
        if "Conjunto" not in self.headers:
            self.headers.append("Conjunto")
            for row in self.raw_data:
                row.append("Treino") # Por padrão
        return self.headers, self.raw_data

    def apply_split(self, stratified, train_ratio):
        """Aplica a aleatorização e divide os dados armazenados em memória."""
        if not self.raw_data:
            return [], []
            
        has_conjunto = "Conjunto" in self.headers
        class_col_idx = len(self.headers) - 2 if has_conjunto else len(self.headers) - 1

        if stratified:
            groups = {}
            for row in self.raw_data:
                c = row[class_col_idx]
                if c not in groups:
                    groups[c] = []
                groups[c].append(row)
            
            final_data = []
            for c, group in groups.items():
                random.shuffle(group)
                split_idx = int(len(group) * train_ratio)
                for i, row in enumerate(group):
                    row[-1] = "Treino" if i < split_idx else "Teste"
                    final_data.append(row)
            self.raw_data = final_data 
        else:
            random.shuffle(self.raw_data)
            split_idx = int(len(self.raw_data) * train_ratio)
            for i, row in enumerate(self.raw_data):
                row[-1] = "Treino" if i < split_idx else "Teste"
                
        return self.headers, self.raw_data

    def get_full_dataset(self):
        """Transforma a matriz bruta de strings em arrays tipados (float)."""
        if not self.raw_data:
            raise Exception("Dataset não carregado")
            
        dataset = {key: [] for key in self.keys}
        class_data = []
        conjunto_data = []
        
        cols = len(self.headers)
        conj_idx = self.headers.index("Conjunto") if "Conjunto" in self.headers else -1

        for row in self.raw_data:
            try:
                for j, key in enumerate(self.keys):
                    if j < cols:
                        val = float(row[j].replace(',', '.'))
                        dataset[key].append(val)
                    else:
                        dataset[key].append(0.0)
                
                if cols >= 6:
                    c = row[4].strip()
                elif cols >= 5:
                    c = row[4].strip()
                else:
                    c = "Desconhecida"
                    
                conj = row[conj_idx].strip() if conj_idx != -1 else "Treino"
                    
                class_data.append(c)
                conjunto_data.append(conj)
            except Exception:
                pass
                
        return dataset, class_data, conjunto_data

    def prepare_data(self, dataset, class_data, conjunto_data, params):
        c1 = params.get("Classe 1", "").replace("Iris-", "")
        c2 = params.get("Classe 2", "").replace("Iris-", "")
        alvo = params.get("Classe Alvo", "").replace("Iris-", "")

        is_perceptron = (params.get('Algoritmo') == "Perceptron")
        is_ova = is_perceptron and params.get("Estratégia") == "Um contra todos"

        if is_perceptron:
            is_multiclass = True if is_ova else False
        else:
            is_multiclass = params.get("Multiclasse", True)

        filtered_dataset = {k: [] for k in self.keys}
        filtered_class_data = []
        filtered_conjunto_data = []
        X_train = []
        y_train = []

        for i in range(len(class_data)):
            classe_atual = class_data[i]
            classe_exibicao = classe_atual

            # SEGREGAÇÃO VISUAL: Agrupa as classes para "Um contra todos"
            if is_ova:
                classe_exibicao = alvo if classe_atual == alvo else "Resto"
            # PULA as que não forem C1 e C2 se for binário normal
            elif not is_multiclass:
                if classe_atual not in [c1, c2]:
                    continue

            for k in self.keys:
                filtered_dataset[k].append(dataset[k][i])

            filtered_class_data.append(classe_exibicao)
            filtered_conjunto_data.append(conjunto_data[i])

            if conjunto_data[i] == "Treino":
                ponto = [dataset[k][i] for k in self.keys]
                X_train.append(ponto)
                y_train.append(classe_exibicao)

        return {
            'X_train': X_train,
            'y_train': y_train,
            'filtered_dataset': filtered_dataset,
            'filtered_class_data': filtered_class_data,
            'filtered_conjunto_data': filtered_conjunto_data,
            'is_ova': is_ova,
            'c1': c1,
            'alvo': alvo
        }

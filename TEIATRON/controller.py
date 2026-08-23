# controller.py
from data_manager import DataManager
from ml_engine import (
    MinDistanceClassifier, MaxDistanceClassifier, 
    PerceptronClassifier, OptimalBayesMAP, NaiveBayesMAP,
    NeuralNetworkClassifier, SVMClassifier
)
import numpy as np

MODEL_MAP = {
    "Distância Mínima": MinDistanceClassifier,
    "Distância Máxima": MaxDistanceClassifier,
    "Perceptron": PerceptronClassifier,
    "Problema do XOR": PerceptronClassifier,
    "Bayes Ótimo": OptimalBayesMAP,
    "Naive Bayes": NaiveBayesMAP,
    "Rede Neural (MLP)": NeuralNetworkClassifier,
    "Máquina de Vetores de Suporte (SVM)": SVMClassifier
}

class MLController:
    def __init__(self, log_callback):
        self.data_manager = DataManager()
        self.log_callback = log_callback
        self.current_model = None

    def get_algorithm_metadata(self, algo_name):
        model_class = MODEL_MAP.get(algo_name)
        if model_class:
            return model_class.get_hyperparameters()
        return []

    def handle_load_csv(self, file_path):
        return self.data_manager.load_csv(file_path)
        
    def handle_split_data(self, stratified, train_ratio):
        return self.data_manager.apply_split(stratified, train_ratio)

    def train_model(self, dataset, class_data, conjunto_data, params):
        self.log_callback("[SISTEMA] Iniciando preparação dos dados (via Controller)...")
        
        data = self.data_manager.prepare_data(dataset, class_data, conjunto_data, params)
        X_train, y_train = data['X_train'], data['y_train']
        
        if not X_train:
            raise Exception("Nenhum dado de treino encontrado.")

        self.log_callback(f"Algoritmo selecionado: {params.get('Algoritmo')}")
        algo_name = params.get('Algoritmo')
        
        if algo_name == "Distância Mínima":
            self.current_model = MinDistanceClassifier()
            centroides = self.current_model.train(X_train, y_train)
            self.log_callback("\n[TREINAMENTO CONCLUÍDO]")
            for c, vals in centroides.items():
                self.log_callback(f"Centróide '{c}': [{', '.join([f'{v:.2f}' for v in vals])}]")

        elif algo_name == "Distância Máxima":
            self.current_model = MaxDistanceClassifier()
            classes_ops = self.current_model.train(X_train, y_train)
            self.log_callback("\n[TREINAMENTO CONCLUÍDO]")
            self.log_callback(f"Classes armazenadas: {', '.join(classes_ops)}")

        elif algo_name == "Perceptron":
            self.current_model = PerceptronClassifier()
            bias = params.get("Bias Inicial", 0.0)
            pesos_str = params.get("Pesos Iniciais", "0,0,0,0")
            try:
                pesos_list = [bias] + [float(w.strip()) for w in pesos_str.split(",")]
            except ValueError:
                raise Exception("Formato inválido de Pesos. Use números separados por vírgula.")
                
            num_features = len(params.get("selected_features", [1,2,3,4]))
            if len(pesos_list) - 1 > num_features:
                pesos_list = pesos_list[:num_features+1]
            elif len(pesos_list) - 1 < num_features:
                pesos_list += [0.0] * (num_features - (len(pesos_list) - 1))

            regra_delta = params.get("Regra Delta", False)
            epocas = params.get("Épocas", 100)
            lr = params.get("Learning Rate", 0.01)
            
            c1 = data['c1']
            alvo = data['alvo']
            is_ova = data['is_ova']
            
            self.current_model.train(
                X_train, y_train, 
                classe_alvo=alvo if is_ova else c1, 
                epocas=epocas, 
                learning_rate=lr, 
                pesos_iniciais=pesos_list, 
                regra_delta=regra_delta
            )
            
            erros = self.current_model.historico_erros
            total_epocas = len(erros)
            self.log_callback("\n[HISTÓRICO DE ERROS DE CLASSIFICAÇÃO]")
            
            for ep in range(total_epocas):
                if total_epocas <= 50 or ep < 5 or ep >= total_epocas - 5 or ep % (total_epocas // 10) == 0:
                    self.log_callback(f"  ↳ Época {ep + 1:03d}: {erros[ep]} erros")
                elif ep == 5 and total_epocas > 50:
                    self.log_callback("  ↳ ... [ocultado para otimização] ...")
            
            self.log_callback("\n[TREINAMENTO CONCLUÍDO]")
            p_finais = ", ".join([f"{p:.4f}" for p in self.current_model.pesos])
            self.log_callback(f"Pesos Finais [Bias, W1..W4]: [{p_finais}]")
            if erros and erros[-1] == 0:
                self.log_callback("★ Convergiu perfeitamente! ★")
                
        elif algo_name == "Problema do XOR":
            self.log_callback("\n[INICIANDO O FAMOSO PROBLEMA DO XOR]")
            self.log_callback("Aviso Teórico: O XOR não é linearmente separável.")
            
            X_train_xor = [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0]
            ]
            y_train_xor = ["Classe 0", "Classe 1", "Classe 1", "Classe 0"]
            
            self.current_model = PerceptronClassifier()
            epocas = params.get("Épocas", 100)
            lr = params.get("Learning Rate", 0.1)
            bias = params.get("Bias Inicial", 0.0)
            
            pesos_str = params.get("Pesos Iniciais", "0,0,0,0")
            try:
                pesos_list = [bias] + [float(w.strip()) for w in pesos_str.split(",")]
            except ValueError:
                pesos_list = [0.0, 0.0, 0.0, 0.0, 0.0]
            
            if len(pesos_list) < 5:
                pesos_list += [0.0] * (5 - len(pesos_list))
            
            self.current_model.train(
                X_train_xor, y_train_xor, 
                classe_alvo="Classe 1", 
                epocas=epocas, 
                learning_rate=lr, 
                pesos_iniciais=pesos_list, 
                regra_delta=True
            )
            
            erros = self.current_model.historico_erros
            total_epocas = len(erros)
            self.log_callback("\n[HISTÓRICO DE ERROS DA REGRA DELTA]")
            
            for ep in range(total_epocas):
                if total_epocas <= 50 or ep < 5 or ep >= total_epocas - 5 or ep % (total_epocas // 10) == 0:
                    self.log_callback(f"  ↳ Época {ep + 1:03d}: {erros[ep]} erros")
            
            self.log_callback("\n[TREINAMENTO CONCLUÍDO]")
            p_finais = ", ".join([f"{p:.4f}" for p in self.current_model.pesos])
            self.log_callback(f"Pesos Finais: [{p_finais}]")
            
            if erros[-1] > 0:
                self.log_callback("★ Conclusão: Como esperado, o modelo não zerou os erros! O hiperplano linear não consegue separar o XOR. ★")
                
            data['filtered_dataset'] = {
                "Sepal Length": [0.0, 0.0, 1.0, 1.0],
                "Sepal Width":  [0.0, 1.0, 0.0, 1.0],
                "Petal Length": [0.0, 0.0, 0.0, 0.0],
                "Petal Width":  [0.0, 0.0, 0.0, 0.0]
            }
            data['filtered_class_data'] = y_train_xor
            data['filtered_conjunto_data'] = ["Treino"] * 4
            params["selected_features"] = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]

        elif algo_name in ["Bayes Ótimo", "Naive Bayes"]:
            X_train_np = np.array(X_train)
            y_train_np = np.array(y_train)
            
            # Executar testes estatísticos sobre a base de dados
            self.log_callback("\n[ANÁLISE ESTATÍSTICA DO DATASET]")
            # Helper for Mardia
            def mardia_test(X_class):
                n, p = X_class.shape
                if n < 2: return True, 1.0, 1.0
                mean = np.mean(X_class, axis=0)
                cov = np.cov(X_class, rowvar=False)
                try:
                    inv_cov = np.linalg.pinv(cov)
                except:
                    return True, 1.0, 1.0
                
                diff = X_class - mean
                D = diff @ inv_cov @ diff.T
                skewness = np.sum(D**3) / (n**2)
                kurtosis = np.sum(np.diag(D)**2) / n
                
                import scipy.stats as st
                df_skew = p * (p + 1) * (p + 2) / 6.0
                stat_skew = n * skewness / 6.0
                pval_skew = 1.0 - st.chi2.cdf(stat_skew, df_skew)
                
                mean_kurt = p * (p + 2)
                var_kurt = 8.0 * p * (p + 2) / n
                stat_kurt = (kurtosis - mean_kurt) / np.sqrt(var_kurt)
                pval_kurt = 2.0 * (1.0 - st.norm.cdf(abs(stat_kurt)))
                
                return (pval_skew > 0.05 and pval_kurt > 0.05), pval_skew, pval_kurt
                
            try:
                import pingouin as pg
                import pandas as pd
                import scipy.stats as st
                
                df_X = pd.DataFrame(X_train_np)
                df_X['class'] = y_train_np
                classes_presentes = np.unique(y_train_np)
                
                # 1. Teste de Normalidade Multivariada (Mardia e Henze-Zirkler)
                self.log_callback("1. Testes de Normalidade Multivariada:")
                for c in classes_presentes:
                    c_data = df_X[df_X['class'] == c].drop(columns=['class'])
                    if len(c_data) > 0:
                        # Mardia
                        is_mardia, p_skew, p_kurt = mardia_test(c_data.values)
                        self.log_callback(f"  ↳ {c} (Mardia): {'Normal' if is_mardia else 'NÃO Normal'} (p_assimetria={p_skew:.3f}, p_curtose={p_kurt:.3f})")
                        
                        # Henze-Zirkler
                        hz = pg.multivariate_normality(c_data, alpha=0.05)
                        self.log_callback(f"  ↳ {c} (Henze-Zirkler): {'Normal' if hz.normal else 'NÃO Normal'} (p-valor: {hz.pval:.4f})")
                
                # 2. Teste de Homocedasticidade (Box's M Test)
                if len(classes_presentes) > 1:
                    try:
                        box_m = pg.box_m(data=df_X, dvs=df_X.columns[:-1].tolist(), group='class')
                        pval_box = box_m['pval'][0]
                        equal_cov = pval_box > 0.05
                        self.log_callback(f"\n2. Teste M de Box (Igualdade de Covariâncias):")
                        self.log_callback(f"  ↳ Covariâncias {'Iguais (LDA recomendado)' if equal_cov else 'DIFERENTES (QDA recomendado)'} (p-valor: {pval_box:.4f})")
                    except Exception as e:
                        pass
                        
            except ImportError:
                self.log_callback("  (Bibliotecas 'pingouin', 'pandas' e 'scipy' não instaladas. Rode 'pip install pingouin scipy pandas')")
                
            
            if algo_name == "Bayes Ótimo":
                self.current_model = OptimalBayesMAP()
                self.current_model.train(X_train_np, y_train_np, **params)
                self.log_callback("\n[TREINAMENTO BAYES ÓTIMO CONCLUÍDO]")
                self.log_callback(f"Prior: {params.get('Probabilidade a Priori', 'Uniforme (Equiprovável)')}")
                self.log_callback(f"Covariância: {params.get('Matriz de Covariância', 'Individuais (QDA - Curvas)')}")
                
                classes_treinadas = self.current_model.classes
                if len(classes_treinadas) >= 2:
                    self.log_callback("\n[SUPERFÍCIES DE DECISÃO (W, w, w0)]")
                    for i in range(len(classes_treinadas)):
                        for j in range(i + 1, len(classes_treinadas)):
                            c1_c, c2_c = classes_treinadas[i], classes_treinadas[j]
                            W, w, w0 = self.current_model.get_decision_surface(c1_c, c2_c)
                            self.log_callback(f"► Fronteira: {c1_c} x {c2_c}")
                            self.log_callback(f"  Matriz W:\n{np.array_str(W, precision=3, suppress_small=True)}")
                            self.log_callback(f"  Vetor w: {np.round(w, 3)}")
                            self.log_callback(f"  Const w0: {w0:.3f}\n")
                            
            elif algo_name == "Naive Bayes":
                self.current_model = NaiveBayesMAP()
                self.current_model.train(X_train_np, y_train_np, **params)
                self.log_callback("\n[TREINAMENTO NAIVE BAYES CONCLUÍDO]")
                self.log_callback(f"Prior: {params.get('Probabilidade a Priori', 'Uniforme (Equiprovável)')}")

        elif algo_name == "Rede Neural (MLP)":
            self.current_model = NeuralNetworkClassifier()
            epocas = params.get("Épocas", 10000)
            lr = params.get("Learning Rate", 0.5)
            hidden = params.get("Neurônios Ocultos", 2)
            
            self.current_model.train(
                X_train, y_train,
                epocas=epocas,
                learning_rate=lr,
                hidden_neurons=hidden
            )
            
            self.log_callback("\n[TREINAMENTO REDE NEURAL CONCLUÍDO]")
            erros = self.current_model.historico_erros
            if erros:
                self.log_callback(f"Erro inicial: {erros[0]:.6f}")
                self.log_callback(f"Erro final:   {erros[-1]:.6f}")
                
        elif algo_name == "Máquina de Vetores de Suporte (SVM)":
            self.current_model = SVMClassifier()
            kernel = params.get("Kernel", "linear")
            C_val = params.get("C (Regularização)", 1.0)
            degree = params.get("Grau (Poly)", 3)
            
            self.current_model.train(X_train, y_train, Kernel=kernel, **{"C (Regularização)": C_val, "Grau (Poly)": degree})
            self.log_callback("\n[TREINAMENTO SVM CONCLUÍDO]")
            self.log_callback(f"Kernel selecionado: {kernel}")
            if kernel == "poly":
                self.log_callback(f"Grau polinomial: {degree}")
            self.log_callback(f"Vetores de suporte encontrados: {len(self.current_model.model.support_)}")

        if hasattr(self, 'current_model') and self.current_model is not None:
            self.current_model.selected_features = params.get("selected_features", ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"])

        return self.current_model, data

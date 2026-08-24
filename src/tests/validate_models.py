import os
import sys
import numpy as np

# Ensure src is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PyQt6.QtWidgets import QApplication
from core.data_manager import DataManager
from core.ml_engine import ClassificadorMetricas

from algorithms.distance import MinDistanceClassifier
from algorithms.perceptron import PerceptronClassifier
from algorithms.bayes import OptimalBayesMAP, NaiveBayesMAP
from algorithms.svm import SVMClassifier
from algorithms.neural_network import NeuralNetworkClassifier

from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

from tests.visual_validator import VisualValidator

def evaluate_models(teiatron_model, sklearn_model, X_train, y_train, X_test, y_test, model_name, features, kwargs=None):
    if kwargs is None: kwargs = {}
    
    print(f"\n{'='*50}")
    print(f"[{model_name}] Iniciando Validação...")
    
    # 1. Train Both Models
    teiatron_model.train(X_train, y_train, **kwargs)
    # Alguns modelos SKLearn podem precisar de classes numéricas ou strings. Scikit suporta strings na maioria.
    sklearn_model.fit(X_train, y_train)
    
    # Adicionando atributos necessários pro PlotEngine
    if not hasattr(teiatron_model, 'selected_features'):
        teiatron_model.selected_features = features
    
    # 2. Predict on Test Set
    y_pred_teiatron = []
    for pt in X_test:
        res = teiatron_model.predict(pt)
        # Handle tuple return (class, distances/probs)
        pred = res[0] if isinstance(res, tuple) else res
        y_pred_teiatron.append(pred)
        
    y_pred_sklearn = sklearn_model.predict(X_test)
    
    # 3. Calculate TEIATRON Metrics
    # Construir a estrutura que ClassificadorMetricas espera (lista de listas nxn)
    unique_classes = list(sorted(set(y_train) | set(y_test)))
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    n = len(unique_classes)
    matriz = [[0 for _ in range(n)] for _ in range(n)]
    
    for real, pred in zip(y_test, y_pred_teiatron):
        real_idx = class_to_idx.get(real, -1)
        pred_idx = class_to_idx.get(pred, -1)
        if real_idx != -1 and pred_idx != -1:
            matriz[real_idx][pred_idx] += 1
            
    metricas = ClassificadorMetricas(matriz)
    acc_teia = metricas.acerto_geral()
    kappa_teia = metricas.coeficiente_kappa()
    
    # 4. Calculate SKLearn Metrics
    acc_sk = accuracy_score(y_test, y_pred_sklearn)
    kappa_sk = cohen_kappa_score(y_test, y_pred_sklearn)
    
    # 5. Display Results
    print(f"\n[Acurácia Global]")
    print(f"TEIATRON: {acc_teia*100:.2f}% | Scikit-Learn: {acc_sk*100:.2f}%")
    print(f"\n[Coeficiente Kappa]")
    print(f"TEIATRON: {kappa_teia:.4f} | Scikit-Learn: {kappa_sk:.4f}")
    
    # Verificação de Identidade (Quantas predições foram exatamente iguais)
    matches = sum(1 for a, b in zip(y_pred_teiatron, y_pred_sklearn) if a == b)
    print(f"\nPredições idênticas: {matches} / {len(y_test)} ({(matches/len(y_test))*100:.2f}%)")
    
    return teiatron_model, sklearn_model


def main():
    app = QApplication(sys.argv)
    
    dm = DataManager()
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'Iris_data.csv'))
    headers, raw_data = dm.load_csv(data_path)
    
    # Aplicar embaralhamento 70/30 estratificado (isso assegura que o gabarito e o Teiatron recebam os mesmos dados)
    dm.apply_split(stratified=True, train_ratio=0.7)
    
    dataset, class_data, conjunto_data = dm.get_full_dataset()
    features = list(dataset.keys())
    
    # Extrair manualmente para ter certeza que pegamos X_test também
    X_train, y_train, X_test, y_test = [], [], [], []
    for i in range(len(class_data)):
        ponto = [dataset[k][i] for k in features]
        if conjunto_data[i] == "Treino":
            X_train.append(ponto)
            y_train.append(class_data[i])
        else:
            X_test.append(ponto)
            y_test.append(class_data[i])
            
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"Dataset carregado. Treino: {len(y_train)} amostras | Teste: {len(y_test)} amostras.")
    
    # ========================================================
    # SELECIONE O MODELO PARA TESTAR ABAIXO
    # ========================================================
    
    models_to_test = [
        (MinDistanceClassifier(), NearestCentroid(), "Distância Mínima (Centróides)", {}),
        (OptimalBayesMAP(), LinearDiscriminantAnalysis(), "Bayes Ótimo (LDA / Agrupada)", {'Matriz de Covariância': 'Agrupada (LDA - Retas)'}),
        (OptimalBayesMAP(), QuadraticDiscriminantAnalysis(), "Bayes Ótimo (QDA / Individuais)", {'Matriz de Covariância': 'Individuais (QDA - Curvas)'}),
        (NaiveBayesMAP(), GaussianNB(), "Naive Bayes Gaussiano", {}),
        (NeuralNetworkClassifier(), MLPClassifier(hidden_layer_sizes=(2,), max_iter=10000, learning_rate_init=0.5, activation='logistic', solver='sgd', momentum=0, random_state=42), "Rede Neural (MLP)", {'epocas': 1000, 'learning_rate': 0.5, 'hidden_neurons': 2}),
        # SVM is linear by default in Teiatron using SGD. 
        # (SVMClassifier(), SVC(kernel='linear', C=1.0), "Support Vector Machine (Linear)", {})
    ]
    
    # We will test Perceptron separately because SKLearn's Perceptron uses different epoch/learning_rate logic.
    # Same for MLP and SVM, which have very specific hyperparameters.
    
    for t_m, s_m, name, kw in models_to_test:
        evaluate_models(t_m, s_m, X_train, y_train, X_test, y_test, name, features, kw)
        
    # --- Teste Binário Especial para o Perceptron Clássico ---
    # Manter apenas 2 classes (setosa e versicolor)
    bin_idx_tr = np.isin(y_train, ["setosa", "versicolor"])
    bin_idx_ts = np.isin(y_test, ["setosa", "versicolor"])
    
    # Se por algum motivo as classes estiverem com 'Iris-' (depende do CSV), vamos tentar capturar
    if not np.any(bin_idx_tr):
        bin_idx_tr = np.isin(y_train, ["Iris-setosa", "Iris-versicolor"])
        bin_idx_ts = np.isin(y_test, ["Iris-setosa", "Iris-versicolor"])
        alvo = 'Iris-setosa'
    else:
        alvo = 'setosa'
        
    X_train_bin, y_train_bin = X_train[bin_idx_tr], y_train[bin_idx_tr]
    X_test_bin, y_test_bin = X_test[bin_idx_ts], y_test[bin_idx_ts]
    
    t_perc = PerceptronClassifier()
    s_perc = Perceptron(max_iter=100, eta0=0.01, random_state=42, shuffle=False)
    kw_perc = {'classe_alvo': alvo, 'epocas': 100, 'learning_rate': 0.01}
    evaluate_models(t_perc, s_perc, X_train_bin, y_train_bin, X_test_bin, y_test_bin, "Perceptron Clássico (Binário)", features, kw_perc)
    models_to_test.append((t_perc, s_perc, "Perceptron Clássico (Binário)", kw_perc))
    
    t_nn = NeuralNetworkClassifier()
    s_nn = MLPClassifier(hidden_layer_sizes=(4, 4), max_iter=1000, random_state=42)
    kw_nn = {'epocas': 1000, 'learning_rate': 0.1, 'camadas_ocultas': '4,4', 'semente_aleatoria': 42}
    evaluate_models(t_nn, s_nn, X_train, y_train, X_test, y_test, "Rede Neural (MLP)", features, kw_nn)
    models_to_test.append((t_nn, s_nn, "Rede Neural (MLP)", kw_nn))
    
    # --- Teste da Regra Delta (ADALINE) ---
    from sklearn.linear_model import SGDClassifier
    t_perc_delta = PerceptronClassifier()
    s_perc_delta = SGDClassifier(loss='squared_error', max_iter=100, learning_rate='constant', eta0=0.01, random_state=42, shuffle=False)
    kw_perc_delta = {'classe_alvo': alvo, 'epocas': 100, 'learning_rate': 0.01, 'regra_delta': True}
    evaluate_models(t_perc_delta, s_perc_delta, X_train_bin, y_train_bin, X_test_bin, y_test_bin, "Perceptron c/ Regra Delta (ADALINE)", features, kw_perc_delta)
    models_to_test.append((t_perc_delta, s_perc_delta, "Perceptron c/ Regra Delta (ADALINE)", kw_perc_delta))
        
    print("\nVisualização Visual: Deseja visualizar as fronteiras de decisão geradas?")
    print("Digite o número do modelo para visualizar (ou pressione Enter para pular):")
    for i, (_, _, name, _) in enumerate(models_to_test):
        print(f"[{i}] {name}")
    
    escolha = input("> ")
    if escolha.strip().isdigit():
        idx = int(escolha.strip())
        if 0 <= idx < len(models_to_test):
            t_m, s_m, name, _ = models_to_test[idx]
            
            print(f"\nAbrindo interface visual para: {name}...")
            
            # Se for o Perceptron binário, usar os dados filtrados
            if "Binário" in name or "ADALINE" in name:
                X_full = np.concatenate((X_train_bin, X_test_bin))
                y_full = np.concatenate((y_train_bin, y_test_bin))
            else:
                X_full = np.concatenate((X_train, X_test))
                y_full = np.concatenate((y_train, y_test))
            
            window = VisualValidator(t_m, s_m, X_full, y_full, class_names=sorted(set(y_full)), feature_names=features)
            window.show()
            sys.exit(app.exec())

if __name__ == "__main__":
    main()

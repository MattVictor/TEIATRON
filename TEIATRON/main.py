import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QSplitter, QStackedWidget, QLabel, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from ml_engine import MinDistanceClassifier, MaxDistanceClassifier

from config import setup_pyqtgraph, BG_MAIN, ACCENT_COLOR

from view_input import InputCard, InputExpandedPage
from view_charts import ChartsCard, ChartsExpandedPage
from view_algorithm import AlgorithmCard, AlgorithmExpandedPage
from view_accuracy import AccuracyCard, AccuracyExpandedPage

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dashboard ML & Analytics - Final")
        self.resize(1150, 800)
        self.setStyleSheet(f"background-color: {BG_MAIN};")
        
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # --- 1. INSTANCIANDO OS 4 CARDS ---
        self.card_input = InputCard(
            on_expand_callback=lambda: self.stack.setCurrentIndex(1),
            on_classify_callback=self.classify_point # Conecta o botão ao Main
        )     
        self.card_charts = ChartsCard(lambda: self.stack.setCurrentIndex(2))
        
        # Passando o callback de treinamento para o Algoritmo
        self.card_algo = AlgorithmCard(
            on_expand_callback=lambda: self.stack.setCurrentIndex(3),
            on_train_callback=self.train_model
        )
        self.card_accuracy = AccuracyCard(lambda: self.stack.setCurrentIndex(4))

        # --- 2. LAYOUT DO DASHBOARD ---
        dash_widget = QWidget()
        dash_layout = QVBoxLayout(dash_widget)
        dash_layout.setContentsMargins(20, 20, 20, 20)

        lbl_title = QLabel("Dashboard ML & Analytics")
        lbl_title.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        lbl_title.setStyleSheet(f"color: {ACCENT_COLOR}; margin-bottom: 15px;")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        dash_layout.addWidget(lbl_title)
        
        s_horiz1 = QSplitter(Qt.Orientation.Horizontal)
        s_horiz1.addWidget(self.card_input)
        s_horiz1.addWidget(self.card_charts)

        s_horiz2 = QSplitter(Qt.Orientation.Horizontal)
        s_horiz2.addWidget(self.card_algo)
        s_horiz2.addWidget(self.card_accuracy)

        s_vert = QSplitter(Qt.Orientation.Vertical)
        s_vert.addWidget(s_horiz1)
        s_vert.addWidget(s_horiz2)
        
        dash_layout.addWidget(s_vert, stretch=1)
        self.stack.addWidget(dash_widget)

        # --- 3. PÁGINAS EXPANDIDAS ---
        self.page_input = InputExpandedPage(self.card_input.update_preview_text, lambda: self.stack.setCurrentIndex(0))
        self.page_charts = ChartsExpandedPage(self.card_charts.preview_plot, lambda: self.stack.setCurrentIndex(0))
        
        # INJETAMOS O self.train_model AQUI NO FINAL:
        self.page_algo = AlgorithmExpandedPage(self.card_algo.update_preview_text, lambda: self.stack.setCurrentIndex(0), self.train_model)
        
        # Onde estava: self.page_accuracy = AccuracyExpandedPage(...)
        self.page_accuracy = AccuracyExpandedPage(lambda: self.stack.setCurrentIndex(0), self.evaluate_current_model)

        # --- 4. ADICIONANDO PÁGINAS AO STACK ---
        self.stack.addWidget(self.page_input)    # Index 1
        self.stack.addWidget(self.page_charts)   # Index 2
        self.stack.addWidget(self.page_algo)     # Index 3
        self.stack.addWidget(self.page_accuracy) # Index 4
    
    def train_model(self):
        self.page_algo.clear_logs()
        self.page_algo.append_log("[SISTEMA] Iniciando preparação dos dados...")
        
        try:
            dataset, class_data, conjunto_data = self.page_input.get_full_dataset()
            params = self.page_algo.get_current_params()
            
            if not dataset or not class_data:
                raise Exception("Dataset não carregado.")
                
            self.page_algo.append_log(f"Algoritmo selecionado: {params['Algoritmo']}")
            
            # --- 1. FILTRAGEM DOS DADOS ---
            keys = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
            c1 = params.get("Classe 1", "").replace("Iris-","")
            c2 = params.get("Classe 2", "").replace("Iris-","")
            alvo = params.get("Classe Alvo", "").replace("Iris-","")
            
            is_perceptron = (params['Algoritmo'] == "Perceptron")
            is_ova = is_perceptron and params.get("Estratégia") == "Um contra todos"
            
            if is_perceptron:
                is_multiclass = True if is_ova else False
            else:
                is_multiclass = params.get("Multiclasse", True)

            filtered_dataset = {k: [] for k in keys}
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
                    
                for k in keys:
                    filtered_dataset[k].append(dataset[k][i])
                    
                # Adiciona com o nome "Sanitizado" (Alvo vs Resto)
                filtered_class_data.append(classe_exibicao)
                filtered_conjunto_data.append(conjunto_data[i])

                if conjunto_data[i] == "Treino":
                    ponto = [dataset[k][i] for k in keys]
                    X_train.append(ponto)
                    y_train.append(classe_exibicao) 

            if len(X_train) == 0:
                raise Exception("Nenhum dado de treino encontrado.")

            # --- 2. TREINAMENTO (MOTOR ML) ---
            from ml_engine import MinDistanceClassifier, MaxDistanceClassifier, PerceptronClassifier
            
            if params['Algoritmo'] == "Distância Mínima":
                self.current_model = MinDistanceClassifier()
                centroides = self.current_model.train(X_train, y_train)
                self.page_algo.append_log("\n[TREINAMENTO CONCLUÍDO]")
                for c, vals in centroides.items():
                    self.page_algo.append_log(f"Centróide '{c}': [{', '.join([f'{v:.2f}' for v in vals])}]")

            elif params['Algoritmo'] == "Distância Máxima":
                self.current_model = MaxDistanceClassifier()
                classes_ops = self.current_model.train(X_train, y_train)
                self.page_algo.append_log("\n[TREINAMENTO CONCLUÍDO]")
                self.page_algo.append_log(f"Classes armazenadas: {', '.join(classes_ops)}")

            elif params['Algoritmo'] == "Perceptron":
                self.current_model = PerceptronClassifier()
                bias = params.get("Bias Inicial", 0.0)
                pesos_str = params.get("Pesos Iniciais", "0,0,0,0")
                try:
                    pesos_list = [bias] + [float(w.strip()) for w in pesos_str.split(",")]
                except ValueError:
                    raise Exception("Formato inválido de Pesos. Use números separados por vírgula.")
                    
                if len(pesos_list) != 5:
                    raise Exception("Você precisa informar exatamente 4 pesos (W1 a W4).")

                regra_delta = params.get("Regra Delta", False)
                epocas = params.get("Épocas", 100)
                lr = params.get("Learning Rate", 0.01)
                
                # RODA O TREINO
                self.current_model.train(X_train, y_train, alvo if is_ova else c1, epocas, lr, pesos_list, regra_delta)
                
                erros = self.current_model.historico_erros
                total_epocas = len(erros)
                self.page_algo.append_log("\n[HISTÓRICO DE ERROS DE CLASSIFICAÇÃO]")
                
                for ep in range(total_epocas):
                    if total_epocas <= 50 or ep < 5 or ep >= total_epocas - 5 or ep % (total_epocas // 10) == 0:
                        self.page_algo.append_log(f"  ↳ Época {ep + 1:03d}: {erros[ep]} erros")
                    elif ep == 5 and total_epocas > 50:
                        self.page_algo.append_log("  ↳ ... [ocultado para otimização] ...")
                
                self.page_algo.append_log("\n[TREINAMENTO CONCLUÍDO]")
                p_finais = ", ".join([f"{p:.4f}" for p in self.current_model.pesos])
                self.page_algo.append_log(f"Pesos Finais [Bias, W1..W4]: [{p_finais}]")
                if erros and erros[-1] == 0:
                    self.page_algo.append_log("★ Convergiu perfeitamente! ★")

            # --- 3. ATUALIZAR GRÁFICOS ---
            self.page_algo.append_log("\n[SISTEMA] Atualizando gráficos...")
            self.page_charts.clear_charts()
            
            # Envia os dados com a classe "Resto" já mapeada
            self.page_charts.set_dataset(filtered_dataset, filtered_class_data, filtered_conjunto_data)
            self.page_charts.set_trained_model(self.current_model)
            
            if params['Algoritmo'] == "Perceptron":
                y_erros = self.current_model.historico_erros
                x_epocas = list(range(1, len(y_erros) + 1))
                self.page_charts.add_chart("Evolução do Erro", "Linha", x_epocas, y_erros)
            
            # === CORREÇÃO: Limpa todas as checkboxes ANTES de forçar as 2 iniciais ===
            for chk in self.page_charts.checkboxes:
                chk.blockSignals(True) # Evita disparar eventos no meio da limpeza
                chk.setChecked(False)
                chk.blockSignals(False)
            
            # Agora marca apenas as duas primeiras com segurança
            self.page_charts.checkboxes[0].setChecked(True) 
            self.page_charts.checkboxes[1].setChecked(True) 
            
            # ... (seu código de gerar o gráfico continua intacto aqui) ...
            self.page_charts.plot_custom_chart()
            
            # =========================================================
            # --- 4. AVALIAÇÃO DE DESEMPENHO (DADOS DE TESTE) ---
            # =========================================================
            # --- 4. AVALIAÇÃO DE DESEMPENHO DINÂMICA ---
            # Salva o estado dos dados para o avaliador poder recalcular
            self.eval_data = {
                "dataset": filtered_dataset,
                "classes": filtered_class_data,
                "conjuntos": filtered_conjunto_data
            }
            
            # Executa a avaliação inicial baseada no que estiver selecionado no ComboBox
            modo_atual = self.page_accuracy.combo_mode.currentText()
            self.evaluate_current_model(modo_atual)
            
            # Pega o texto da miniatura do card para mostrar no pop-up
            resumo_acc = self.card_accuracy.preview_label.text().split("\n")[1]

            msg = QMessageBox(self)
            msg.setWindowTitle("Treinamento Concluído")
            msg.setText(f"Modelo treinado com sucesso!\n\n{resumo_acc}")
            
        except Exception as e:
            # ... resto do seu código (except) ...
            
            msg = QMessageBox(self)
            msg.setWindowTitle("Treinamento Concluído")
            msg.setText("Modelo treinado com sucesso!")
            
        except Exception as e:
            self.page_algo.append_log(f"[ERRO CRÍTICO] {e}")
            msg = QMessageBox(self)
            msg.setWindowTitle("Treinamento Interrompido")
            msg.setText(f"{e}")
            
        msg.setStyleSheet("""
                QMessageBox { background-color: #1E1E1E; color: #FFFFFF; }
                QLabel { color: #FFFFFF; font-size: 14px; }
                QPushButton { background-color: #00E5FF; color: #000000; padding: 5px 15px; font-weight: bold; border-radius: 3px; }
        """)
        msg.exec()

    def classify_point(self):
        if not hasattr(self, 'current_model'):
            QMessageBox.warning(self, "Aviso", "Treine um modelo primeiro!")
            return

        current_inputs = self.page_input.get_current_inputs()
        
        ponto = [
            current_inputs["Sepal Length"],
            current_inputs["Sepal Width"],
            current_inputs["Petal Length"],
            current_inputs["Petal Width"]
        ]
        
        # Chama a matemática pura
        resultado = self.current_model.predict(ponto)
        
        if isinstance(resultado, tuple):
            predicted_class, detalhes = resultado
        else:
            predicted_class = resultado
            detalhes = None

        current_inputs["class"] = predicted_class
        
        self.page_charts.set_classified_point(current_inputs)
        
        # --- Formatação Dinâmica do Pop-up ---
        info_html = f"<b style='color: #00E5FF; font-size: 16px;'>{predicted_class}</b><br><br>"
        
        if detalhes:
            model_name = self.current_model.__class__.__name__
            
            if model_name == "MinDistanceClassifier":
                info_html += "<i>Distâncias até os centróides:</i><br>"
                for cls_name, dist in detalhes.items():
                    if cls_name == predicted_class:
                        info_html += f"<span style='color: #4CAF50;'><b>• {cls_name}: {dist:.4f}</b></span><br>"
                    else:
                        info_html += f"• {cls_name}: {dist:.4f}<br>"
                        
            elif model_name == "MaxDistanceClassifier":
                info_html += "<i>Distâncias máximas registradas:</i><br>"
                for cls_name, dist in detalhes.items():
                    if cls_name == predicted_class:
                        info_html += f"<span style='color: #4CAF50;'><b>• {cls_name}: {dist:.4f}</b></span><br>"
                    else:
                        info_html += f"• {cls_name}: {dist:.4f}<br>"
                        
            elif model_name == "PerceptronClassifier":
                info_html += "<i>Matemática do Neurônio:</i><br>"
                ativacao = detalhes.get("Ativação (Soma Ponderada)", 0)
                info_html += f"• Ativação Contínua (Soma): {ativacao:.4f}<br>"
                
                # Explica a decisão do degrau de forma didática
                if ativacao >= 0:
                    info_html += "• Saída Degrau: <span style='color: #4CAF50;'><b>1 (>= 0)</b></span><br>"
                else:
                    info_html += "• Saída Degrau: <span style='color: #F44336;'><b>0 (< 0)</b></span><br>"
        
        msg = QMessageBox(self)
        msg.setWindowTitle("Resultado da Classificação")
        msg.setText("O algoritmo classificou este ponto como:")
        msg.setInformativeText(info_html)
        msg.setStyleSheet("""
            QMessageBox { background-color: #1E1E1E; color: #FFFFFF; }
            QLabel { color: #FFFFFF; font-size: 14px; }
            QPushButton { background-color: #00E5FF; color: #000000; padding: 5px 15px; font-weight: bold; border-radius: 3px; }
        """)
        msg.exec()
        
    def evaluate_current_model(self, modo_selecionado="Apenas Teste"):
        """Calcula a Matriz de Confusão On-the-fly baseado na seleção do usuário (Teste/Treino/Ambos)."""
        if not hasattr(self, 'current_model') or not hasattr(self, 'eval_data'):
            self.card_accuracy.update_preview("Aguardando modelo e dados para avaliação...")
            return

        from ml_engine import ClassificadorMetricas
        
        filtered_dataset = self.eval_data["dataset"]
        filtered_class_data = self.eval_data["classes"]
        filtered_conjunto_data = self.eval_data["conjuntos"]
        keys = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]

        # Descobre dinamicamente os nomes e a ordem das classes presentes no modelo
        classes_unicas = list(sorted(set(filtered_class_data)))
        n = len(classes_unicas)
        class_to_idx = {c: i for i, c in enumerate(classes_unicas)}
        
        # Cria matriz de zeros N x N
        matriz = [[0 for _ in range(n)] for _ in range(n)]
        
        total_avaliado = 0
        for i in range(len(filtered_class_data)):
            conjunto = filtered_conjunto_data[i]
            
            # Filtro da UI
            if modo_selecionado == "Apenas Teste" and conjunto != "Teste": continue
            if modo_selecionado == "Apenas Treino" and conjunto != "Treino": continue

            total_avaliado += 1
            real = filtered_class_data[i]
            ponto = [filtered_dataset[k][i] for k in keys]
            
            res = self.current_model.predict(ponto)
            pred = res[0] if isinstance(res, tuple) else res
            
            real_idx = class_to_idx[real]
            
            # Tratamento de segurança caso o modelo preveja algo fora do radar (ex: OvA onde retorna Resto)
            pred_idx = class_to_idx.get(pred, -1)
            if pred_idx != -1:
                matriz[real_idx][pred_idx] += 1

        if total_avaliado > 0:
            metrics = ClassificadorMetricas(matriz)
            self.page_accuracy.update_metrics(matriz, classes_unicas, metrics)
            self.card_accuracy.update_preview(
                f"[{modo_selecionado}]\nAcerto Geral: {metrics.acerto_geral()*100:.2f}%\n"
                f"Kappa: {metrics.coeficiente_kappa():.4f}\n\nClique para ver a Matriz {n}x{n}."
            )
        else:
            self.card_accuracy.update_preview(f"Nenhum dado encontrado para: {modo_selecionado}")
            
if __name__ == "__main__":
    app = QApplication(sys.argv)
    setup_pyqtgraph() 
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
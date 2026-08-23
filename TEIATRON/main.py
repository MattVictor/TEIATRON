import sys
import numpy as np
import os
import pickle
from PyQt6.QtWidgets import (QMenuBar, QDialog, QListWidget, QInputDialog, QHBoxLayout, QComboBox, QPushButton, QCheckBox,
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QSplitter, QStackedWidget, QLabel, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from ml_engine import MinDistanceClassifier, MaxDistanceClassifier
from controller import MLController

from config import setup_pyqtgraph, BG_MAIN, ACCENT_COLOR

from view_input import InputCard, InputExpandedPage
from view_charts import ChartsCard, ChartsExpandedPage
from view_algorithm import AlgorithmCard, AlgorithmExpandedPage
from view_accuracy import AccuracyCard, AccuracyExpandedPage

class ModelHistoryDialog(QDialog):
    """Janela pop-up que lista os modelos salvos no histórico."""
    def __init__(self, models_dir, parent=None):
        super().__init__(parent)
        self.models_dir = models_dir
        self.selected_model = None
        
        self.setWindowTitle("Histórico de Treinamentos Salvos")
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background-color: #1E1E1E; color: #FFFFFF;")

        layout = QVBoxLayout(self)

        # Lista visual
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet("""
            QListWidget { 
                background-color: #2D2D2D; 
                color: #FFFFFF; 
                border: 1px solid #444444; 
                border-radius: 4px; 
                font-size: 13px; 
                padding: 5px;
            }
            QListWidget::item:hover { background-color: #3D3D3D; }
            QListWidget::item:selected { background-color: #0078D7; color: white; font-weight: bold; }
        """)
        self.list_widget.itemDoubleClicked.connect(self.confirm_selection)
        layout.addWidget(self.list_widget)

        # Botões de Ação
        btn_layout = QHBoxLayout()
        
        btn_load = QPushButton("Carregar Modelo")
        btn_load.setStyleSheet("background-color: #0078D7; color: white; font-weight: bold; padding: 6px 15px; border-radius: 4px;")
        btn_load.clicked.connect(self.confirm_selection)
        
        btn_delete = QPushButton("Excluir")
        btn_delete.setStyleSheet("background-color: transparent; color: #FF4D4D; padding: 6px 15px; border: 1px solid #FF4D4D; border-radius: 4px;")
        btn_delete.clicked.connect(self.delete_selected)
        
        btn_layout.addWidget(btn_delete)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_load)
        layout.addLayout(btn_layout)

        self.load_history()

    def load_history(self):
        """Varre a pasta e joga os nomes limpos na lista."""
        self.list_widget.clear()
        if os.path.exists(self.models_dir):
            arquivos = [f for f in os.listdir(self.models_dir) if f.endswith(".pkl")]
            for f in arquivos:
                self.list_widget.addItem(f.replace(".pkl", ""))

    def confirm_selection(self):
        """Define o modelo selecionado e fecha a janela para carregar no sistema."""
        current_item = self.list_widget.currentItem()
        if current_item:
            self.selected_model = current_item.text()
            self.accept() # Fecha o pop-up com sinal de sucesso

    def delete_selected(self):
        """Remove o arquivo selecionado."""
        current_item = self.list_widget.currentItem()
        if not current_item:
            return
            
        nome = current_item.text()
        resposta = QMessageBox.question(
            self, "Confirmar Exclusão", 
            f"Deseja apagar permanentemente o modelo '{nome}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if resposta == QMessageBox.StandardButton.Yes:
            filepath = os.path.join(self.models_dir, f"{nome}.pkl")
            if os.path.exists(filepath):
                os.remove(filepath)
                self.load_history()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dashboard ML & Analytics - Final")
        self.resize(1150, 800)
        self.setStyleSheet(f"background-color: {BG_MAIN};")
        
        # --- 1. CONFIGURAÇÃO DO DIRETÓRIO DE MODELOS ---
        self.models_dir = "saved_models"
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        # --- 2. CONFIGURAÇÃO DA MENUBAR PADRÃO DO PYQT6 ---
        # Cria a barra no topo absoluto da janela
        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar) # Vincula nativamente à MainWindow
        self.menu_bar.setStyleSheet("""
            QMenuBar { 
                background-color: #1A1A1A; 
                color: #FFFFFF; 
                font-size: 13px;
                border-bottom: 1px solid #333333;
            }
            QMenuBar::item { background-color: transparent; padding: 5px 10px; }
            QMenuBar::item:selected { background-color: #333333; color: #00E5FF; }
        """)

        # Adiciona os menus e as ações
        menu_modelos = self.menu_bar.addMenu("Modelos")
        
        acao_historico = menu_modelos.addAction("Histórico de Treinamento")
        acao_historico.triggered.connect(self.open_history_dialog)
        
        acao_salvar = menu_modelos.addAction("Salvar Modelo Atual")
        acao_salvar.triggered.connect(self.save_model)
        
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
        self.card_accuracy = AccuracyCard(self.open_accuracy_page)

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
        self.page_input = InputExpandedPage(
            self.card_input.update_preview_text, 
            lambda: self.stack.setCurrentIndex(0),
            on_import_callback=lambda f: self.controller.handle_load_csv(f),
            on_split_callback=lambda s, tr: self.controller.handle_split_data(s, tr)
        )
        self.page_charts = ChartsExpandedPage(self.card_charts.preview_plot, lambda: self.stack.setCurrentIndex(0))
        
        # INJETAMOS O self.train_model AQUI NO FINAL:
        self.page_algo = AlgorithmExpandedPage(
            self.card_algo.update_preview_text, 
            lambda: self.stack.setCurrentIndex(0), 
            self.train_model,
            get_metadata_callback=lambda algo: self.controller.get_algorithm_metadata(algo)
        )
        
        # Onde estava: self.page_accuracy = AccuracyExpandedPage(...)
        self.page_accuracy = AccuracyExpandedPage(lambda: self.stack.setCurrentIndex(0), self.evaluate_current_model)
        self.controller = MLController(self.page_algo.append_log)
        
        # Agora que o controller e a page_algo estão criados, nós forçamos o primeiro build do form dinâmico
        self.page_algo.build_dynamic_form()
        
        # --- 4. ADICIONANDO PÁGINAS AO STACK ---
        self.stack.addWidget(self.page_input)    # Index 1
        self.stack.addWidget(self.page_charts)   # Index 2
        self.stack.addWidget(self.page_algo)     # Index 3
        self.stack.addWidget(self.page_accuracy) # Index 4
    
    def train_model(self):
        self.page_algo.clear_logs()
        
        try:
            dataset, class_data, conjunto_data = self.controller.data_manager.get_full_dataset()
            params = self.page_algo.get_current_params()
            params["selected_features"] = self.page_input.get_selected_features()
            
            if not params["selected_features"]:
                raise Exception("Selecione pelo menos uma característica na aba de Entrada para treinar!")
            
            if not dataset or not class_data:
                raise Exception("Dataset não carregado.")
                
            self.current_model, data_dict = self.controller.train_model(dataset, class_data, conjunto_data, params)
            
            filtered_dataset = data_dict['filtered_dataset']
            filtered_class_data = data_dict['filtered_class_data']
            filtered_conjunto_data = data_dict['filtered_conjunto_data']
            
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
            
            # === CORREÇÃO: Limpa todas as checkboxes ANTES de forçar as iniciais ===
            for chk in self.page_charts.checkboxes:
                chk.blockSignals(True) # Evita disparar eventos no meio da limpeza
                chk.setChecked(False)
                chk.blockSignals(False)
            
            # Marca dinamicamente até 2 características baseadas na escolha do usuário
            checked_count = 0
            for chk in self.page_charts.checkboxes:
                if chk.text() in params["selected_features"]:
                    chk.setChecked(True)
                    checked_count += 1
                    if checked_count >= 2:
                        break
            
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
            preview_text = self.card_accuracy.preview_label.text().split("\n")
            resumo_acc = preview_text[1] if len(preview_text) > 1 else preview_text[0]

            msg = QMessageBox(self)
            msg.setWindowTitle("Treinamento Concluído")
            msg.setText(f"Modelo treinado com sucesso!\n\n{resumo_acc}")
            
        except Exception as e:
            self.page_algo.append_log(f"[ERRO CRÍTICO] {e}")
            import traceback
            traceback.print_exc()
            msg = QMessageBox(self)
            msg.setWindowTitle("Treinamento Interrompido")
            msg.setText(f"Ocorreu um erro:\n{e}")
            
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
        
        selected_features = getattr(self.current_model, 'selected_features', ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"])
        ponto = [current_inputs[k] for k in selected_features]
        
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
        
    def evaluate_current_model(self, _=None):
        """Calcula a Matriz e compara com outro modelo, se selecionado."""
        if not hasattr(self, 'current_model') or not hasattr(self, 'eval_data'):
            self.card_accuracy.update_preview("Aguardando modelo e dados para avaliação...")
            return

        from ml_engine import ClassificadorMetricas
        
        modo_selecionado = self.page_accuracy.combo_mode.currentText()
        modelo_comparacao = self.page_accuracy.combo_compare.currentText()
        
        filtered_dataset = self.eval_data["dataset"]
        filtered_class_data = self.eval_data["classes"]
        filtered_conjunto_data = self.eval_data["conjuntos"]
        keys = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]

        classes_unicas = list(sorted(set(filtered_class_data)))
        n = len(classes_unicas)
        class_to_idx = {c: i for i, c in enumerate(classes_unicas)}
        
        # Função auxiliar para rodar a matriz de um modelo
        def computar_matriz_modelo(modelo):
            matriz = [[0 for _ in range(n)] for _ in range(n)]
            total = 0
            for i in range(len(filtered_class_data)):
                conjunto = filtered_conjunto_data[i]
                if modo_selecionado == "Apenas Teste" and conjunto != "Teste": continue
                if modo_selecionado == "Apenas Treino" and conjunto != "Treino": continue

                total += 1
                real = filtered_class_data[i]
                selected_features = getattr(modelo, 'selected_features', keys)
                ponto = [filtered_dataset[k][i] for k in selected_features]
                
                res = modelo.predict(ponto)
                pred = res[0] if isinstance(res, tuple) else res
                
                real_idx = class_to_idx[real]
                pred_idx = class_to_idx.get(pred, -1)
                if pred_idx != -1:
                    matriz[real_idx][pred_idx] += 1
            return matriz, total

        # Roda para o modelo ATUAL
        matriz_atual, total_avaliado = computar_matriz_modelo(self.current_model)

        if total_avaliado > 0:
            metrics_current = ClassificadorMetricas(matriz_atual)
            metrics_compare = None
            
            # Tenta carregar e rodar o modelo COMPARADO
            if modelo_comparacao != "Nenhum":
                filepath = os.path.join(self.models_dir, f"{modelo_comparacao}.pkl")
                if os.path.exists(filepath):
                    try:
                        with open(filepath, 'rb') as f:
                            dados = pickle.load(f)
                            modelo_b = dados["model"]
                            matriz_comp, _ = computar_matriz_modelo(modelo_b)
                            metrics_compare = ClassificadorMetricas(matriz_comp)
                    except:
                        pass # Falhou em ler, ignora comparação
            
            self.page_accuracy.update_metrics(matriz_atual, classes_unicas, metrics_current, metrics_compare)
            
            self.card_accuracy.update_preview(
                f"[{modo_selecionado}]\nAcerto Geral: {metrics_current.acerto_geral()*100:.2f}%\n"
                f"Kappa: {metrics_current.coeficiente_kappa():.4f}\n\nClique para ver a Matriz e a Comparação."
            )
        else:
            self.card_accuracy.update_preview(f"Nenhum dado encontrado para: {modo_selecionado}")
    
    def open_history_dialog(self):
        """Abre o Pop-Up do Histórico e gerencia o modelo que o usuário escolher."""
        dialog = ModelHistoryDialog(self.models_dir, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Se o usuário confirmou a seleção, o nome do modelo estará aqui
            if dialog.selected_model:
                self.load_model_by_name(dialog.selected_model)

    def save_model(self):
        """Salva os parâmetros do modelo atual em um arquivo binário."""
        if not hasattr(self, 'current_model'):
            QMessageBox.warning(self, "Aviso", "Treine um modelo primeiro antes de salvar!")
            return

        nome, ok = QInputDialog.getText(self, "Salvar Treinamento", "Digite um nome para este modelo:")
        if ok and nome.strip():
            nome_arquivo = nome.strip()
            filepath = os.path.join(self.models_dir, f"{nome_arquivo}.pkl")
            
            dados_salvos = {
                "model": self.current_model,
                "eval_data": getattr(self, 'eval_data', None)
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(dados_salvos, f)

            self.page_algo.append_log(f"\n[SISTEMA] Modelo '{nome_arquivo}.pkl' salvo com sucesso!")
            QMessageBox.information(self, "Sucesso", f"O modelo '{nome_arquivo}' foi guardado no histórico!")

    def load_model_by_name(self, nome_modelo):
        """Carrega e restaura as informações do arquivo .pkl selecionado."""
        filepath = os.path.join(self.models_dir, f"{nome_modelo}.pkl")
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    dados_salvos = pickle.load(f)

                self.current_model = dados_salvos["model"]
                
                # Restaura os dados para plotagem e acurácia automática se existirem
                if dados_salvos.get("eval_data"):
                    self.eval_data = dados_salvos["eval_data"]
                    
                    # Atualiza os gráficos
                    self.page_charts.set_trained_model(self.current_model)
                    self.page_charts.set_dataset(
                        self.eval_data["dataset"], 
                        self.eval_data["classes"], 
                        self.eval_data["conjuntos"]
                    )
                    self.page_charts.plot_custom_chart()
                    
                    # Atualiza a aba de Acurácia
                    modo_atual = self.page_accuracy.combo_mode.currentText()
                    self.evaluate_current_model(modo_atual)

                self.page_algo.append_log(f"\n[SISTEMA] O modelo '{nome_modelo}' foi carregado com sucesso do histórico!")
                QMessageBox.information(self, "Sucesso", f"Modelo '{nome_modelo}' carregado e pronto para classificar novos pontos!")
                
            except Exception as e:
                QMessageBox.critical(self, "Erro ao Carregar", f"Ocorreu um erro ao carregar o arquivo binário:\n{e}")
    
    def open_accuracy_page(self):
        """Função chamada ao clicar no card de Acurácia. Popula o ComboBox e abre a página correta."""
        # 1. Guarda o que estava selecionado para não perder a referência
        atual_selecionado = self.page_accuracy.combo_compare.currentText()
        
        # Bloqueia os sinais para não disparar recálculos enquanto adiciona os itens
        self.page_accuracy.combo_compare.blockSignals(True) 
        self.page_accuracy.combo_compare.clear()
        self.page_accuracy.combo_compare.addItem("Nenhum")
        
        # 2. Varre a pasta e adiciona os modelos salvos
        if hasattr(self, 'models_dir') and os.path.exists(self.models_dir):
            arquivos = [f for f in os.listdir(self.models_dir) if f.endswith(".pkl")]
            for f in arquivos:
                nome_limpo = f.replace(".pkl", "")
                self.page_accuracy.combo_compare.addItem(nome_limpo)
                
        # 3. Restaura a seleção se ela ainda existir
        idx = self.page_accuracy.combo_compare.findText(atual_selecionado)
        if idx >= 0:
            self.page_accuracy.combo_compare.setCurrentIndex(idx)
        else:
            self.page_accuracy.combo_compare.setCurrentIndex(0)
            
        self.page_accuracy.combo_compare.blockSignals(False)
        
        # 4. Atualiza a matriz e a tabela antes de mostrar a tela
        self.evaluate_current_model()
        
        # 5. A SOLUÇÃO: Troca de página usando a referência direta do widget
        self.stack.setCurrentWidget(self.page_accuracy)
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    setup_pyqtgraph() 
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
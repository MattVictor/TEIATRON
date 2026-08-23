# TEIATRON

Um dashboard interativo e educacional para análise, treinamento e visualização de algoritmos clássicos de Machine Learning. 

O grande diferencial do **TEIATRON** é a sua transparência: o motor de inteligência artificial foi desenvolvido **100% do zero usando matemática pura**. Nenhuma biblioteca de Machine Learning de alto nível (como Scikit-Learn ou TensorFlow) foi utilizada para o back-end dos modelos clássicos. O objetivo é permitir que estudantes e desenvolvedores vejam exatamente como a matemática funciona "por debaixo do capô".

---

## 🎯 Objetivo do Projeto
Desmistificar algoritmos de classificação através de uma interface visual rica, permitindo que o usuário acompanhe o treinamento passo a passo, visualize hiperplanos de decisão projetados no espaço e valide o aprendizado através de métricas estatísticas detalhadas.

## 🚀 Principais Funcionalidades

* **Algoritmos Implementados do Zero:**
  * Classificador de Distância Mínima (Centróides)
  * Classificador de Distância Máxima (Minimização do pior caso)
  * Perceptron Clássico e Regra Delta (Adaline)
  * Bayes Ótimo (QDA/LDA) e Naive Bayes
  * Rede Neural Artificial (MLP)
  * Máquina de Vetores de Suporte (SVM)
* **Estratégias de Classificação:** Suporte nativo a problemas Binários e Multiclasse utilizando a abordagem *Um-Contra-Todos* (OvA).
* **Visualização Matemática Avançada:** 
  * Gráficos de dispersão interativos com projeção de fronteiras de decisão (Retas, Curvas, Isocurvas) extraídas a partir da Álgebra Linear.
  * Curvas de aprendizado (MSE) em tempo real para Redes Neurais e Perceptrons.
* **Métricas Estatísticas (Testes e Avaliação):**
  * Divisão dinâmica de amostras (Train/Test Split) com amostragem estratificada automática.
  * Geração de Matriz de Confusão para avaliação dos modelos.
  * Cálculo rigoroso de Acurácia Global, Acurácia do Produtor, Acurácia do Usuário, Kappa, Tau, Matthews (MCC) e Fb-Score.
  * Comparação entre modelos (Teste de McNemar).
  * Análise de Normalidade Multivariada (Mardia, Henze-Zirkler).

---

## 📁 Arquitetura do Sistema (MVC)

O TEIATRON foi reescrito para utilizar uma arquitetura altamente escalável e modular, baseada nos princípios MVC (Model-View-Controller):

* **`algorithms/`**: Coração matemático do projeto. Cada classificador possui um arquivo próprio herdado de `BaseClassifier`. Todo o cálculo é feito utilizando Python puro e numpy matricial.
* **`core/`**: Regras de negócio e Controladores. Aqui residem o `controller.py` (que faz a ponte entre algoritmos e interface), o `data_manager.py` (gerenciador de arquivos e splits de dados) e o `plot_engine.py` (um conversor agnóstico que extrai e resolve equações algébricas matemáticas para desenhá-las na interface gráfica).
* **`views/`**: Componentes puramente visuais (PyQt6). Separa de forma elegante a entrada de dados, gráficos, os logs de treinamento e as abas de acurácia.
* **`models/`**: Diretório reservado para o salvamento e carregamento contínuo de modelos treinados (`.pkl`).

---

## 🛠️ Dependências e Instalação

O projeto foi construído em **Python 3.10+**. 

### Como Instalar:
1. Clone o repositório na sua máquina local:
   ```bash
   git clone https://github.com/SEU_USUARIO/TEIATRON.git
   cd TEIATRON
   ```
2. Crie e ative um ambiente virtual (Opcional, mas recomendado):
   ```bash
   python -m venv venv
   # No Windows:
   venv\Scripts\activate
   # No Linux/Mac:
   source venv/bin/activate
   ```
3. Instale as dependências executando:
   ```bash
   pip install -r requirements.txt
   ```
   *(Dependências incluem: `PyQt6` e `pyqtgraph` para a interface, `numpy` para as matrizes, `pandas` e `pingouin` para testes estatísticos.)*

---

## ▶️ Como Usar

Para iniciar a aplicação, basta executar o ponto de entrada na raiz do projeto:

```bash
python main.py
```

1. Na aba **Entrada de Dados**, clique em Importar Dataset e escolha seu arquivo `.csv` (como o `Iris_data.csv`).
2. Os dados serão automaticamente embaralhados e divididos em Treino e Teste.
3. Escolha seu Algoritmo na aba de Treinamento, configure os Hiperparâmetros e inicie a execução.
4. Analise as matrizes e pesos gerados no Terminal/Log em tempo real.
5. Verifique a Acurácia e salve o modelo para uso posterior.

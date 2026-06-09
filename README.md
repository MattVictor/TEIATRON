# TEIATRON

Um dashboard interativo e educacional para análise, treinamento e visualização de algoritmos clássicos de Machine Learning. 

O grande diferencial do **TEIATRON** é a sua transparência: o motor de inteligência artificial foi desenvolvido **100% do zero usando matemática pura**. Nenhuma biblioteca de Machine Learning (como Scikit-Learn ou TensorFlow) foi utilizada para os cálculos. O objetivo é permitir que estudantes e desenvolvedores vejam exatamente como a matemática funciona "por debaixo do capô".

---

## Objetivo do Projeto
Desmistificar algoritmos de classificação através de uma interface visual rica, permitindo que o usuário acompanhe o treinamento passo a passo, visualize hiperplanos de decisão projetados no espaço e valide o aprendizado através de métricas estatísticas detalhadas.

## Principais Funcionalidades

* **Algoritmos Implementados do Zero:**
  * Classificador de Distância Mínima (Centróides)
  * Classificador de Distância Máxima (Minimização do pior caso)
  * Perceptron Clássico (Função Degrau)
  * Regra Delta / Adaline (Otimização Contínua)
* **Estratégias de Classificação:** Suporte nativo a problemas Binários e Multiclasse utilizando a abordagem *Um-Contra-Todos* (OvA).
* **Visualização Matemática Avançada:** * Gráficos de dispersão com projeção de hiperplanos 4D em planos 2D (utilizando o cálculo de *Bias Efetivo*).
  * Gráficos de linha acompanhando a queda dos erros de época em época.
* **Métricas Estatísticas:** Geração dinâmica de Matriz de Confusão e cálculo de coeficientes como Acurácia Global, Produtor, Usuário, Kappa, Tau, Matthews (MCC) e Fb-Score.
* **Problema do XOR:** Módulo especial demonstrando a incapacidade de modelos lineares na resolução do clássico problema do Ou-Exclusivo.

---

## ⚙️ Dependências e Instalação

O projeto foi construído em **Python 3.10+**. A lógica matemática roda com bibliotecas nativas, sendo as dependências externas focadas apenas na interface gráfica (GUI).

### Bibliotecas Necessárias:
* `PyQt6` (Criação das janelas, botões e layout)
* `pyqtgraph` (Renderização de alto desempenho para os gráficos)
* `numpy` (Utilizado exclusivamente para vetorização de coordenadas na camada gráfica, não afeta o motor de ML)

### Como Instalar:
1. Clone o repositório:
   ```bash
   git clone [https://github.com/SEU_USUARIO/TEIATRON.git](https://github.com/SEU_USUARIO/TEIATRON.git)
   cd TEIATRON

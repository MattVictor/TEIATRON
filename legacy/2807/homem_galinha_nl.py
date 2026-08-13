import math
import tkinter as tk
from tkinter import messagebox, scrolledtext
import matplotlib

# Forçamos o Matplotlib a usar o backend nativo do Tkinter para evitar conflitos
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

# ==========================================
# 1. CLASSE DA REDE NEURAL
# ==========================================
class RedeNeuralPR711:
    def __init__(self, taxa_aprendizagem=0.05):
        self.eta = taxa_aprendizagem
        self.w1, self.w2, self.w3, self.w4 = 0.15, 0.20, 0.25, 0.30
        self.w5, self.w6, self.w7, self.w8 = 0.40, 0.45, 0.50, 0.55
        self.b1, self.b2 = 0.35, 0.60

    def sigmoide(self, x):
        return 1 / (1 + math.exp(-x))

    def derivada_sigmoide(self, out):
        return out * (1 - out)
        
    def propagar_frente(self, i1, i2):
        out_h1 = self.sigmoide((self.w1 * i1) + (self.w2 * i2) + self.b1)
        out_h2 = self.sigmoide((self.w3 * i1) + (self.w4 * i2) + self.b1)
        out_o1 = self.sigmoide((self.w5 * out_h1) + (self.w6 * out_h2) + self.b2)
        out_o2 = self.sigmoide((self.w7 * out_h1) + (self.w8 * out_h2) + self.b2)
        return out_h1, out_h2, out_o1, out_o2

    def treinar(self, dados_treino, epocas=10000):
        historico_erros = []
        for epoca in range(epocas):
            erro_epoca = 0
            for (i1, i2, target_o1, target_o2) in dados_treino:
                # Feedforward
                out_h1, out_h2, out_o1, out_o2 = self.propagar_frente(i1, i2)
                
                # Erro
                erro_epoca += 0.5 * ((target_o1 - out_o1)**2) + 0.5 * ((target_o2 - out_o2)**2)
                
                # Backpropagation
                delta_o1 = (out_o1 - target_o1) * self.derivada_sigmoide(out_o1)
                delta_o2 = (out_o2 - target_o2) * self.derivada_sigmoide(out_o2)
                delta_h1 = (delta_o1 * self.w5 + delta_o2 * self.w7) * self.derivada_sigmoide(out_h1)
                delta_h2 = (delta_o1 * self.w6 + delta_o2 * self.w8) * self.derivada_sigmoide(out_h2)
                
                # Atualização
                self.w5 -= self.eta * (delta_o1 * out_h1)
                self.w6 -= self.eta * (delta_o1 * out_h2)
                self.w7 -= self.eta * (delta_o2 * out_h1)
                self.w8 -= self.eta * (delta_o2 * out_h2)
                self.b2 -= self.eta * (delta_o1 + delta_o2) 
                
                self.w1 -= self.eta * (delta_h1 * i1)
                self.w2 -= self.eta * (delta_h1 * i2)
                self.w3 -= self.eta * (delta_h2 * i1)
                self.w4 -= self.eta * (delta_h2 * i2)
                self.b1 -= self.eta * (delta_h1 + delta_h2)
            
            historico_erros.append(erro_epoca / len(dados_treino))
            
        return historico_erros

# ==========================================
# 2. INTERFACE GRÁFICA (Tkinter)
# ==========================================
class AppGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Classificador Homem vs Galinha (PR_711)")
        self.root.geometry("450x550")
        
        self.rede = None
        self.dados_treino = [
            (0.1, 0.2, 1.0, 0.0), # Homem
            (0.8, 0.9, 0.0, 1.0)  # Galinha
        ]

        self.construir_interface()

    def construir_interface(self):
        # ---- PAINEL DE TREINAMENTO ----
        frame_treino = tk.LabelFrame(self.root, text="Configurações de Treinamento", padx=10, pady=10)
        frame_treino.pack(padx=10, pady=10, fill="x")

        tk.Label(frame_treino, text="Taxa de Aprendizagem (η):").grid(row=0, column=0, sticky="w")
        self.ent_eta = tk.Entry(frame_treino)
        self.ent_eta.insert(0, "0.05")
        self.ent_eta.grid(row=0, column=1, pady=5)

        tk.Label(frame_treino, text="Número de Épocas:").grid(row=1, column=0, sticky="w")
        self.ent_epocas = tk.Entry(frame_treino)
        self.ent_epocas.insert(0, "10000")
        self.ent_epocas.grid(row=1, column=1, pady=5)

        self.btn_treinar = tk.Button(frame_treino, text="1. Iniciar Treinamento", bg="#4CAF50", fg="white", 
                                     font=("Arial", 10, "bold"), command=self.acao_treinar)
        self.btn_treinar.grid(row=2, column=0, columnspan=2, pady=10, ipadx=50)

        # ---- PAINEL DE CLASSIFICAÇÃO ----
        frame_class = tk.LabelFrame(self.root, text="Classificar Novo Padrão", padx=10, pady=10)
        frame_class.pack(padx=10, pady=10, fill="x")

        tk.Label(frame_class, text="Entrada 1 (i1):").grid(row=0, column=0, sticky="w")
        self.ent_i1 = tk.Entry(frame_class)
        self.ent_i1.insert(0, "0.1")
        self.ent_i1.grid(row=0, column=1, pady=5)

        tk.Label(frame_class, text="Entrada 2 (i2):").grid(row=1, column=0, sticky="w")
        self.ent_i2 = tk.Entry(frame_class)
        self.ent_i2.insert(0, "0.2")
        self.ent_i2.grid(row=1, column=1, pady=5)

        self.btn_classificar = tk.Button(frame_class, text="2. Classificar Padrão", bg="#2196F3", fg="white", 
                                         font=("Arial", 10, "bold"), state=tk.DISABLED, command=self.acao_classificar)
        self.btn_classificar.grid(row=2, column=0, columnspan=2, pady=10, ipadx=50)

        # ---- LOG DE SAÍDA ----
        tk.Label(self.root, text="Log de Eventos:").pack(anchor="w", padx=10)
        self.log = scrolledtext.ScrolledText(self.root, width=50, height=10)
        self.log.pack(padx=10, pady=5)
        self.imprimir_log("Pronto para treinar. Ajuste os parâmetros e clique em 'Iniciar Treinamento'.")

    def imprimir_log(self, mensagem):
        """Função auxiliar para escrever no painel de texto."""
        self.log.insert(tk.END, mensagem + "\n")
        self.log.see(tk.END) # Rola para o final
        self.root.update()   # Força a interface a se atualizar imediatamente

    def acao_treinar(self):
        try:
            eta = float(self.ent_eta.get())
            epocas = int(self.ent_epocas.get())
        except ValueError:
            messagebox.showerror("Erro", "Por favor, insira valores numéricos válidos.")
            return

        self.btn_treinar.config(state=tk.DISABLED)
        self.imprimir_log(f"\n[!] Iniciando treinamento... (η={eta}, Épocas={epocas})")
        self.imprimir_log("Aguarde, isso pode levar alguns segundos...")
        
        # Treinamento
        self.rede = RedeNeuralPR711(taxa_aprendizagem=eta)
        erros = self.rede.treinar(self.dados_treino, epocas=epocas)
        
        self.imprimir_log(f"[✓] Treinamento Concluído! Erro final: {erros[-1]:.6f}")
        self.btn_treinar.config(state=tk.NORMAL)
        self.btn_classificar.config(state=tk.NORMAL)
        
        # Exibe o gráfico do Matplotlib de forma segura
        plt.figure("Curva de Aprendizagem")
        plt.plot(erros, color='red', label='Erro Médio')
        plt.title("Erro Médio vs Épocas")
        plt.xlabel("Épocas")
        plt.ylabel("Erro Médio (MSE)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def acao_classificar(self):
        if not self.rede:
            return
            
        try:
            i1 = float(self.ent_i1.get())
            i2 = float(self.ent_i2.get())
        except ValueError:
            messagebox.showerror("Erro", "Por favor, insira valores numéricos válidos para as entradas.")
            return
            
        out_h1, out_h2, out_o1, out_o2 = self.rede.propagar_frente(i1, i2)
        
        self.imprimir_log(f"\n--- Testando Padrão: [{i1}, {i2}] ---")
        self.imprimir_log(f"Saída Bruta: Homem={out_o1:.4f}, Galinha={out_o2:.4f}")
        
        if out_o1 > out_o2:
            self.imprimir_log("=> RESULTADO: HOMEM")
        else:
            self.imprimir_log("=> RESULTADO: GALINHA")

if __name__ == "__main__":
    # Inicializa o Tkinter
    root = tk.Tk()
    app = AppGUI(root)
    # Roda o loop principal da interface
    root.mainloop()
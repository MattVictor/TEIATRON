# config.py
import pyqtgraph as pg

# Paleta de Alto Contraste (Dark Mode)
BG_MAIN = "#121212"
BG_CARD = "#1E1E1E"
TEXT_PRIMARY = "#FFFFFF"
TEXT_SECONDARY = "#A0A0A0"
ACCENT_COLOR = "#00E5FF"  # Ciano brilhante
ACCENT_TEXT = "#000000"   # Texto preto para contrastar
WARNING_COLOR = "#FFEA00" # Amarelo

# # Cores de Fundo (Light Mode)
# BG_MAIN = "#F4F6F9"       # Fundo geral da janela (cinza muito claro)
# BG_CARD = "#FFFFFF"       # Fundo dos cards (branco puro)

# # Cores de Texto
# TEXT_PRIMARY = "#2C3E50"  # Texto principal (azul escuro/quase preto, descansa mais a vista que o preto puro)
# TEXT_SECONDARY = "#7F8C8D" # Texto secundário (cinza médio)

# # Cores de Destaque
# ACCENT_COLOR = "#0078D7"  # Azul moderno (estilo Windows/Material Design)
# ACCENT_TEXT = "#FFFFFF"   # Texto sobre o botão de destaque (branco fica perfeito no azul)
# WARNING_COLOR = "#D32F2F" # Vermelho alerta vivo e com bom contraste no fundo claro

# # Cores das Classes no Gráfico (Ajustadas para ficarem vibrantes no fundo branco)
# CLASS_COLORS = [
#     "#E53935", # Vermelho
#     "#1E88E5", # Azul
#     "#43A047"  # Verde
# ]

def setup_pyqtgraph():
    """Configurações globais do pyqtgraph."""
    pg.setConfigOption('background', BG_CARD)
    pg.setConfigOption('foreground', TEXT_PRIMARY)
    pg.setConfigOption('antialias', True)
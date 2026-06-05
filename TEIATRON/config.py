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

def setup_pyqtgraph():
    """Configurações globais do pyqtgraph."""
    pg.setConfigOption('background', BG_CARD)
    pg.setConfigOption('foreground', TEXT_PRIMARY)
    pg.setConfigOption('antialias', True)
# base_components.py
from PyQt6.QtWidgets import QFrame, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from config import *

class BaseCard(QFrame):
    """Molde base para os Cards do Dashboard."""
    def __init__(self, title, on_expand_callback):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(f"""
            BaseCard {{ background-color: {BG_CARD}; border: 2px solid #333333; border-radius: 12px;}}
            BaseCard:hover {{ border: 2px solid {ACCENT_COLOR}; }}
        """)
        
        self.layout = QVBoxLayout(self)
        
        top_bar = QHBoxLayout()
        btn_expand = QPushButton("+")
        btn_expand.setFixedSize(32, 32)
        btn_expand.setStyleSheet(f"""
            QPushButton {{ background-color: {ACCENT_COLOR}; color: {ACCENT_TEXT}; border-radius: 16px; font-weight: bold; font-size: 18px; }}
            QPushButton:hover {{ background-color: #00B3CC; }}
        """)
        btn_expand.clicked.connect(on_expand_callback)
        
        title_label = QLabel(title)
        title_label.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        title_label.setStyleSheet(f"color: {TEXT_PRIMARY}; text-align: center;")
        
        top_bar.addWidget(btn_expand)
        top_bar.addWidget(title_label)
        top_bar.addStretch()
        self.layout.addLayout(top_bar)
        
    def add_preview_content(self, widget):
        self.layout.addWidget(widget, stretch=1)

class BaseExpandedPage(QWidget):
    """Molde base: Agora simula um 'Card Gigante' expandido."""
    def __init__(self, title, on_back_callback):
        super().__init__()
        
        # Layout da página inteira (com margens para separar o card da borda da janela)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Criando o frame que simula o Card Expandido
        self.card_frame = QFrame()
        self.card_frame.setObjectName("ExpandedCard")
        self.card_frame.setStyleSheet(f"""
            QFrame#ExpandedCard {{ 
                background-color: {BG_CARD}; 
                border: 2px solid {ACCENT_COLOR}; 
                border-radius: 12px; 
            }}
        """)
        
        # Layout interno do card
        self.card_layout = QVBoxLayout(self.card_frame)
        self.card_layout.setContentsMargins(25, 25, 25, 25)
        
        # Barra Superior: Título na esquerda, Botão '-' na direita
        top_bar = QHBoxLayout()
        
        title_label = QLabel(title)
        title_label.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        title_label.setStyleSheet(f"color: {ACCENT_COLOR}; border: none;") 
        
        btn_collapse = QPushButton("-")
        btn_collapse.setFixedSize(32, 32)
        btn_collapse.setStyleSheet(f"""
            QPushButton {{ 
                background-color: {ACCENT_COLOR}; 
                color: {ACCENT_TEXT}; 
                border-radius: 16px; 
                font-weight: bold; 
                font-size: 26px; 
                padding-bottom: 4px; /* Ajuste sutil para centralizar o texto do traço */
            }}
            QPushButton:hover {{ background-color: #00B3CC; }}
        """)
        btn_collapse.clicked.connect(on_back_callback)
        
        top_bar.addWidget(title_label)
        top_bar.addStretch() # Empurra o título para a esquerda e o botão para a direita
        top_bar.addWidget(btn_collapse)
        
        self.card_layout.addLayout(top_bar)
        
        # Linha divisória abaixo do título (Design clean)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("background-color: #444444; max-height: 1px; margin-top: 5px; margin-bottom: 15px;")
        self.card_layout.addWidget(line)
        
        # Adiciona o card estilizado ao layout invisível da página
        main_layout.addWidget(self.card_frame)
        
    def add_main_content(self, widget):
        """Os conteúdos das páginas específicas agora são injetados diretamente dentro do card_layout."""
        self.card_layout.addWidget(widget, stretch=1)
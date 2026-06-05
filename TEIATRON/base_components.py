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
            BaseCard {{ background-color: {BG_CARD}; border: 2px solid #333333; border-radius: 12px; }}
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
        title_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        title_label.setStyleSheet(f"color: {TEXT_PRIMARY};")
        
        top_bar.addWidget(btn_expand)
        top_bar.addWidget(title_label)
        top_bar.addStretch()
        self.layout.addLayout(top_bar)
        
    def add_preview_content(self, widget):
        self.layout.addWidget(widget, stretch=1)

class BaseExpandedPage(QWidget):
    """Molde base para as Páginas Expandidas."""
    def __init__(self, title, on_back_callback):
        super().__init__()
        self.layout = QVBoxLayout(self)
        
        btn_back = QPushButton("← Voltar")
        btn_back.setFixedSize(120, 35)
        btn_back.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        btn_back.setStyleSheet(f"""
            QPushButton {{ background-color: #333333; color: {TEXT_PRIMARY}; border-radius: 6px; }}
            QPushButton:hover {{ background-color: #444444; color: {ACCENT_COLOR}; }}
        """)
        btn_back.clicked.connect(on_back_callback)
        self.layout.addWidget(btn_back)

        title_label = QLabel(title)
        title_label.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        title_label.setStyleSheet(f"color: {ACCENT_COLOR}; margin-bottom: 10px;")
        self.layout.addWidget(title_label)
        
    def add_main_content(self, widget):
        self.layout.addWidget(widget, stretch=1)
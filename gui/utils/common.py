from PyQt6.QtCore import pyqtSignal, QObject
from PyQt6.QtGui import QFont

# --- 配置与信号 ---
FONT_TITLE = QFont("Microsoft YaHei", 10, QFont.Weight.Bold)
FONT_LABEL = QFont("Microsoft YaHei", 9)
FONT_DATA = QFont("Consolas", 10)


class NodeSignal(QObject):
    selected = pyqtSignal(object)


NODE_SIGNALS = NodeSignal()


class SocketData():
    def __init__(self, name: str, volume: int):
        super().__init__()
        self.name = name
        self.volume = volume

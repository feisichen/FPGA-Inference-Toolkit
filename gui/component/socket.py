from typing import Dict, Any
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPen, QColor, QBrush, QPainter
from PyQt6.QtWidgets import (QGraphicsItem)

from gui.component.base_elem import Serializable
from gui.utils.common import FONT_LABEL, NODE_SIGNALS, SocketData


class Socket(QGraphicsItem, Serializable):
    def __init__(self, parent, socket_data, is_input=True, is_reentrant=False):
        super().__init__(parent)
        self.is_input = is_input
        self.socket_data = socket_data
        self.edges = []
        self.is_reentrant = is_reentrant
        self.setZValue(10)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)

    def add_edge(self, edge):
        if edge not in self.edges: self.edges.append(edge)

    def is_occupied(self):
        if self.is_reentrant:
            return False
        return len(self.edges) > 0

    def boundingRect(self):
        return QRectF(0, 0, 24, 24)

    def remove_edge(self, edge):
        if edge in self.edges:
            self.edges.remove(edge)
        parent = self.parentItem()
        parent.remove_edge(edge)

    def paint(self, painter, option, widget):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor("#2ecc71") if self.is_input else QColor("#e74c3c")
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(Qt.GlobalColor.white, 2))
        painter.drawEllipse(4, 4, 16, 16)
        painter.setFont(FONT_LABEL)
        painter.setPen(QColor("#ffffff"))
        if self.is_reentrant:
            painter.drawText(QRectF(-40, 25, 100, 40), Qt.AlignmentFlag.AlignCenter, str(self.socket_data.name))
        else:
            if self.is_input:
                painter.drawText(25, 17, str(self.socket_data.name))
            else:
                painter.drawText(-100, 17, 95, 20, Qt.AlignmentFlag.AlignRight, str(self.socket_data.name))

    def mousePressEvent(self, event):
        NODE_SIGNALS.selected.emit(self)
        super().mousePressEvent(event)

    # ========== Serializable 接口实现 ==========
    def serialize(self) -> Dict[str, Any]:
        """序列化 Socket 数据"""
        return {
            "name": self.socket_data.name,
            "volume": self.socket_data.volume,
            "is_input": self.is_input,
            "is_reentrant": self.is_reentrant,
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any], parent=None):
        """从数据创建 Socket"""
        socket_data = SocketData(data["name"], data["volume"])
        return cls(parent, socket_data, data["is_input"], data.get("is_reentrant", False))

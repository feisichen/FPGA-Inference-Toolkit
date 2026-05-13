from typing import Dict, Any
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QPen, QPainterPath, QColor
from PyQt6.QtWidgets import (QGraphicsItem, QGraphicsPathItem)

from gui.component.base_elem import Serializable
from gui.utils.common import NODE_SIGNALS


class Edge(QGraphicsPathItem, Serializable):
    def __init__(self, start_socket, end_socket, option=None):
        super().__init__()
        self.start_socket = start_socket
        self.end_socket = end_socket
        self.start_socket.add_edge(self)
        self.end_socket.add_edge(self)
        self.option = option
        self.depth = 0
        self.coarse_parallelism = 1
        self.base_name = ""
        self.extra_name = ""
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setPen(QPen(QColor("#3498db"), 3))
        self.setZValue(0)
        self.update_path()

    def update_path(self):
        p1 = self.start_socket.scenePos() + QPointF(12, 12)
        p2 = self.end_socket.scenePos() + QPointF(12, 12)
        path = QPainterPath()
        path.moveTo(p1)
        dist = abs(p2.y() - p1.y()) * 0.5
        path.cubicTo(p1.x(), p1.y() + dist, p2.x(), p2.y() - dist, p2.x(), p2.y())
        self.setPath(path)

    def paint(self, painter, option, widget):
        color = QColor("#f1c40f") if self.isSelected() else QColor("#3498db")
        width = 5 if self.isSelected() else 3
        self.setPen(QPen(color, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        super().paint(painter, option, widget)

    # 点击连线时发射信号
    def mousePressEvent(self, event):
        NODE_SIGNALS.selected.emit(self)
        super().mousePressEvent(event)

    def remove_self(self):
        self.start_socket.remove_edge(self)
        self.end_socket.remove_edge(self)
        # if self in self.end_socket.edges: self.end_socket.edges.remove(self)
        if self.scene(): self.scene().removeItem(self)

    def set_depth(self, depth):
        self.depth = depth

    def get_base_name(self):
        return self.base_name

    def get_total_name(self):
        return self.base_name + self.extra_name

    # ========== Serializable 接口实现 ==========
    def serialize(self) -> Dict[str, Any]:
        """序列化 Edge 数据"""
        return {
            "depth": self.depth,
            "coarse_parallelism": self.coarse_parallelism,
            # "base_name": self.base_name,
            # "extra_name": self.extra_name,
            "option": self.option,
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any], start_socket, end_socket):
        """从数据创建 Edge"""
        edge = cls(start_socket, end_socket, data.get("option"))
        edge.depth = data.get("depth", 1)
        edge.coarse_parallelism = data.get("coarse_parallelism", 1)
        # edge.base_name = data.get("base_name", "")
        # edge.extra_name = data.get("extra_name", "")
        return edge

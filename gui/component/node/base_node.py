from abc import abstractmethod
from typing import Dict, Any, Type

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPen, QColor, QPainter
from PyQt6.QtWidgets import (QGraphicsItem)

from gui.component.base_elem import Serializable
from gui.component.socket import Socket
from gui.utils.common import FONT_TITLE, NODE_SIGNALS, SocketData


class BaseNode(QGraphicsItem, Serializable):
    @classmethod
    def get_node_class(cls, class_name: str) -> Type['BaseNode']:
        """根据类名获取 Node 类（从 config.register_gui_cls 中查找）"""
        from config.config import register_gui_cls
        for label, node_class in register_gui_cls:
            if node_class.__name__ == class_name:
                return node_class
        return None

    def __init__(self, name, view_name, default_depth, inputs, outputs):
        super().__init__()
        self.name = name
        self.view_name = view_name
        self.default_depth = default_depth
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges)
        self.width, self.height = 200, 120
        self.in_sockets = [Socket(self, d, True) for d in inputs]
        self.out_sockets = [Socket(self, d, False) for d in outputs]
        self._arrange_sockets()

    def _arrange_sockets(self):
        for i, s in enumerate(self.in_sockets): s.setPos(-12, 40 + i * 35)
        for i, s in enumerate(self.out_sockets): s.setPos(self.width - 12, 40 + i * 35)

    def boundingRect(self):
        return QRectF(-10, -10, self.width + 20, self.height + 50)

    def paint(self, painter, option, widget):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor("#2d2d30"))
        pen_color = QColor("#0078d4") if self.isSelected() else QColor("#ffffff")
        painter.setPen(QPen(pen_color, 2 if not self.isSelected() else 4))
        painter.drawRoundedRect(0, 0, self.width, self.height, 8, 8)
        painter.setFont(FONT_TITLE)
        painter.setPen(Qt.GlobalColor.white)
        painter.drawText(0, 5, self.width, 30, Qt.AlignmentFlag.AlignCenter, self.view_name)

    def mousePressEvent(self, event):
        NODE_SIGNALS.selected.emit(self)
        super().mousePressEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemScenePositionHasChanged:
            for s in self.in_sockets + self.out_sockets:
                for e in s.edges: e.update_path()
        return super().itemChange(change, value)

    def remove_edge(self, edge):
        return

    def print_node(self):
        pass

    @abstractmethod
    def gen_channel_code(self, base_name, depth_val, output):
        pass

    @abstractmethod
    def gen_node_code(self, output):
        pass

    @classmethod
    @abstractmethod
    def init_widget(cls, panel):
        pass

    @classmethod
    @abstractmethod
    def hide_widget(cls, item):
        pass

    @abstractmethod
    def set_widget_value(self, item, panel):
        pass

    # ========== Serializable 接口实现 ==========
    def serialize(self) -> Dict[str, Any]:
        """序列化 Node 数据，子类应重写此方法添加特有属性"""
        return {
            "class": self.__class__.__name__,
            "name": self.name,
            "view_name": self.view_name,
            "pos": (self.pos().x(), self.pos().y()),
            "width": self.width,
            "height": self.height,
            "in_sockets": [s.serialize() for s in self.in_sockets],
            "out_sockets": [s.serialize() for s in self.out_sockets],
        }

    def _serialize_extra(self) -> Dict[str, Any]:
        """子类重写此方法来序列化特有属性"""
        return {}

    @classmethod
    def deserialize(cls, data: Dict[str, Any], scene=None):
        """从数据创建 Node，子类一般不需要重写此方法"""
        node_class = cls.get_node_class(data["class"])
        if node_class is None:
            raise ValueError(f"未注册的 Node 类型: {data['class']}")

        # 子类需要实现 _deserialize_create 来创建实例
        node = node_class._deserialize_create(data)
        node.setPos(*data["pos"])
        if "width" in data:
            node.width = data["width"]
        if "height" in data:
            node.height = data["height"]

        # 子类需要实现 _deserialize_extra 来恢复特有属性
        node._deserialize_extra(data)
        return node

    @classmethod
    def _deserialize_create(cls, data: Dict[str, Any]) -> 'BaseNode':
        """子类必须重写此方法来创建实例"""
        raise NotImplementedError("子类必须实现 _deserialize_create")

    def _deserialize_extra(self, data: Dict[str, Any]):
        """子类重写此方法来恢复特有属性"""
        pass

    @classmethod
    @abstractmethod
    def is_creatable(cls):
        pass

    @classmethod
    @abstractmethod
    def is_deletable(cls):
        pass
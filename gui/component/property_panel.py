from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QFormLayout, QVBoxLayout,
                             QWidget, QHBoxLayout,
                             QLabel, QSpinBox, QCheckBox)

from config.config import register_gui_cls
from gui.component.edge import Edge
from gui.component.node.base_node import BaseNode
from gui.component.node.global_memory_node import GlobalMemoryNode
from gui.component.socket import Socket


class PropertyPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.current_item = None
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(15, 20, 15, 20)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # 1. 标题
        self.title = QLabel("属性面板")
        self.title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.layout.addWidget(self.title)

        # 2. 动态文本属性容器 (使用 QFormLayout)
        self.info_widget = QWidget()
        self.info_layout = QFormLayout(self.info_widget)
        self.info_layout.setContentsMargins(0, 10, 0, 10)
        self.layout.addWidget(self.info_widget)

        # 3. 编辑控件 (Edge 专用)
        self.edge_depth = QWidget()
        e_lay = QHBoxLayout(self.edge_depth)
        e_lay.addWidget(QLabel("通道缓冲区大小 (Depth):"))
        self.size_spin = QSpinBox()
        self.size_spin.setRange(0, 2147483647)
        self.size_spin.valueChanged.connect(self.on_depth_changed)
        e_lay.addWidget(self.size_spin)
        self.layout.addWidget(self.edge_depth)
        self.edge_depth.hide()

        self.widget_map = {}

        for name, cls in register_gui_cls:
            cls.init_widget(self)

    def clear_info(self):
        """清空动态生成的表单行"""
        while self.info_layout.rowCount() > 0:
            self.info_layout.removeRow(0)

    def add_info_row(self, label, value):
        """方便地添加一行显示信息"""
        val_label = QLabel(str(value))
        val_label.setStyleSheet("color: #666;")  # 调浅颜色区分标签和值
        self.info_layout.addRow(f"{label}:", val_label)

    def set_item(self, item):
        self.current_item = item
        self.clear_info()

        # 隐藏所有编辑控件
        self.edge_depth.hide()
        for name, cls in register_gui_cls:
            cls.hide_widget(item)

        if isinstance(item, BaseNode):
            self.title.setText("节点属性")
            self.add_info_row("名称", item.view_name)
            item.set_widget_value(item, self)

        elif isinstance(item, Edge):
            self.title.setText("连线属性")
            start_name = item.start_socket.parentItem().name if item.start_socket else "N/A"
            self.add_info_row("来源", start_name)
            end_name = item.end_socket.parentItem().name if item.end_socket else "N/A"
            self.add_info_row("目标", end_name)
            if (not isinstance(item.start_socket.parentItem(), GlobalMemoryNode)) and (
            not isinstance(item.end_socket.parentItem(), GlobalMemoryNode)):
                self.edge_depth.show()
            self.size_spin.setValue(int(item.depth))

        elif isinstance(item, Socket):
            self.title.setText("端口属性")
            # 这里可以显示多行而不会被覆盖
            self.add_info_row("端口名", item.socket_data.name)
            self.add_info_row("端口类型", "输入" if item.is_input else "输出")
            self.add_info_row("数据大小", item.socket_data.volume or "无限制")

    def on_depth_changed(self, val):
        if isinstance(self.current_item, Edge):
            self.current_item.set_depth(val)

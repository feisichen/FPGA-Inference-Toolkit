# import sys
# import inspect
# from PyQt6.QtWidgets import (QDialog, QFormLayout, QSpinBox, QDialogButtonBox, QLineEdit, QApplication, QGraphicsView,
#                              QGraphicsScene,
#                              QGraphicsItem, QGraphicsPathItem, QVBoxLayout,
#                              QWidget, QPushButton, QInputDialog, QHBoxLayout,
#                              QMessageBox, QLabel, QSpinBox, QCheckBox, QFrame,
#                              QSplitter)
# from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal, QObject
# from PyQt6.QtGui import QPen, QPainterPath, QColor, QBrush, QPainter, QFont
#
# INPUT_SIZE = 100
#
#
#
#
# # --- 4. 动态属性面板 ---
# class PropertyPanel(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.current_item = None
#         self.init_ui()
#
#     def init_ui(self):
#         self.layout = QVBoxLayout(self)
#         self.layout.setContentsMargins(15, 20, 15, 20)
#         self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)
#
#         # 1. 标题
#         self.title = QLabel("属性面板")
#         self.title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
#         self.layout.addWidget(self.title)
#
#         # 2. 动态文本属性容器 (使用 QFormLayout)
#         self.info_widget = QWidget()
#         self.info_layout = QFormLayout(self.info_widget)
#         self.info_layout.setContentsMargins(0, 10, 0, 10)
#         self.layout.addWidget(self.info_widget)
#
#         # 3. 编辑控件 (ConvNode 专用)
#         self.p_widget = QWidget()
#         p_lay = QHBoxLayout(self.p_widget)
#         p_lay.setContentsMargins(0, 0, 0, 0)
#         p_lay.addWidget(QLabel("并行度:"))
#         self.p_spin = QSpinBox()
#         self.p_spin.setRange(1, 2147483647)
#         self.p_spin.valueChanged.connect(self.on_p_changed)
#         p_lay.addWidget(self.p_spin)
#         self.layout.addWidget(self.p_widget)
#         self.p_widget.hide()
#
#         self.bypass_check = QCheckBox("启用旁路 (Bypass)")
#         self.bypass_check.stateChanged.connect(self.on_bypass_changed)
#         self.layout.addWidget(self.bypass_check)
#         self.bypass_check.hide()
#
#         # 4. 编辑控件 (Edge 专用)
#         self.edge_depth = QWidget()
#         e_lay = QHBoxLayout(self.edge_depth)
#         e_lay.addWidget(QLabel("通道缓冲区大小 (Depth):"))
#         self.size_spin = QSpinBox()
#         self.size_spin.setRange(0, 2147483647)
#         self.size_spin.valueChanged.connect(self.on_depth_changed)
#         e_lay.addWidget(self.size_spin)
#         self.layout.addWidget(self.edge_depth)
#         self.edge_depth.hide()
#
#     def clear_info(self):
#         """清空动态生成的表单行"""
#         while self.info_layout.rowCount() > 0:
#             self.info_layout.removeRow(0)
#
#     def add_info_row(self, label, value):
#         """方便地添加一行显示信息"""
#         val_label = QLabel(str(value))
#         val_label.setStyleSheet("color: #666;")  # 调浅颜色区分标签和值
#         self.info_layout.addRow(f"{label}:", val_label)
#
#     def set_item(self, item):
#         self.current_item = item
#         self.clear_info()
#
#         # 隐藏所有编辑控件
#         self.p_widget.hide()
#         self.bypass_check.hide()
#         self.edge_depth.hide()
#
#         if isinstance(item, BaseNode):
#             self.title.setText("节点属性")
#             self.add_info_row("名称", item.name)
#             if isinstance(item, ConvNode):
#                 self.p_widget.show()
#                 self.bypass_check.show()
#                 self.p_spin.setValue(item.parallelism)
#                 self.bypass_check.setChecked(item.is_bypass)
#
#         elif isinstance(item, Edge):
#             self.title.setText("连线属性")
#             start_name = item.start_socket.parentItem().name if item.start_socket else "N/A"
#             self.add_info_row("来源", start_name)
#             end_name = item.end_socket.parentItem().name if item.end_socket else "N/A"
#             self.add_info_row("目标", end_name)
#             if (not isinstance(item.start_socket.parentItem(), GlobalMemoryNode)) and (not isinstance(item.end_socket.parentItem(), GlobalMemoryNode)):
#                 self.edge_depth.show()
#             self.size_spin.setValue(int(item.depth))
#
#         elif isinstance(item, Socket):
#             self.title.setText("端口属性")
#             # 这里可以显示多行而不会被覆盖
#             self.add_info_row("端口名", item.socket_data.name)
#             self.add_info_row("端口类型", "输入" if item.is_input else "输出")
#             self.add_info_row("数据大小", item.socket_data.volume or "无限制")
#
#     def on_depth_changed(self, val):
#         if isinstance(self.current_item, Edge):
#             self.current_item.depth = val
#             # 这里可以根据 size 动态改变连线的粗细（
#
#     def on_p_changed(self, val):
#         if isinstance(self.current_item, BaseNode):
#             self.current_item.parallelism = val
#             self.current_item.update()
#
#     def on_bypass_changed(self, state):
#         if isinstance(self.current_item, ConvNode):
#             self.current_item.set_bypass(state == 2)
#
#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = NodeEditor()
#     window.show()
#     sys.exit(app.exec())

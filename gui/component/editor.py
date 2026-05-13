import inspect
from datetime import datetime

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (QDialog, QFormLayout, QDialogButtonBox, QLineEdit, QGraphicsView,
                             QVBoxLayout,
                             QWidget, QPushButton, QHBoxLayout,
                             QSplitter, QFileDialog, QMessageBox)

from gui.component.node.base_node import BaseNode
from gui.component.node.global_memory_node import GlobalMemoryNode, add_storage
from gui.component.property_panel import PropertyPanel
from gui.component.scene import FlowScene
from gui.utils.common import NODE_SIGNALS, SocketData
from config.config import register_gui_cls, header_files

button_style = """
                    QPushButton {
                        /* 深邃蓝渐变 */
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                                    stop:0 #005A9E, stop:1 #003366);
                        color: #FFFFFF;
                        font-family: 'Segoe UI', 'Verdana', sans-serif;
                        font-size: 17px;
                        font-weight: 800;
                        /* 微调字间距，防止过宽溢出 */
                        letter-spacing: 0.5px; 

                        border: 1px solid #002D52;
                        border-radius: 30px;

                        /* 确保内部文字居中且不被裁切 */
                        padding: 0px 10px;
                    }

                    QPushButton:hover {
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                                    stop:0 #0078D4, stop:1 #004D8C);
                        border: 1px solid #0093FF;
                        /* 悬停时稍微发光 */
                        outline: none;
                    }

                    QPushButton:pressed {
                        background-color: #002244;
                        /* 按下时的小位移反馈 */
                        padding-left: 2px;
                        padding-top: 2px;
                    }
                """


class CanvasView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QColor("#1e1e1e"))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

    def wheelEvent(self, event):
        """缩放画布"""
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        if event.angleDelta().y() > 0:
            self.scale(zoom_in_factor, zoom_in_factor)
        else:
            self.scale(zoom_out_factor, zoom_out_factor)

    def mousePressEvent(self, event):
        """鼠标中键点击启用平移"""
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            # 修复：使用 event.position() 获取 QPointF，或直接传递 event.globalPosition()
            fake_event = event.__class__(
                event.type(),
                event.position(),  # PyQt6 推荐使用 position() 返回 QPointF
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                event.modifiers()
            )
            super().mousePressEvent(fake_event)
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """释放中键恢复模式"""
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        super().mouseReleaseEvent(event)


class NodeConfigDialog(QDialog):
    def __init__(self, node_name, param_name_list, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Configure {node_name}")
        self.setMinimumWidth(300)
        self.layout = QFormLayout(self)
        self.param_editors = {}  # 存储输入控件

        for param in param_name_list:
            # 使用 QLineEdit 代替 QSpinBox
            line_edit = QLineEdit()
            line_edit.setPlaceholderText(f"请输入 {param}")
            self.param_editors[param] = line_edit
            self.layout.addRow(param, line_edit)

        # 按钮
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addRow(self.buttons)  # 注意：FormLayout 建议用 addRow

    def get_values(self):
        # 返回文本值
        return {k: v.text() for k, v in self.param_editors.items()}


def _get_connected_cluster(start_node):
    """使用深度优先搜索(DFS)查找与特定 GlobalMemory 连通的所有节点"""
    cluster = set()
    stack = [start_node]
    while stack:
        node = stack.pop()
        if node not in cluster:
            cluster.add(node)
            # 遍历所有插槽，寻找通过边连接的邻居节点
            for socket in (node.in_sockets + node.out_sockets):
                for edge in socket.edges:
                    # 找到边另一端的 Socket
                    neighbor_socket = edge.end_socket if edge.start_socket == socket else edge.start_socket
                    if neighbor_socket:
                        neighbor_node = neighbor_socket.parentItem()
                        if neighbor_node not in cluster:
                            stack.append(neighbor_node)
    return cluster


class NodeEditor(QWidget):
    def __init__(self, input_size):
        super().__init__()
        self.node_count = 0
        self.resize(1300, 850)
        self.setWindowTitle("Compiler Hardware Editor")
        layout = QVBoxLayout(self)

        top_bar = QHBoxLayout()

        # 导出按钮
        export_layout = QHBoxLayout()
        export_layout.setContentsMargins(0, 15, 0, 20)
        export_layout.addStretch(1)
        self.export_btn = QPushButton("GENERATE HARDWARE CONFIG ")
        self.export_btn.setFixedSize(380, 60)
        self.export_btn.setStyleSheet(button_style)
        self.export_btn.clicked.connect(self.export_opencl_config)
        export_layout.addWidget(self.export_btn)
        export_layout.addStretch(1)
        layout.addLayout(export_layout)
        # 导出按钮

        # 保存/加载按钮
        file_btn_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                            stop:0 #2d89ef, stop:1 #1a5fb4);
                color: #FFFFFF;
                font-family: 'Segoe UI', 'Verdana', sans-serif;
                font-size: 13px;
                font-weight: 600;
                border: 1px solid #15539e;
                border-radius: 6px;
                padding: 8px 20px;
                min-width: 80px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                            stop:0 #4a9eff, stop:1 #2d7fd4);
                border: 1px solid #3d8ae6;
            }
            QPushButton:pressed {
                background-color: #15539e;
            }
        """
        file_layout = QHBoxLayout()
        self.save_btn = QPushButton("💾 保存")
        self.save_btn.setStyleSheet(file_btn_style)
        self.save_btn.clicked.connect(self.save_scene)
        self.load_btn = QPushButton("📂 加载")
        self.load_btn.setStyleSheet(file_btn_style)
        self.load_btn.clicked.connect(self.load_scene)
        file_layout.addWidget(self.save_btn)
        file_layout.addWidget(self.load_btn)
        file_layout.addStretch()
        layout.addLayout(file_layout)

        # 修改按钮连接逻辑，传递标签名以便弹窗显示
        for label, cls in register_gui_cls:
            if cls.is_creatable():
                b = QPushButton(f"+ {label}")
                # 捕获 func
                b.clicked.connect(lambda chk, c=cls: self.create_node(c, label))
                top_bar.addWidget(b)

        layout.addLayout(top_bar)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.scene = FlowScene()
        self.view = CanvasView(self.scene)
        self.view.setScene(self.scene)
        self.view.setBackgroundBrush(QColor("#1e1e1e"))
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)

        self.props = PropertyPanel()
        splitter.addWidget(self.view)
        splitter.addWidget(self.props)
        splitter.setStretchFactor(0, 1)
        layout.addWidget(splitter)

        # gm = GlobalMemoryNode()
        # add_storage("image", input_size)
        # gm.setPos(400, 500)
        # self.scene.addItem(gm)

        NODE_SIGNALS.selected.connect(self.props.set_item)

    def add_node(self, node):
        col_count = 5  # 每行放5个
        width_step = 120
        height_step = 80

        x = (self.node_count % col_count) * width_step
        y = (self.node_count // col_count) * height_step

        node.setPos(self.view.mapToScene(x + 50, y + 50))
        self.scene.addItem(node)
        self.node_count += 1

    def create_node(self, cls, label):
        sig = inspect.signature(cls.__init__)
        params = sig.parameters
        required_params = [p.name for p in params.values() if p.name != 'self']

        dialog = NodeConfigDialog(label, required_params, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            config = dialog.get_values()  # 这里的 v 全是 QLineEdit 的 str

            filtered_data = {}
            for k, v in config.items():
                if k in required_params:
                    # 获取该参数在类定义中的类型注解 (例如 int)
                    param_info = params[k]
                    target_type = param_info.annotation

                    try:
                        if target_type == int:
                            filtered_data[k] = int(v)
                        elif target_type == float:
                            filtered_data[k] = float(v)
                        elif target_type == bool:
                            # 处理布尔值：只有当字符串为 "true", "1", "yes" 时为 True
                            filtered_data[k] = v.lower() in ("true", "1", "yes", "y")
                        else:
                            filtered_data[k] = v  # 默认为 str
                    except (ValueError, TypeError):
                        print(f"参数类型转换失败: {k} 预期为 {target_type}")
                        filtered_data[k] = v  # 转换失败则回退到原始字符串

            node = cls(**filtered_data)
            node.setPos(self.view.mapToScene(200, 150))
            self.scene.addItem(node)

    def export_opencl_config(self):
        gm_nodes = [item for item in self.scene.items() if isinstance(item, GlobalMemoryNode)]
        all_clusters_code = []

        header_lines = ["/* Auto-Generated OpenCL Hardware Configuration */"]
        for header in header_files:
            header_lines.append(f'#include "{header}"')
        header_lines.append("#pragma OPENCL EXTENSION cl_intel_channels : enable\n")
        header_str = "\n".join(header_lines)

        for gm in gm_nodes:
            full_cluster = _get_connected_cluster(gm)
            nodes_to_gen = [n for n in full_cluster if isinstance(n, BaseNode)
                            and not isinstance(n, GlobalMemoryNode)]

            if not nodes_to_gen:
                continue

            cluster_output = [header_str,
                              f"// ========================================",
                              f"// --- Cluster: {gm.name} ---",
                              "// ========================================",
                              "// --- Channels ---"]

            for node in nodes_to_gen:
                base_name = f"ch_{node.name}"
                depth_val = 0
                if node.out_sockets and node.out_sockets[0].edges:
                    depth_val = node.out_sockets[0].edges[0].depth
                node.gen_channel_code(base_name, depth_val, cluster_output)

            cluster_output.append("\n// --- Node Logic ---")
            for node in nodes_to_gen:
                cluster_output.append(f"// Layer: {node.name}")
                node.gen_node_code(cluster_output)
                cluster_output.append("")

            all_clusters_code.append("\n".join(cluster_output))

        from manager.node_manager import NodeManager
        NodeManager.print_node()
        # 打印验证
        # for i, code in enumerate(all_clusters_code):
        #     print(f"\n--- CLUSTER {i} OUTPUT ---")
        #     print(code)

        for i, code in enumerate(all_clusters_code):
            filename = f"output/device/dev{i}.cl"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(code)
            print(f"已写入文件: {filename}")

    def save_scene(self):
        """保存场景到文件"""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "保存场景", "", "场景文件 (*.json);;所有文件 (*)"
        )
        if filepath:
            try:
                self.scene.save_to_file(filepath)
                QMessageBox.information(self, "成功", f"场景已保存到: {filepath}")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"保存失败: {e}")

    def load_scene(self):
        """从文件加载场景"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "加载场景", "", "场景文件 (*.json);;所有文件 (*)"
        )
        if filepath:
            self.scene.load_from_file(filepath)
            QMessageBox.information(self, "成功", f"场景已加载: {filepath}")
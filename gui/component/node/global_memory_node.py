from typing import Dict, Any, Set
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPen, QColor, QPainter
from PyQt6.QtWidgets import QPushButton, QInputDialog, QMessageBox

from gui.component.node.base_node import BaseNode
from gui.component.socket import Socket
from gui.utils.common import SocketData, FONT_TITLE

globalMemory_num = 0
stored_datas: Dict[str, SocketData] = {}
deletable_keys: Set[str] = set()  # 记录可通过 UI 删除的 stored_datas 键名
all_nodes = []
panel = None  # 属性面板引用


def add_storage(name, size):
    global stored_datas
    stored_datas[name] = SocketData(name, size)
    for n in all_nodes:
        n.add_storage_by_node(name, size)


def add_storage_with_delete(name, size):
    """添加可删除的存储数据"""
    global stored_datas, deletable_keys
    if name in stored_datas:
        return False
    stored_datas[name] = SocketData(name, size)
    deletable_keys.add(name)
    for n in all_nodes:
        n.add_storage_by_node(name, size)
    return True


def remove_storage(name):
    """删除存储数据（仅限可删除的）"""
    global stored_datas, deletable_keys
    if name not in deletable_keys:
        return False
    if name in stored_datas:
        del stored_datas[name]
        deletable_keys.discard(name)
        for n in all_nodes:
            n.remove_storage_by_node(name)
        return True
    return False


def set_stored_datas(data_dict):
    """设置存储数据字典（用于反序列化）"""
    global stored_datas, deletable_keys
    if isinstance(data_dict, dict):
        stored_datas = {k: SocketData(v["name"], v["volume"]) for k, v in data_dict.items()}
    else:
        stored_datas = {}
    deletable_keys = set()  # 反序列化时重置可删除列表


def set_deletable_keys(keys_set):
    """设置可删除键名集合（用于反序列化）"""
    global deletable_keys
    deletable_keys = set(keys_set) if keys_set else set()


def get_stored_datas():
    """获取存储数据字典（用于序列化）"""
    return {k: {"name": v.name, "volume": v.volume} for k, v in stored_datas.items()}


def get_deletable_keys():
    """获取可删除键名集合（用于序列化）"""
    return list(deletable_keys)


class GlobalMemoryNode(BaseNode):
    def __init__(self, name: str = None):
        global globalMemory_num
        global stored_datas
        global all_nodes
        # input_size = int(input_size)
        node_name = name if name else f"GLOBAL_MEMORY_{globalMemory_num}"

        super().__init__(node_name, node_name, 0, [], [])
        in_socket = Socket(self, SocketData("Write_In", 0), True, is_reentrant=True)
        # if input_size > 0:
        #     add_storage("image", input_size)
        self.in_sockets.append(in_socket)
        if self.scene():
            self.scene().addItem(in_socket)

        self.width, self.height = 400, 250
        self._update_size()

        for name, socket_data in stored_datas.items():
            self.add_storage_by_node(name, socket_data.volume)

        globalMemory_num += 1
        self._arrange_sockets()
        all_nodes.append(self)

    def _update_size(self):
        """根据 stored_datas 数量动态调整节点尺寸"""
        global stored_datas
        data_count = len(stored_datas)
        # 高度：基础高度 + 每条数据增加 28px，最小高度 250
        self.height = max(250, 90 + data_count * 28)
        # 宽度：基础宽度 + 每条数据增加 15px，最小宽度 400
        self.width = max(400, 280 + data_count * 80)

    def _arrange_sockets(self):
        all_sockets = self.in_sockets + self.out_sockets
        spacing = self.width / (len(all_sockets) + 1)
        for i, s in enumerate(all_sockets):
            s.setPos(spacing * (i + 1) - 12, self.height - 12)

    def add_storage_by_node(self, name, size):
        new_out = Socket(self, SocketData(name, size), False, is_reentrant=True)
        self.out_sockets.append(new_out)
        if self.scene(): self.scene().addItem(new_out)
        self._update_size()
        self._arrange_sockets()
        self.update()

    def remove_storage_by_node(self, name):
        """移除指定名称的存储数据对应的 Socket"""
        for s in self.out_sockets[:]:
            if s.socket_data.name == name:
                for e in s.edges[:]:
                    e.remove_self()
                self.out_sockets.remove(s)
                if self.scene():
                    self.scene().removeItem(s)
                break
        self._update_size()
        self._arrange_sockets()
        self.update()

    def remove_edge(self, edge):
        global stored_datas, deletable_keys
        if edge.option in stored_datas:
            del stored_datas[edge.option]
            deletable_keys.discard(edge.option)
            # 通知所有节点移除对应的 socket
            for n in all_nodes:
                n.remove_storage_by_node(edge.option)

    def paint(self, painter, option, widget):
        global stored_datas
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor("#1a1a1a"))
        painter.setPen(QPen(QColor("#0078d4"), 3))
        painter.drawRoundedRect(0, 0, self.width, self.height, 12, 12)
        painter.setFont(FONT_TITLE)
        painter.setPen(Qt.GlobalColor.white)
        painter.drawText(0, 10, self.width, 30, Qt.AlignmentFlag.AlignCenter, self.name)
        for i, (name, socket_data) in enumerate(stored_datas.items()):
            painter.setBrush(QColor("#0e639c"))
            painter.drawRect(40, 50 + i * 28, self.width - 80, 22)
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(50, 66 + i * 28, f"Data: {name}")

    def gen_channel_code(self, base_name, depth_val, output):
        pass

    def gen_node_code(self, output):
        pass

    @classmethod
    def init_widget(cls, p):
        global panel
        panel = p
        m_name = cls.__name__
        widget_map = panel.widget_map
        widget_map.setdefault(m_name, {})

        # 添加存储数据按钮
        widget_map[m_name]["add_btn"] = QPushButton("添加存储数据")
        add_btn = widget_map[m_name]["add_btn"]
        add_btn.clicked.connect(cls.on_add_storage)
        panel.layout.addWidget(add_btn)
        add_btn.hide()

        # 删除按钮容器（存储动态创建的删除按钮）
        widget_map[m_name]["delete_buttons"] = []

    @classmethod
    def hide_widget(cls, item):
        global panel
        m_name = cls.__name__
        widget_map = panel.widget_map
        if m_name in widget_map:
            widget_map[m_name]["add_btn"].hide()
            # 隐藏所有删除按钮
            for btn in widget_map[m_name]["delete_buttons"]:
                btn.hide()

    def set_widget_value(self, item, p):
        global panel, deletable_keys
        m_name = self.__class__.__name__
        widget_map = panel.widget_map

        # 显示添加按钮
        widget_map[m_name]["add_btn"].show()

        # 清除旧的删除按钮
        for btn in widget_map[m_name]["delete_buttons"]:
            btn.deleteLater()
        widget_map[m_name]["delete_buttons"] = []

        # 为可删除的存储数据创建删除按钮
        for name in deletable_keys:
            btn = QPushButton(f"删除: {name}")
            btn.clicked.connect(lambda checked, n=name: self._on_delete_storage(n))
            panel.layout.addWidget(btn)
            widget_map[m_name]["delete_buttons"].append(btn)

    @classmethod
    def on_add_storage(cls):
        global panel, stored_datas
        name, ok = QInputDialog.getText(panel, "添加存储数据", "输入名称:")
        if ok and name:
            if name in stored_datas:
                QMessageBox.warning(panel, "警告", f"名称 '{name}' 已存在!")
                return
            size, ok2 = QInputDialog.getInt(panel, "添加存储数据", "输入大小:", 0, 0, 2147483647)
            if ok2:
                add_storage_with_delete(name, size)
                # 刷新属性面板
                if isinstance(panel.current_item, GlobalMemoryNode):
                    panel.current_item.set_widget_value(panel.current_item, panel)

    def _on_delete_storage(self, name):
        global panel
        remove_storage(name)
        # 刷新属性面板
        if isinstance(panel.current_item, GlobalMemoryNode):
            panel.current_item.set_widget_value(panel.current_item, panel)

    # ========== 序列化 ==========
    def serialize(self) -> Dict[str, Any]:
        data = super().serialize()
        data["stored_datas"] = get_stored_datas()
        data["deletable_keys"] = get_deletable_keys()
        return data

    @classmethod
    def _deserialize_create(cls, data: Dict[str, Any]) -> 'GlobalMemoryNode':
        # 先恢复全局存储数据
        if "stored_datas" in data:
            set_stored_datas(data["stored_datas"])
        if "deletable_keys" in data:
            set_deletable_keys(data["deletable_keys"])
        return cls(name=data["name"])

    @classmethod
    def is_creatable(cls):
        return True

    @classmethod
    def is_deletable(cls):
        return True

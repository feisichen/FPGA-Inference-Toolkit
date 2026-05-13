import json
from typing import Dict, Any, List, Tuple
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QGraphicsScene,
                             QInputDialog, QMessageBox, QFileDialog)

from gui.component.edge import Edge
from gui.component.node.base_node import BaseNode
from gui.component.node.global_memory_node import GlobalMemoryNode, add_storage, set_stored_datas, get_stored_datas, \
    all_nodes
from gui.component.node.mem_read_node import MemReadNode
from gui.component.node.mem_write_node import MemWriteNode
from gui.component.socket import Socket


class FlowScene(QGraphicsScene):
    def __init__(self):
        super().__init__()
        self.start_socket = None
        self._node_id_counter = 0  # 用于生成唯一的节点ID

    def _get_node_id(self, node: BaseNode) -> int:
        """为节点分配一个唯一ID（基于对象id）"""
        return id(node)

    def _build_node_index(self) -> Tuple[Dict[int, BaseNode], List[Dict]]:
        """构建节点索引和序列化数据"""
        node_index = {}  # id -> node
        nodes_data = []

        for item in self.items():
            if isinstance(item, BaseNode):
                node_id = self._get_node_id(item)
                node_index[node_id] = item
                node_data = item.serialize()
                node_data["_id"] = node_id
                nodes_data.append(node_data)

        return node_index, nodes_data

    def _build_edge_data(self, node_index: Dict[int, BaseNode]) -> List[Dict]:
        """构建边的序列化数据"""
        edges_data = []

        for item in self.items():
            if isinstance(item, Edge):
                start_node = item.start_socket.parentItem()
                end_node = item.end_socket.parentItem()

                start_node_id = self._get_node_id(start_node)
                end_node_id = self._get_node_id(end_node)

                # 找到 socket 在父节点中的索引
                start_socket_idx = start_node.out_sockets.index(item.start_socket)
                end_socket_idx = end_node.in_sockets.index(item.end_socket)

                edge_data = item.serialize()
                edge_data.update({
                    "start_node_id": start_node_id,
                    "start_socket_idx": start_socket_idx,
                    "end_node_id": end_node_id,
                    "end_socket_idx": end_socket_idx,
                })
                edges_data.append(edge_data)

        return edges_data

    def serialize(self) -> Dict[str, Any]:
        """序列化整个场景"""
        node_index, nodes_data = self._build_node_index()
        edges_data = self._build_edge_data(node_index)

        return {
            "version": 1,
            "nodes": nodes_data,
            "edges": edges_data,
            "stored_datas": get_stored_datas(),
        }

    def deserialize(self, data: Dict[str, Any]):
        """从数据恢复场景"""
        # 清空当前场景
        self.clear_scene()

        # 恢复存储数据
        if "stored_datas" in data:
            set_stored_datas(data["stored_datas"])

        # 创建节点映射表
        node_map = {}  # old_id -> new_node
        node_name_map = {}  # name -> new_node
        # 恢复节点
        for node_data in data.get("nodes", []):
            node = BaseNode.deserialize(node_data)
            self.addItem(node)
            node_map[node_data["_id"]] = node
            node_name_map[node_data["name"]] = node

        from manager.node_manager import NodeManager
        NodeManager.bind_gui_node(node_name_map)

        # 恢复边
        for edge_data in data.get("edges", []):
            start_node = node_map.get(edge_data["start_node_id"])
            end_node = node_map.get(edge_data["end_node_id"])

            if start_node and end_node:
                start_socket = start_node.out_sockets[edge_data["start_socket_idx"]]
                end_socket = end_node.in_sockets[edge_data["end_socket_idx"]]

                edge = Edge.deserialize(edge_data, start_socket, end_socket)
                self.addItem(edge)

    def clear_scene(self):
        """清空场景（保留 GlobalMemoryNode）"""
        items_to_remove = []
        for item in self.items():
            if isinstance(item, Edge):
                items_to_remove.append(item)
            elif isinstance(item, BaseNode):
                items_to_remove.append(item)

        for item in items_to_remove:
            if isinstance(item, Edge):
                item.remove_self()
            else:
                self.removeItem(item)

        from manager.node_manager import NodeManager
        NodeManager.clear_gui_node()

    def save_to_file(self, filepath: str):
        """保存场景到文件"""
        data = self.serialize()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_from_file(self, filepath: str):
        """从文件加载场景"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.deserialize(data)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            for item in self.selectedItems():
                if isinstance(item, BaseNode) and item.is_deletable():
                    for s in item.in_sockets + item.out_sockets:
                        for e in s.edges[:]:
                            e.remove_self()
                    self.removeItem(item)
                elif isinstance(item, Edge):
                    item.remove_self()
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        items = self.items(event.scenePos())
        sock = next((i for i in items if isinstance(i, Socket)), None)
        if sock:
            if sock.is_occupied(): return
            self.start_socket = sock
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        items = self.items(event.scenePos())
        end_s = next((i for i in items if isinstance(i, Socket)), None)
        if end_s and self.start_socket and end_s != self.start_socket:
            if self.start_socket.is_input != end_s.is_input:
                start_socket = self.start_socket if not self.start_socket.is_input else end_s
                end_socket = end_s if end_s.is_input else self.start_socket
                end_p, start_p = end_socket.parentItem(), start_socket.parentItem()

                if not self.checkEdge(start_socket, end_socket):
                    pass
                else:
                    if isinstance(end_p, GlobalMemoryNode):
                        name, ok = QInputDialog.getText(None, "存储", "命名数据:")
                        add_storage(name if ok and name else "data", start_socket.socket_data.volume)
                        self.addItem(Edge(start_socket, end_socket, name if ok and name else "data"))
                    else:
                        self.addItem(Edge(start_socket, end_socket))
        self.start_socket = None
        super().mouseReleaseEvent(event)

    def checkEdge(self, start_socket, end_socket):
        end_p, start_p = end_socket.parentItem(), start_socket.parentItem()
        if (end_socket.is_occupied()) or (start_socket.is_occupied()):
            return False
        elif isinstance(start_p, GlobalMemoryNode) and not isinstance(end_p, MemReadNode):
            QMessageBox.warning(None, "错误", "GM数据需接MEM_READ")
            return False
        elif isinstance(end_p, GlobalMemoryNode) and not isinstance(start_p, MemWriteNode):
            QMessageBox.warning(None, "错误", "存入GM需接MEM_WRITE")
            return False
        if start_socket.socket_data.volume != 0 and end_socket.socket_data.volume != 0 and start_socket.socket_data.volume != end_socket.socket_data.volume:
            QMessageBox.warning(None, "错误", "输入输出数据维度不匹配")
            return False
        return True

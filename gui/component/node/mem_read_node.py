from typing import Dict, Any
from gui.component.node.base_node import BaseNode
from gui.utils.common import SocketData

mem_read_num = 0


class MemReadNode(BaseNode):
    def __init__(self, data_size: int, name: str = None):
        global mem_read_num
        self._data_size = data_size
        node_name = name if name else f"Mem_Read{mem_read_num}"
        super().__init__(node_name, node_name, 1, [SocketData("input", data_size)],
                         [SocketData("output", data_size)])
        mem_read_num += 1

    def gen_channel_code(self, base_name, depth_val, output):
        if depth_val <= 0:
            depth_val = self.default_depth
        output.append(f"channel int8_t {base_name} __attribute__((depth({depth_val})));")
        self.out_sockets[0].edges[0].base_name = base_name

    def gen_node_code(self, output):
        macro = "MEM_READ"
        data_size = self.out_sockets[0].socket_data.volume
        output_name = self.out_sockets[0].edges[0].get_total_name()
        output.append(f"{macro}({self.name}, {output_name}, {data_size})")

    @classmethod
    def init_widget(cls, panel):
        pass

    @classmethod
    def hide_widget(cls, item):
        pass

    def set_widget_value(self, item, panel):
        pass

    # ========== 序列化 ==========
    def serialize(self) -> Dict[str, Any]:
        data = super().serialize()
        data["data_size"] = self._data_size
        return data

    @classmethod
    def _deserialize_create(cls, data: Dict[str, Any]) -> 'MemReadNode':
        return cls(data_size=data["data_size"], name=data["name"])

    @classmethod
    def is_creatable(cls):
        return True

    @classmethod
    def is_deletable(cls):
        return True

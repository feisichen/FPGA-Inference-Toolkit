from typing import Dict, Any
from gui.component.node.base_node import BaseNode
from gui.utils.common import SocketData

mem_write_num = 0


class MemWriteNode(BaseNode):
    def __init__(self, data_size: int, name: str = None):
        global mem_write_num
        self._data_size = data_size
        node_name = name if name else f"Mem_Write{mem_write_num}"
        super().__init__(node_name, node_name, 1, [SocketData("input", data_size)],
                         [SocketData("output", data_size)])
        mem_write_num += 1

    def gen_channel_code(self, base_name, depth_val, output):
        if depth_val <= 0:
            depth_val = self.default_depth
        output.append(f"channel int8_t {base_name} __attribute__((depth({depth_val})));")
        self.out_sockets[0].edges[0].base_name = base_name

    def gen_node_code(self, output):
        macro = "MEM_WRITE"
        data_size = self.in_sockets[0].socket_data.volume
        input_name = self.in_sockets[0].edges[0].get_total_name()
        output.append(f"{macro}({self.name}, {input_name}, {data_size})")

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
    def _deserialize_create(cls, data: Dict[str, Any]) -> 'MemWriteNode':
        return cls(data_size=data["data_size"], name=data["name"])

    @classmethod
    def is_creatable(cls):
        return True

    @classmethod
    def is_deletable(cls):
        return True

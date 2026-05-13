from typing import Dict, Any
from gui.component.node.base_node import BaseNode
from gui.component.node.conv_node import ConvNode
from gui.utils.common import SocketData

fork_num = 0


class ForkNode(BaseNode):
    def __init__(self, data_size: int, name: str = None):
        global fork_num
        self.is_merge = None
        self._data_size = data_size
        node_name = name if name else f"Fork{fork_num}"
        super().__init__(node_name, node_name, 0, [SocketData("input", data_size)],
                         [SocketData("output1", data_size), SocketData("output2", data_size)])
        fork_num += 1

    def gen_channel_code(self, base_name, depth_val, output):
        if depth_val <= 0:
            depth_val = self.in_sockets[0].edges[0].start_socket.parentItem().default_depth
        next_node = self.out_sockets[0].edges[0].end_socket.parentItem()
        if isinstance(next_node, ConvNode) and next_node.is_csm and next_node.is_bypass:
            self.is_merge = True
            output.append(f"channel int8_t {base_name}[2] __attribute__((depth({depth_val})));")
            self.out_sockets[0].edges[0].base_name = base_name
            self.out_sockets[0].edges[0].extra_name = "[0]"
            self.out_sockets[1].edges[0].base_name = base_name
            self.out_sockets[1].edges[0].extra_name = "[1]"
        else:
            output.append(f"channel int8_t {base_name}_0 __attribute__((depth({depth_val})));")
            output.append(f"channel int8_t {base_name}_1 __attribute__((depth({depth_val})));")
            self.out_sockets[0].edges[0].base_name = base_name
            self.out_sockets[0].edges[0].extra_name = "_0"
            self.out_sockets[1].edges[0].base_name = base_name
            self.out_sockets[1].edges[0].extra_name = "_1"

    def gen_node_code(self, output):
        if self.is_merge:
            macro = "GEN_FORK_MERGE"
            data_size = self.out_sockets[0].socket_data.volume
            input_name = self.in_sockets[0].edges[0].get_total_name()
            output_name = self.out_sockets[0].edges[0].get_base_name()
            output.append(f"{macro}({self.name}, {input_name}, {output_name}, {data_size}, int8_t)")
        else:
            macro = "GEN_FORK_PIXEL"
            data_size = self.out_sockets[0].socket_data.volume
            input_name = self.in_sockets[0].edges[0].get_total_name()
            output_name_0 = self.out_sockets[0].edges[0].get_total_name()
            output_name_1 = self.out_sockets[1].edges[0].get_total_name()
            output.append(
                f"{macro}({self.name}, {input_name}, {output_name_0}, {output_name_1}, {data_size}, int8_t)")

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
    def _deserialize_create(cls, data: Dict[str, Any]) -> 'ForkNode':
        return cls(data_size=data["data_size"], name=data["name"])

    @classmethod
    def is_creatable(cls):
        return True

    @classmethod
    def is_deletable(cls):
        return True

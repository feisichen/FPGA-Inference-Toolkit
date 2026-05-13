from typing import Dict, Any
from gui.component.node.base_node import BaseNode
from gui.utils.common import SocketData


def is_deletable():
    return False


class AddNode(BaseNode):
    def __init__(self, name: str, view_name: str, data_size: int, default_depth: int = 1):
        self.data_size = data_size
        super().__init__(name, view_name, default_depth, [SocketData("input_y", data_size), SocketData("input_x", data_size)],
                         [SocketData("output", data_size)])

    def gen_channel_code(self, base_name, depth_val, output):
        if depth_val <= 0:
            depth_val = self.default_depth
        output.append(f"channel int8_t {base_name} __attribute__((depth({depth_val})));")
        self.out_sockets[0].edges[0].base_name = base_name

    def gen_node_code(self, output):
        macro = "GEN_ADD_NONGLOBAL"
        input_name_0 = self.in_sockets[0].edges[0].get_total_name()
        input_name_1 = self.in_sockets[1].edges[0].get_total_name()
        output_name = self.out_sockets[0].edges[0].get_total_name()
        output.append(f"{macro}({self.name}, {input_name_1}, {input_name_0}, {output_name}, int)")

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
        data["data_size"] = self.data_size
        return data

    @classmethod
    def _deserialize_create(cls, data: Dict[str, Any]) -> 'AddNode':
        return cls(
            name=data["name"],
            view_name=data.get("view_name", data["name"]),
            data_size=data["data_size"]
        )

    @classmethod
    def is_creatable(cls):
        return False

    @classmethod
    def is_deletable(cls):
        return False

from typing import Dict, Any
from gui.component.node.base_node import BaseNode
from gui.utils.common import SocketData


class ConcatNode(BaseNode):
    def __init__(self, name: str, view_name: str, input_x_size: int, input_y_size: int, default_depth: int = 1):
        self.input_x_size = input_x_size
        self.input_y_size = input_y_size
        super().__init__(name, view_name, default_depth, [SocketData("input_y", input_y_size), SocketData("input_x", input_x_size)],
                         [SocketData("output", input_x_size + input_y_size)])

    def gen_channel_code(self, base_name, depth_val, output):
        if depth_val <= 0:
            depth_val = self.default_depth
        output.append(f"channel int8_t {base_name} __attribute__((depth({depth_val})));")
        self.out_sockets[0].edges[0].base_name = base_name

    def gen_node_code(self, output):
        macro = "GEN_CONCAT_QUANT_NONGLOBAL"
        input_name_0 = self.in_sockets[0].edges[0].get_total_name()
        input_name_1 = self.in_sockets[1].edges[0].get_total_name()
        output_name = self.out_sockets[0].edges[0].get_total_name()
        output.append(
            f"{macro}({self.name}, {input_name_0}, {input_name_1}, {output_name},int8_t)")

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
        data.update({
            "input_x_size": self.input_x_size,
            "input_y_size": self.input_y_size,
        })
        return data

    @classmethod
    def _deserialize_create(cls, data: Dict[str, Any]) -> 'ConcatNode':
        return cls(
            name=data["name"],
            view_name=data.get("view_name", data["name"]),
            input_x_size=data["input_x_size"],
            input_y_size=data["input_y_size"],
        )

    @classmethod
    def is_creatable(cls):
        return False

    @classmethod
    def is_deletable(cls):
        return False

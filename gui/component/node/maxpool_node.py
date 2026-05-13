from typing import Dict, Any
from gui.component.node.base_node import BaseNode
from gui.utils.common import SocketData


class MaxPoolNode(BaseNode):
    def __init__(self, name: str, view_name: str, input_size: int, output_size: int, default_depth: int = 1):
        self.input_size = input_size
        self.output_size = output_size
        super().__init__(name, view_name, default_depth, [SocketData("input", input_size)], [SocketData("output", output_size)])

    def gen_channel_code(self, base_name, depth_val, output):
        if depth_val <= 0:
            depth_val = self.default_depth
        output.append(f"channel int8_t {base_name} __attribute__((depth({depth_val})));")
        self.out_sockets[0].edges[0].base_name = base_name

    def gen_node_code(self, output):
        macro = "GEN_MAXPOOL"
        input_name = self.in_sockets[0].edges[0].get_total_name()
        output_name = self.out_sockets[0].edges[0].get_total_name()
        output.append(f"{macro}({self.name}, {input_name}, {output_name}, int8_t)")

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
            "input_size": self.input_size,
            "output_size": self.output_size,
        })
        return data

    @classmethod
    def _deserialize_create(cls, data: Dict[str, Any]) -> 'MaxPoolNode':
        return cls(
            name=data["name"],
            view_name=data.get("view_name", data["name"]),
            input_size=data["input_size"],
            output_size=data["output_size"],
        )

    @classmethod
    def is_creatable(cls):
        return False

    @classmethod
    def is_deletable(cls):
        return False

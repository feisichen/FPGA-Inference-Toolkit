

from typing import Dict, List

import onnx
from onnx import numpy_helper

import dataflow.dataflow
from gui.component.node.concat_node import ConcatNode
from onnx_operator.base import BaseOperator, DeviceType
from utils import quant


class ConcatOperator(BaseOperator):
    m_nodes = {}

    def __init__(self):
        self.valid = False
        self.name = ""
        self.view_name = ""
        self.channels_x = None
        self.channels_y = None
        self.size = None
        self.num_add_x = None
        self.n_add_x = None
        self.num_add_y = None
        self.n_add_y = None
        self.zero_point_x = None
        self.zero_point_y = None
        self.zero_point_output = None

    @classmethod
    def create_node(cls, name, node: onnx.NodeProto, node_map: Dict[str, onnx.NodeProto],
                    dataflow_map: Dict[str, onnx.ValueInfoProto], tensor_map: Dict[str, onnx.TensorProto],
                    start_flow_map: Dict[str, List[onnx.NodeProto]], end_flow_map: Dict[str, List[onnx.NodeProto]]):
        if node.op_type != "Concat":
            return
        if node.name == "/backbone/Concat":
            print("123")
        try:
            output_name = node.output[0]
            output_node = end_flow_map[output_name][0]
            if output_node.op_type != "QuantizeLinear":
                return

            scale_output = numpy_helper.to_array(tensor_map[output_node.input[1]]).item()
            zp_output = numpy_helper.to_array(tensor_map[output_node.input[2]]).item()

            def get_in_params(inp_name):
                inp_node = start_flow_map[inp_name][0]
                if inp_node.op_type != "DequantizeLinear":
                    raise Exception(f"Input {inp_name} is not DequantizeLinear")
                scale = numpy_helper.to_array(tensor_map[inp_node.input[1]]).item()
                zp = numpy_helper.to_array(tensor_map[inp_node.input[2]]).item()

                df = dataflow_map[inp_name]
                dims = dataflow.dataflow.get_dims_from_dataflow(df)
                if df is None or len(dims) < 3:
                    raise Exception("Invalid dimensions")
                c = dims[1]
                s = dims[2] if len(dims) == 4 else None
                return scale, zp, c, s

            inputs = node.input
            if len(inputs) < 2:
                return

            scale_y, zp_y, channels_y, size_y = get_in_params(inputs[0])

            for i in range(1, len(inputs)):
                scale_x, zp_x, channels_x, size_x = get_in_params(inputs[i])

                size = size_y if size_y is not None else size_x
                if size_y is not None and size_x is not None and size_y != size_x:
                    raise Exception("Size mismatch between concatenated inputs")

                [num_add_y], [n_add_y] = quant.quant([scale_y / scale_output])
                [num_add_x], [n_add_x] = quant.quant([scale_x / scale_output])

                # 实例化子 Concat
                op = cls()
                op.valid = True
                # 多输入时使用 _1, _2 等后缀区分名称
                op.name = f"{name}_{i}" if len(inputs) > 2 else name
                op.view_name = f"{node.name}_{i}" if len(inputs) > 2 else node.name

                op.channels_x = channels_x
                op.channels_y = channels_y
                op.size = size
                op.num_add_x = num_add_x
                op.n_add_x = n_add_x
                op.num_add_y = num_add_y
                op.n_add_y = n_add_y
                op.zero_point_x = zp_x
                op.zero_point_y = zp_y
                op.zero_point_output = zp_output

                cls.m_nodes[op.name] = op

                scale_y = scale_output
                zp_y = zp_output
                channels_y += channels_x  # 通道数累加
                size_y = size

        except Exception as e:
            return

    @classmethod
    def create_gui_code(cls, window):
        for node in cls.m_nodes.values():
            if node.valid:
                window.add_node(ConcatNode(node.name, node.view_name, node.size * node.size * node.channels_x,
                                           node.size * node.size * node.channels_y, default_depth=node.channels_x + node.channels_y))

    @classmethod
    def print_all_nodes(cls, device_type: DeviceType = DeviceType.DEVICE):
        all_content = []
        for idx in sorted(list(cls.m_nodes.keys())):
            node_output = cls.m_nodes[idx].print_node()
            all_content.append(node_output)
        return "\n".join(all_content), ""

    def print_node(self, device_type: DeviceType = DeviceType.DEVICE):
        if not self.valid:
            return ""

        L = self.name

        # 确保所有数值都是整数，避免出现 16.0 这种格式
        lines = [
            f"// --- Concat {self.view_name} Configuration ---",
            f"#define FILTERS_X_CONCAT_{L} {int(self.channels_x)}",
            f"#define FILTERS_Y_CONCAT_{L} {int(self.channels_y)}",
            f"#define HEIGHT_CONCAT_{L} {int(self.size)}",
            f"#define WIDTH_CONCAT_{L} {int(self.size)}",
            f"#define NUM_CONCAT_X_{L} {int(self.num_add_x)}",
            f"#define N_CONCAT_X_{L} {int(self.n_add_x)}",
            f"#define NUM_CONCAT_Y_{L} {int(self.num_add_y)}",
            f"#define N_CONCAT_Y_{L} {int(self.n_add_y)}",
            f"#define ZERO_POINT_IX_CONCAT_{L} {int(self.zero_point_x)}",
            f"#define ZERO_POINT_IY_CONCAT_{L} {int(self.zero_point_y)}",
            f"#define ZERO_POINT_O_CONCAT_{L} {int(self.zero_point_output)}",
            ""  # 末尾空行
        ]

        return "\n".join(lines)

    @classmethod
    def bind_all_gui_node(cls, gui_node_map):
        for node in cls.m_nodes.values():
            node.bind_gui_node(gui_node_map)
        return

    def bind_gui_node(self, gui_node_map):
        if self.valid:
            gui_node_map[self.name].default_depth = self.channels_x + self.channels_y
        return

    @classmethod
    def clear_all_gui_node(cls):
        return

    def clear_gui_node(self):
        return
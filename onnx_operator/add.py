from typing import Dict, List

import onnx
from onnx import numpy_helper

import dataflow.dataflow
from gui.component.node.add_node import AddNode
from onnx_operator.base import BaseOperator, DeviceType
from utils import quant


class AddOperator(BaseOperator):
    m_nodes = {}

    def __init__(self, name, node: onnx.NodeProto, node_map: Dict[str, onnx.NodeProto],
                 dataflow_map: Dict[str, onnx.ValueInfoProto], tensor_map: Dict[str, onnx.TensorProto],
                 start_flow_map: Dict[str, List[onnx.NodeProto]], end_flow_map: Dict[str, List[onnx.NodeProto]]):
        self.valid = True

        self.name = name
        self.view_name = node.name

        self.channels = None
        self.size = None
        self.num_add_x = None
        self.n_add_x = None
        self.num_add_y = None
        self.n_add_y = None
        self.zero_point_x = None
        self.zero_point_y = None
        self.zero_point_output = None
        try:
            self.input_x_dataflow = dataflow_map[node.input[0]]
            self.input_y_dataflow = dataflow_map[node.input[1]]
            self.output_dataflow = dataflow_map[node.output[0]]

            self._extract_input_output_shapes()

            input_x_node = start_flow_map[node.input[0]][0]
            input_y_node = start_flow_map[node.input[1]][0]
            output_node = end_flow_map[node.output[0]][0]

            if input_x_node.op_type != "DequantizeLinear" or input_y_node.op_type != "DequantizeLinear" or output_node.op_type != "QuantizeLinear":
                self.valid = False
                return

            scale_add_x = numpy_helper.to_array(tensor_map[input_x_node.input[1]]).item()
            scale_add_y = numpy_helper.to_array(tensor_map[input_y_node.input[1]]).item()
            scale_output = numpy_helper.to_array(tensor_map[output_node.input[1]]).item()

            [self.num_add_x], [self.n_add_x] = quant.quant([scale_add_x / scale_output])
            [self.num_add_y], [self.n_add_y] = quant.quant([scale_add_y / scale_output])
            self.zero_point_x = numpy_helper.to_array(tensor_map[input_x_node.input[2]]).item()
            self.zero_point_y = numpy_helper.to_array(tensor_map[input_y_node.input[2]]).item()
            self.zero_point_output = numpy_helper.to_array(tensor_map[output_node.input[2]]).item()
        except Exception as e:
            self.valid = False
            return

    @classmethod
    def create_node(cls, name, node: onnx.NodeProto, node_map: Dict[str, onnx.NodeProto],
                    dataflow_map: Dict[str, onnx.ValueInfoProto], tensor_map: Dict[str, onnx.TensorProto],
                    start_flow_map: Dict[str, List[onnx.NodeProto]], end_flow_map: Dict[str, List[onnx.NodeProto]]):
        if node.op_type == "Add":
            cls.m_nodes[name] = AddOperator(
                name, node, node_map, dataflow_map, tensor_map,
                start_flow_map, end_flow_map
            )

    @classmethod
    def create_gui_code(cls, window):
        for node in cls.m_nodes.values():
            if node.valid:
                window.add_node(AddNode(node.name, node.view_name, node.size * node.size * node.channels, default_depth=node.channels))

    @classmethod
    def print_all_nodes(cls, device_type: DeviceType = DeviceType.DEVICE):
        all_content = []
        for idx in sorted(list(cls.m_nodes.keys())):
            node_output = cls.m_nodes[idx].print_node(device_type)
            all_content.append(node_output)
        return "\n".join(all_content), ""

    def print_node(self, device_type: DeviceType = DeviceType.DEVICE):
        if not self.valid:
            return ""

        L = self.name
        # 确保所有数值都是整数，避免出现 16.0 这种格式
        lines = [
            f"// --- Add {self.view_name} Configuration ---",
            f"#define FILTERS_ADD_{L} {int(self.channels)}",
            f"#define HEIGHT_ADD_{L} {int(self.size)}",
            f"#define WIDTH_ADD_{L} {int(self.size)}",
            f"#define NUM_ADD_X_{L} {int(self.num_add_x)}",
            f"#define N_ADD_X_{L} {int(self.n_add_x)}",
            f"#define NUM_ADD_Y_{L} {int(self.num_add_y)}",
            f"#define N_ADD_Y_{L} {int(self.n_add_y)}",
            f"#define ZERO_POINT_IX_ADD_{L} {int(self.zero_point_x)}",
            f"#define ZERO_POINT_IY_ADD_{L} {int(self.zero_point_y)}",
            f"#define ZERO_POINT_O_ADD_{L} {int(self.zero_point_output)}",
            ""  # 末尾空行
        ]

        return "\n".join(lines)

    def _extract_input_output_shapes(self):
        input_dims = dataflow.dataflow.get_dims_from_dataflow(self.input_x_dataflow)
        if self.input_x_dataflow is not None and len(input_dims) >= 3:
            self.channels = input_dims[1]  # 通道数
            if len(input_dims) == 4:  # 2D卷积
                self.size = input_dims[2]
            else:
                raise Exception(f"Invalid input dims: {input_dims}")
        else:
            raise Exception(f"Invalid input dims: {input_dims}")

    @classmethod
    def bind_all_gui_node(cls, gui_node_map):
        for node in cls.m_nodes.values():
            node.bind_gui_node(gui_node_map)
        return

    def bind_gui_node(self, gui_node_map):
        if self.valid:
            gui_node_map[self.name].default_depth = self.channels
        return

    @classmethod
    def clear_all_gui_node(cls):
        return

    def clear_gui_node(self):
        return

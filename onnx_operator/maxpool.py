from typing import Dict, List

import onnx
from onnx import numpy_helper

import dataflow.dataflow
from gui.component.node.maxpool_node import MaxPoolNode
from onnx_operator.base import BaseOperator, DeviceType


class MaxPoolOperator(BaseOperator):
    m_nodes = {}

    def __init__(self, name, node: onnx.NodeProto, node_map: Dict[str, onnx.NodeProto],
                 dataflow_map: Dict[str, onnx.ValueInfoProto], tensor_map: Dict[str, onnx.TensorProto],
                 start_flow_map: Dict[str, List[onnx.NodeProto]], end_flow_map: Dict[str, List[onnx.NodeProto]]):
        self.valid = True

        self.name = name
        self.view_name = node.name

        self.channels = None
        self.input_size = None
        self.kernel_size = None
        self.pad = None
        self.stride = None
        self.output_size = None
        self.zero_point = None
        self.gui_node = None

        try:
            self.input_dataflow = dataflow_map[node.input[0]]
            self.output_dataflow = dataflow_map[node.output[0]]

            # 解析属性
            for attr in node.attribute:
                if attr.name == "kernel_shape":
                    self.kernel_size = list(attr.ints)[0]
                elif attr.name == "strides":
                    self.stride = list(attr.ints)[0]
                elif attr.name == "pads":
                    self.pad = list(attr.ints)[0]

            self._extract_input_output_shapes()

            # 获取零点 (MaxPool通常输入输出Scale/ZP一致，取输入即可)
            input_node = start_flow_map[node.input[0]][0]
            if input_node.op_type != "DequantizeLinear":
                valid = False
                return

            self.zero_point = numpy_helper.to_array(tensor_map[input_node.input[2]]).item()
        except Exception as e:
            self.valid = False
            return

    @classmethod
    def create_node(cls, name, node: onnx.NodeProto, node_map: Dict[str, onnx.NodeProto],
                    dataflow_map: Dict[str, onnx.ValueInfoProto], tensor_map: Dict[str, onnx.TensorProto],
                    start_flow_map: Dict[str, List[onnx.NodeProto]], end_flow_map: Dict[str, List[onnx.NodeProto]]):
        if node.op_type == "MaxPool":
            cls.m_nodes[name] = MaxPoolOperator(
                name, node, node_map, dataflow_map, tensor_map,
                start_flow_map, end_flow_map
            )

    @classmethod
    def create_gui_code(cls, window):
        pass
        # for node in cls.m_nodes.values():
        #     if node.valid:
        #         maxpool_node = MaxPoolNode(node.name, node.view_name, node.input_size * node.input_size * node.channels,
        #                                     node.output_size * node.output_size * node.channels, default_depth=node.channels)
        #         node.gui_node = maxpool_node
        #         window.add_node(maxpool_node)

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

        lines = [
            f"// --- MaxPool {self.view_name} Configuration ---",
            f"#define CHANNELS_MAXPOOL_{L} {int(self.channels)}",
            f"#define HEIGHT_MAXPOOL_{L} {int(self.input_size)}",
            f"#define WIDTH_MAXPOOL_{L} {int(self.input_size)}",
            f"#define KERNEL_SIZE_MAXPOOL_{L} {int(self.kernel_size)}",
            f"#define PAD_MAXPOOL_{L} {int(self.pad)}",
            f"#define STRIDE_MAXPOOL_{L} {int(self.stride)}",
            f"#define ZERO_POINT_MAXPOOL_{L} {int(self.zero_point)}",
            ""  # 末尾空行
        ]

        return "\n".join(lines)

    def _extract_input_output_shapes(self):
        input_dims = dataflow.dataflow.get_dims_from_dataflow(self.input_dataflow)
        if input_dims is not None and len(input_dims) >= 3:
            self.channels = input_dims[1]
            self.input_size = input_dims[2]
        else:
            raise Exception(f"Invalid")

        output_dims = dataflow.dataflow.get_dims_from_dataflow(self.output_dataflow)
        if output_dims is not None and len(output_dims) >= 3:
            self.output_size = output_dims[2]
        else:
            raise Exception(f"Invalid")

        # 如果获取失败，手动计算
        if self.input_size and self.kernel_size and self.stride is not None and self.output_size is None:
            self.output_size = (self.input_size + 2 * self.pad - self.kernel_size) // self.stride + 1

    @classmethod
    def bind_all_gui_node(cls, gui_node_map):
        # pass
        for node in cls.m_nodes.values():
            node.bind_gui_node(gui_node_map)

    def bind_gui_node(self, gui_node_map):
        # gui_node_map[self.name].default_depth = self.channels_x + self.channels_y
        self.gui_node = gui_node_map[self.name]

    @classmethod
    def clear_all_gui_node(cls):
        # pass
        for node in cls.m_nodes.values():
            node.clear_gui_node()

    def clear_gui_node(self):
        # pass
        self.gui_node = None

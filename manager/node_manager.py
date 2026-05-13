import sys
from typing import Dict, List

import onnx
from PyQt6.QtWidgets import (QApplication)

from gui.component.editor import NodeEditor
from gui.component.node.base_node import BaseNode
from utils.const import *
from config.config import register_onnx_cls
from onnx_operator.base import DeviceType


def to_cpp_name(name: str) -> str:
    clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)

    if clean_name[0].isdigit():
        clean_name = "_" + clean_name

    clean_name = re.sub(r'_+', '_', clean_name)

    return clean_name.strip('_')


class NodeManager:
    def __init__(self, input_size, node_map: Dict[str, onnx.NodeProto],
                 dataflow_map: Dict[str, onnx.ValueInfoProto], tensor_map: Dict[str, onnx.TensorProto],
                 start_flow_map: Dict[str, List[onnx.NodeProto]], end_flow_map: Dict[str, List[onnx.NodeProto]]):
        self.start_flow_map = start_flow_map
        self.end_flow_map = end_flow_map  # key是张量名，value是生成该张量的节点

        # 按 op_type 分组，同类型按 name 排序
        from collections import defaultdict
        nodes_by_type = defaultdict(list)
        for node in node_map.values():
            nodes_by_type[node.op_type].append(node)
        
        # 对每组的节点按 name 排序
        for op_type in nodes_by_type:
            nodes_by_type[op_type].sort(key=lambda n: n.name)
        
        # 生成唯一名称：op_type + 序号
        for op_type, nodes in nodes_by_type.items():
            for idx, node in enumerate(nodes):
                unique_name = f"{op_type}_{idx}"
                for cls in register_onnx_cls:
                    cls.create_node(unique_name, node, node_map, dataflow_map, tensor_map, start_flow_map, end_flow_map)

        self.app = QApplication(sys.argv)
        self.window = NodeEditor(input_size)

        for cls in register_onnx_cls:
            cls.create_gui_code(self.window)

    @classmethod
    def bind_gui_node(cls, gui_node_map: Dict[str, BaseNode]):
        for node_cls in register_onnx_cls:
            node_cls.bind_all_gui_node(gui_node_map)

    @classmethod
    def clear_gui_node(cls):
        for node_cls in register_onnx_cls:
            node_cls.clear_all_gui_node()

    def show(self):
        self.window.show()
        sys.exit(self.app.exec())

    @classmethod
    def print_node(cls):
        info_lines = []
        weight_lines = []
        for node_cls in register_onnx_cls:
            info, weight = node_cls.print_all_nodes(device_type=DeviceType.DEVICE)
            info_lines.append(info)
            weight_lines.append(weight)

        # 写入 layer_info.h 并添加头文件保护
        with open('output/device/layer_info.h', 'w', encoding='utf-8') as f:
            f.write('#ifndef LAYER_INFO_H_\n')
            f.write('#define LAYER_INFO_H_\n\n')
            for line in info_lines:
                f.write(line + '\n')
            f.write('\n#endif // LAYER_INFO_H_')

        # 写入 weights.h 并添加头文件保护和 #include "type.h"
        with open('output/device/weights.h', 'w', encoding='utf-8') as f:
            f.write('#ifndef WEIGHTS_H_\n')
            f.write('#define WEIGHTS_H_\n\n')
            f.write('#include "type.h"\n\n')
            for line in weight_lines:
                f.write(line + '\n')
            f.write('\n#endif // WEIGHTS_H_')

        info_lines = []
        weight_lines = []
        for node_cls in register_onnx_cls:
            info, weight = node_cls.print_all_nodes(device_type=DeviceType.HOST)
            info_lines.append(info)
            weight_lines.append(weight)

        # 写入 layer_info.h 并添加头文件保护
        with open('output/host/layer_info.h', 'w', encoding='utf-8') as f:
            f.write('#ifndef LAYER_INFO_H_\n')
            f.write('#define LAYER_INFO_H_\n\n')
            f.write('#include <vector>\n\n')
            for line in info_lines:
                f.write(line + '\n')
            f.write('\n#endif // LAYER_INFO_H_')

        # 写入 weights.h 并添加头文件保护和 #include "type.h"
        with open('output/host/weights.h', 'w', encoding='utf-8') as f:
            f.write('#ifndef WEIGHTS_H_\n')
            f.write('#define WEIGHTS_H_\n\n')
            f.write('#include <stdint.h>\n')
            f.write('#include <vector>\n')
            f.write('#include "AlignedAllocator.h"\n\n')
            for line in weight_lines:
                f.write(line + '\n')
            f.write('\n#endif // WEIGHTS_H_')
from collections import defaultdict

import numpy as np
import onnx
from onnx import shape_inference

from manager.node_manager import NodeManager


def main():
    tensor_map = {}
    node_map = {}
    dataflow_map = {}
    start_flow_map = defaultdict(list)
    end_flow_map = defaultdict(list)

    onnx_path = 'models/yolox_nano_ptq.onnx'
    raw_model = onnx.load(onnx_path)
    model = shape_inference.infer_shapes(raw_model)

    # file_layer_info = open("output/layer_info.h", 'w')

    for initializer in model.graph.initializer:
        tensor_map[initializer.name] = initializer
    for _, node in enumerate(model.graph.node):
        node_map[node.name] = node
        for output_name in node.output:
            start_flow_map[output_name].append(node)
        for input_name in node.input:
            end_flow_map[input_name].append(node)
    for value_info in model.graph.value_info:
        dataflow_map[value_info.name] = value_info

    node_manager = NodeManager(1 * 12 * 208 * 208, node_map, dataflow_map, tensor_map, start_flow_map, end_flow_map)
    node_manager.print_node()
    node_manager.show()


if __name__ == "__main__":
    main()

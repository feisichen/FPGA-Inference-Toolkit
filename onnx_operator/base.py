from abc import abstractmethod
from enum import Enum
from typing import Dict, List

import onnx
from gui.component.node.base_node import BaseNode


class DeviceType(Enum):
    HOST = "HOST"
    DEVICE = "DEVICE"


class BaseOperator:
    @abstractmethod
    def __init__(self, node_map: Dict[str, onnx.NodeProto],
                 dataflow_map: Dict[str, onnx.ValueInfoProto], tensor_map: Dict[str, onnx.TensorProto],
                 start_flow_map: Dict[str, onnx.NodeProto], end_flow_map: Dict[str, onnx.NodeProto]):
        pass

    @abstractmethod
    def print_node(self, device_type: DeviceType = DeviceType.DEVICE):
        pass

    @classmethod
    @abstractmethod
    def print_all_nodes(cls, device_type: DeviceType = DeviceType.DEVICE):  # 修改为 OperatorType 枚举类型
        pass

    @classmethod
    @abstractmethod
    def create_gui_code(cls, window):
        pass

    @classmethod
    @abstractmethod
    def bind_all_gui_node(cls, gui_node_map: Dict[str, BaseNode]):
        pass

    @abstractmethod
    def bind_gui_node(self, gui_node_map: Dict[str, BaseNode]):
        pass

    @classmethod
    @abstractmethod
    def clear_all_gui_node(cls):
        pass

    @abstractmethod
    def clear_gui_node(self):
        pass

    @classmethod
    @abstractmethod
    def create_node(cls, view_name, node: onnx.NodeProto, node_map: Dict[str, onnx.NodeProto],
                    dataflow_map: Dict[str, onnx.ValueInfoProto], tensor_map: Dict[str, onnx.TensorProto],
                    start_flow_map: Dict[str, List[onnx.NodeProto]], end_flow_map: Dict[str, List[onnx.NodeProto]]):
        pass
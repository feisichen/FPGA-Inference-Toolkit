from typing import List, Tuple, Type

from gui.component.node.add_node import AddNode
from gui.component.node.base_node import BaseNode
from gui.component.node.concat_node import ConcatNode
from gui.component.node.conv_node import ConvNode
from gui.component.node.fork_node import ForkNode
from gui.component.node.global_memory_node import GlobalMemoryNode
from gui.component.node.maxpool_node import MaxPoolNode
from gui.component.node.mem_read_node import MemReadNode
from gui.component.node.mem_write_node import MemWriteNode
from onnx_operator.add import AddOperator
from onnx_operator.base import BaseOperator
from onnx_operator.concat import ConcatOperator
from onnx_operator.conv import ConvOperator
from onnx_operator.maxpool import MaxPoolOperator

register_onnx_cls: List[Type[BaseOperator]] = [
    ConvOperator, ConcatOperator, AddOperator
]
register_gui_cls: List[Tuple[str, Type[BaseNode]]] = [
    ("GlobalMemory", GlobalMemoryNode),
    ("Conv", ConvNode),
    ("Add", AddNode),
    ("Concat", ConcatNode),
    # ("MaxPool", MaxPoolNode),
    ("Fork", ForkNode),
    ("Read", MemReadNode),
    ("Write", MemWriteNode)
]
header_files = [
    "type.h",
    "common.h",
    "concat.h",
    "mem_read.h",
    "mem_write.h",
    "add.h",
    "weights.h",
    "csm.h",
    "fork.h"
]
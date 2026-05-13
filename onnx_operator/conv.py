import onnx
from typing import Dict, List
import dataflow.dataflow
from onnx import numpy_helper
import numpy as np
import math

from gui.component.node.base_node import BaseNode
from gui.component.node.conv_node import ConvNode
from onnx_operator.base import BaseOperator, DeviceType
from utils import quant

temptemp = 0
class ConvOperator(BaseOperator):
    conv_nodes = {}
    csm_nodes = {}

    def __init__(self, name, node: onnx.NodeProto, node_map: Dict[str, onnx.NodeProto],
                 dataflow_map: Dict[str, onnx.ValueInfoProto], tensor_map: Dict[str, onnx.TensorProto],
                 start_flow_map: Dict[str, List[onnx.NodeProto]], end_flow_map: Dict[str, List[onnx.NodeProto]],
                 is_csm=False):
        self.valid = True

        self.is_csm = is_csm
        self.name = name
        self.view_name = node.name

        self.sigmoid_table = []
        self.sigmoid_lower_bound = None
        self.sigmoid_upper_bound = None
        self.int_threshold = None
        self.input_channels = None
        self.input_size = None
        self.kernel_size = None
        self.pad = None
        self.stride = None
        self.output_size = None
        self.output_channels = None

        self.zero_point_input = None  # 输入零点
        self.zero_point_output = None  # 输出零点
        self.zero_point_input_sig = None  # sigmoid输入零点
        self.zero_point_output_mul = None  # 乘法的输出零点

        # 量化缩放参数
        self.num_conv = None  # 卷积的分子
        self.n_conv = None  # 卷积的2的幂次分母 (2^N)
        self.num_mul = None  # 乘法的分子
        self.n_mul = None  # 乘法的2的幂次分母 (2^N)

        # 偏置和硬件参数
        self.bias = None
        self.weight = None
        self.gui_node = None
        self.group = 1

        try:
            self.input_dataflow = dataflow_map[node.input[0]]
            self.weight_dataflow = dataflow_map[node.input[1]]
            self.bias_dataflow = dataflow_map[node.input[2]]
            self.output_dataflow = dataflow_map[node.output[0]]

            for attr in node.attribute:
                if attr.name == "kernel_shape":
                    self.kernel_size = list(attr.ints)[0]
                elif attr.name == "strides":
                    self.stride = list(attr.ints)[0]
                elif attr.name == "pads":
                    self.pad = list(attr.ints)[0]
                elif attr.name == "group":
                    self.group = attr.i

            self._extract_input_output_shapes()

            input_node = start_flow_map[node.input[0]][0]
            weight_node = start_flow_map[node.input[1]][0]
            bias_node = start_flow_map[node.input[2]][0]
            output_node = end_flow_map[node.output[0]][0]

            if input_node.op_type != "DequantizeLinear" or output_node.op_type != "QuantizeLinear":
                self.valid = False
                return

            self.bias = numpy_helper.to_array(tensor_map[bias_node.input[0]]).flatten()
            self.weight = numpy_helper.to_array(tensor_map[weight_node.input[0]]).flatten()
            self.zero_point_input = numpy_helper.to_array(tensor_map[input_node.input[2]]).item()
            self.zero_point_output = numpy_helper.to_array(tensor_map[output_node.input[2]]).item()
            scale_input = numpy_helper.to_array(tensor_map[input_node.input[1]]).item()
            scale_output = numpy_helper.to_array(tensor_map[output_node.input[1]]).item()
            weight_scale_tensor = numpy_helper.to_array(tensor_map[weight_node.input[1]]).flatten()
            bias_scale_tensor = numpy_helper.to_array(tensor_map[bias_node.input[1]]).flatten()

            raw_ratios = bias_scale_tensor / scale_output

            num_list, n_list = quant.quant(raw_ratios, 6000)

            raw_bias = numpy_helper.to_array(tensor_map[bias_node.input[0]]).flatten()
            raw_weight = numpy_helper.to_array(tensor_map[weight_node.input[0]]).flatten()

            self._pad_parameters(raw_weight, raw_bias, np.array(num_list), np.array(n_list))

            global temptemp
            if self.group > 1:
                temptemp += 2 * (self.kernel_size * self.kernel_size) * self.output_size * self.output_size * self.output_channels
            else:
                temptemp += 2 * (self.kernel_size * self.kernel_size * self.input_channels) * self.output_size * self.output_size * self.output_channels

            print(temptemp)


            if self.is_csm:
                next_after_conv = self._get_real_next_nodes(node, end_flow_map)
                self.sigmoid_node = next((n for n in next_after_conv if n.op_type == "Sigmoid"), None)
                self.mul_node = next((n for n in next_after_conv if n.op_type == "Mul"), None)
                if not self.mul_node:
                    next_after_sigmoid = self._get_real_next_nodes(self.sigmoid_node, end_flow_map)
                    self.mul_node = next((n for n in next_after_sigmoid if n.op_type == "Mul"), None)
                sig_input_node = start_flow_map[self.sigmoid_node.input[0]][0]
                sig_output_node = end_flow_map[self.sigmoid_node.output[0]][0]
                mul_output_node = end_flow_map[self.mul_node.output[0]][0]
                self.zero_point_input_sig = numpy_helper.to_array(tensor_map[sig_output_node.input[2]]).item()
                self.zero_point_output_mul = numpy_helper.to_array(tensor_map[mul_output_node.input[2]]).item()
                sig_input_zp = numpy_helper.to_array(tensor_map[sig_input_node.input[2]]).item()
                sig_input_scale = numpy_helper.to_array(tensor_map[sig_input_node.input[1]]).item()
                sig_output_scale = numpy_helper.to_array(tensor_map[sig_output_node.input[1]]).item()
                mul_output_scale = numpy_helper.to_array(tensor_map[mul_output_node.input[1]]).item()
                mul_raw_ratios = [sig_input_scale * sig_output_scale / mul_output_scale]
                [self.num_mul], [self.n_mul] = quant.quant(mul_raw_ratios, 10000)
                self.generate_sigmoid_table(sig_input_scale, sig_input_zp)

        except Exception as e:
            self.valid = False
            return

    def generate_sigmoid_table(self, scale_in, zp_in):
        full_range = range(-128, 128)

        raw_table = []
        for x in full_range:
            f_in = x * scale_in
            f_out = 1 / (1 + np.exp(-f_in))
            q_out = int(round(f_out * 255 - 128))
            q_out = max(-128, min(127, q_out))
            raw_table.append(q_out)

        left_idx = 0
        while left_idx < 255 and raw_table[left_idx] == -128:
            left_idx += 1

        right_idx = 255
        while right_idx > 0 and raw_table[right_idx] == 127:
            right_idx -= 1

        start_idx = max(0, left_idx)
        end_idx = min(255, right_idx)

        self.sigmoid_table = raw_table[start_idx: end_idx + 1]
        self.sigmoid_lower_bound = full_range[start_idx]
        self.sigmoid_upper_bound = full_range[end_idx]

    @classmethod
    def create_node(cls, name, node: onnx.NodeProto, node_map: Dict[str, onnx.NodeProto],
                    dataflow_map: Dict[str, onnx.ValueInfoProto], tensor_map: Dict[str, onnx.TensorProto],
                    start_flow_map: Dict[str, List[onnx.NodeProto]], end_flow_map: Dict[str, List[onnx.NodeProto]]):
        if node.op_type == "Conv":
            is_csm = cls._check_csm_structure(node, end_flow_map)
            if is_csm:
                cls.csm_nodes[name] = ConvOperator(
                    name, node, node_map, dataflow_map, tensor_map,
                    start_flow_map, end_flow_map, is_csm=True
                )
            else:
                cls.conv_nodes[name] = ConvOperator(
                    name, node, node_map, dataflow_map, tensor_map,
                    start_flow_map, end_flow_map
                )

    @classmethod
    def create_gui_code(cls, window):
        for node in cls.conv_nodes.values():
            if node.valid:
                conv_node = ConvNode(node.name, node.view_name, node.input_size * node.input_size * node.input_channels,
                                     node.output_size * node.output_size * node.output_channels, False, node.group > 1,
                                     default_depth=node.output_channels)
                node.gui_node = conv_node
                window.add_node(conv_node)
        for node in cls.csm_nodes.values():
            if node.valid:
                conv_node = ConvNode(node.name, node.view_name, node.input_size * node.input_size * node.input_channels,
                                     node.output_size * node.output_size * node.output_channels, True, node.group > 1,
                                     default_depth=node.output_channels)
                node.gui_node = conv_node
                window.add_node(conv_node)

    @classmethod
    def print_all_nodes(cls, device_type: DeviceType = DeviceType.DEVICE):
        # return cls.print_conv_node() + cls.print_csm_node()
        return cls.print_conv_node(device_type) + cls.print_csm_node(device_type), cls.print_conv_weight(
            device_type) + cls.print_csm_weight(device_type)

    def print_node(self, device_type: DeviceType = DeviceType.DEVICE):
        if not self.valid:
            return ""

        L = self.name

        def fmt_list(lst):
            return "{ " + ", ".join(map(str, lst)) + " }"

        temp = self.gui_node.parallelism * 8 // 8

        if temp < 1:
            temp = 1

        raw_banks = self.kernel_size * self.kernel_size * temp

        if raw_banks <= 1:
            num_banks = 1
        else:
            num_banks = 1 << (raw_banks - 1).bit_length()

        lines = [
            f"// --- Layer {self.view_name} Configuration ---",
            f"#define INPUT_CHANNELS_LAYER_{L} {self.input_channels}",
            f"#define INPUT_SIZE_LAYER_{L} {self.input_size}",
            f"#define KERNEL_SIZE_LAYER_{L} {self.kernel_size}",
            f"#define PAD_LAYER_{L} {self.pad}",
            f"#define STRIDE_LAYER_{L} {self.stride}",
            f"#define OUTPUT_SIZE_LAYER_{L} {self.output_size}",
            f"#define OUTPUT_CHANNELS_LAYER_{L} {self.output_channels}",
            f"#define BIAS_CONV_LAYER_{L} {fmt_list(self.bias)}",
            f"#define ZERO_POINT_I_LAYER_{L} {self.zero_point_input}",
            f"#define ZERO_POINT_O_LAYER_{L} {self.zero_point_output}",
            f"#define NUM_CONV_LAYER_{L} {fmt_list(self.num_conv)}",
            f"#define N_CONV_LAYER_{L} {fmt_list(self.n_conv)}",
            f"#define NUM_BANKS_LAYER_{L} {num_banks}",
            ""  # 末尾留一个空行
        ]

        if device_type == DeviceType.DEVICE:
            lines.extend([
                f"__constant int bias_conv_{L}[] = BIAS_CONV_LAYER_{L};",
                f"__constant int num_conv_{L}[] = NUM_CONV_LAYER_{L};",
                f"__constant int n_conv_{L}[] = N_CONV_LAYER_{L};"
            ])
        if device_type == DeviceType.HOST:
            lines.extend([
                f"std::vector<int> bias_conv_{L} = BIAS_CONV_LAYER_{L};",
                f"std::vector<int> num_conv_{L} = NUM_CONV_LAYER_{L};",
                f"std::vector<int> n_conv_{L} = N_CONV_LAYER_{L};"
            ])

        if self.is_csm:
            lines.extend([
                f"#define ZERO_POINT_I_SIG_LAYER_{L} {self.zero_point_input_sig}",
                f"#define ZERO_POINT_O_MUL_LAYER_{L} {self.zero_point_output_mul}",
                f"#define NUM_MUL_LAYER_{L} {self.num_mul}",
                f"#define N_MUL_LAYER_{L} {self.n_mul}",
                ""
            ])
            table_str = ", ".join(map(str, self.sigmoid_table))
            table_len = len(self.sigmoid_table)

            lines.extend([
                f"// Sigmoid Table for Layer {L}: Input x is (raw - zero_point)",
                f"int8_t sigmoid_{L}(int8_t x)",
                "{",
                f"    int8_t table[{table_len}] = {{ {table_str} }};",
                f"    if (x > {self.sigmoid_upper_bound}) return 127;",
                f"    if (x < {self.sigmoid_lower_bound}) return -128;",
                f"    return table[x + {-self.sigmoid_lower_bound}];",
                "}",
                ""
            ])
        return "\n".join(lines)

    def print_weight(self, device_type: DeviceType = DeviceType.DEVICE):
        if not self.valid:
            return ""

        L = self.name

        def fmt_list(lst):
            return "{ " + ", ".join(map(str, lst)) + " }"

        if device_type == DeviceType.DEVICE:
            lines = [
                f"__constant int8_t weights_{L}[] = {fmt_list(self.weight)};"
            ]
        if device_type == DeviceType.HOST:
            lines = [
                f"std::vector<int8_t,  AlignedAllocator<int8_t, 64>> weights_{L} = {fmt_list(self.weight)};"
            ]



        return "\n".join(lines)

    def _extract_input_output_shapes(self):
        input_dims = dataflow.dataflow.get_dims_from_dataflow(self.input_dataflow)
        if self.input_dataflow is not None and len(input_dims) >= 3:
            self.input_channels = (input_dims[1] + 1) // 2 * 2  # 通道数
            if len(input_dims) == 4:  # 2D卷积
                self.input_size = input_dims[2]
            else:
                raise Exception(f"Invalid")
        else:
            raise Exception(f"Invalid")

        weight_dims = dataflow.dataflow.get_dims_from_dataflow(self.weight_dataflow)
        if self.weight_dataflow is not None and len(weight_dims) >= 2:
            self.output_channels = (weight_dims[0] + 1) // 2 * 2  # 输出通道数
        else:
            raise Exception(f"Invalid")

        # 获取输出数据流
        output_dim = dataflow.dataflow.get_dims_from_dataflow(self.output_dataflow)
        if self.output_dataflow is not None and len(output_dim) >= 3:
            if len(output_dim) == 4:  # 2D卷积
                self.output_size = output_dim[2]  # 输出高度
            else:
                raise Exception(f"Invalid")
        else:
            raise Exception(f"Invalid")

        if self.input_size and self.kernel_size and self.stride and self.pad and not self.output_size:
            self.output_size = self._calculate_output_size()

    def _pad_parameters(self, orig_weight, orig_bias, orig_num, orig_n):
        K = self.kernel_size
        orig_c_out = len(orig_bias)
        orig_c_in = self.weight.size // (orig_c_out * K * K)

        reshaped_weight = orig_weight.reshape(orig_c_out, orig_c_in, K, K)
        new_weight = np.zeros((self.output_channels, self.input_channels, K, K), dtype=orig_weight.dtype)
        new_weight[:orig_c_out, :orig_c_in, :, :] = reshaped_weight
        self.weight = new_weight.flatten()

        self.bias = np.zeros(self.output_channels, dtype=orig_bias.dtype)
        self.bias[:orig_c_out] = orig_bias

        new_num = np.zeros(self.output_channels, dtype=orig_num.dtype)
        new_n = np.zeros(self.output_channels, dtype=orig_n.dtype)
        new_num[:orig_c_out] = orig_num
        new_n[:orig_c_out] = orig_n

        self.num_conv = new_num
        self.n_conv = new_n

    def _calculate_output_size(self):
        """根据卷积公式计算输出大小"""
        output_size = (self.input_size + 2 * self.pad - self.kernel_size) // self.stride + 1
        return output_size

    @classmethod
    def _get_real_next_nodes(cls, node: onnx.NodeProto, end_flow_map: Dict[str, List[onnx.NodeProto]]) -> List[
        onnx.NodeProto]:
        """向下搜索，穿透 Q/DQ"""
        real_nodes = []
        for output_name in node.output:
            # print(output_name)
            consumers = end_flow_map.get(output_name, [])
            if not isinstance(consumers, list):
                consumers = [consumers]
            for c in consumers:
                # print(node.name, node.op_type, c.name, c.op_type)
                if c.op_type in ["QuantizeLinear", "DequantizeLinear"]:
                    real_nodes.extend(cls._get_real_next_nodes(c, end_flow_map))
                else:
                    real_nodes.append(c)
        return real_nodes

    @classmethod
    def _check_csm_structure(cls, conv_node: onnx.NodeProto, end_flow_map: Dict[str, List[onnx.NodeProto]]) -> bool:
        next_after_conv = cls._get_real_next_nodes(conv_node, end_flow_map)

        # for n in next_after_conv:
        #     print(n.op_type)

        sigmoid_node = next((n for n in next_after_conv if n.op_type == "Sigmoid"), None)
        if not sigmoid_node:
            return False

        mul_node = next((n for n in next_after_conv if n.op_type == "Mul"), None)
        if not mul_node:
            next_after_sigmoid = cls._get_real_next_nodes(sigmoid_node, end_flow_map)
            mul_node = next((n for n in next_after_sigmoid if n.op_type == "Mul"), None)

        if not mul_node:
            return False

        return True

    @classmethod
    def print_conv_node(cls, device_type: DeviceType = DeviceType.DEVICE):
        all_content = []
        for idx in sorted(list(cls.conv_nodes.keys())):
            node_output = cls.conv_nodes[idx].print_node(device_type)
            all_content.append(node_output)
        return "\n".join(all_content)

    @classmethod
    def print_conv_weight(cls, device_type: DeviceType = DeviceType.DEVICE):
        all_content = []
        for idx in sorted(list(cls.conv_nodes.keys())):
            node_output = cls.conv_nodes[idx].print_weight(device_type)
            all_content.append(node_output)
        return "\n".join(all_content)

    @classmethod
    def print_csm_node(cls, device_type: DeviceType = DeviceType.DEVICE):
        all_content = []
        for idx in sorted(list(cls.csm_nodes.keys())):
            node_output = cls.csm_nodes[idx].print_node(device_type)
            all_content.append(node_output)
        return "\n".join(all_content)

    @classmethod
    def print_csm_weight(cls, device_type: DeviceType = DeviceType.DEVICE):
        all_content = []
        for idx in sorted(list(cls.csm_nodes.keys())):
            node_output = cls.csm_nodes[idx].print_weight(device_type)
            all_content.append(node_output)
        return "\n".join(all_content)

    @classmethod
    def bind_all_gui_node(cls, gui_node_map: Dict[str, BaseNode]):
        for node in cls.conv_nodes.values():
            node.bind_gui_node(gui_node_map)
        for node in cls.csm_nodes.values():
            node.bind_gui_node(gui_node_map)

    def bind_gui_node(self, gui_node_map: Dict[str, BaseNode]):
        gui_node_map[self.name].default_depth = self.output_channels
        gui_node_map[self.name].father = self
        self.gui_node = gui_node_map[self.name]

    @classmethod
    def clear_all_gui_node(cls):
        for node in cls.conv_nodes.values():
            node.clear_gui_node()
        for node in cls.csm_nodes.values():
            node.clear_gui_node()

    def clear_gui_node(self):
        self.gui_node = None

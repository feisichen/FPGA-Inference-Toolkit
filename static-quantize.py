import os
import numpy as np
import time
import onnx
from PIL import Image
import onnxruntime
from onnxruntime.quantization import (
    quantize_static, CalibrationMethod, CalibrationDataReader,
    QuantFormat, QuantType
)

# --- 通用配置：定义通常不建议量化的算子类型 ---
# Concat, Reshape, Transpose, Flatten: 属于排布操作，无计算量，量化反而增加转换开销
# Sigmoid, Softmax: 对精度影响极大，通常保留 FP32
DEFAULT_EXCLUDE_OP_TYPES = {'Concat', 'Reshape', 'Transpose', 'Flatten', 'Softmax', 'Resize'}


class DataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder, model_path):
        self.image_folder = calibration_image_folder
        self.model_path = model_path
        self.preprocess_flag = True
        self.enum_data_dicts = []

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            # 获取模型输入节点信息
            session = onnxruntime.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
            input_node = session.get_inputs()[0]
            input_name = input_node.name
            _, channels, height, width = input_node.shape

            # 处理校准数据
            nhwc_data_list = preprocess_func(self.image_folder, height, width, channels)
            self.enum_data_dicts = iter([{input_name: data} for data in nhwc_data_list])
        return next(self.enum_data_dicts, None)


def preprocess_func(images_folder, height, width, channels=3, size_limit=0):
    """
    通用的预处理函数
    """
    image_names = os.listdir(images_folder)
    if size_limit > 0:
        image_names = image_names[:size_limit]

    processed_data = []
    for image_name in image_names:
        image_path = os.path.join(images_folder, image_name)
        try:
            img = Image.open(image_path).convert("RGB")
            img = img.resize((width, height), Image.BILINEAR)

            # 标准化到 [0, 1]
            input_data = np.array(img).astype(np.float32) / 255.0

            # HWC -> CHW (符合 ONNX 标准输入)
            if channels == 3:
                input_data = input_data.transpose(2, 0, 1)

            # 增加 Batch 维度 (1, C, H, W)
            input_data = np.expand_dims(input_data, axis=0)
            processed_data.append(input_data)
        except Exception as e:
            print(f"跳过无法读取的图片: {image_path}, 错误: {e}")

    return processed_data


def get_nodes_to_exclude(model_path, exclude_types):
    """
    自动解析 ONNX 模型，获取所有指定类型的算子名称
    """
    model = onnx.load(model_path)
    nodes_to_exclude = []
    for node in model.graph.node:
        if node.op_type in exclude_types:
            nodes_to_exclude.append(node.name)
    return nodes_to_exclude


def benchmark(model_path):
    session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_node = session.get_inputs()[0]
    input_name = input_node.name
    input_shape = input_node.shape

    # 将动态维度替换为 1
    fixed_shape = [s if isinstance(s, int) else 1 for s in input_shape]
    input_data = np.zeros(fixed_shape, np.float32)

    # Warm up
    _ = session.run([], {input_name: input_data})

    runs = 20
    start_time = time.perf_counter()
    for _ in range(runs):
        session.run([], {input_name: input_data})
    end_time = time.perf_counter()

    avg_ms = ((end_time - start_time) / runs) * 1000
    print(f"模型 {os.path.basename(model_path)} 平均推理耗时: {avg_ms:.2f} ms")


def main():
    # --- 1. 路径设置 ---
    input_model_path = 'models/yolox_nano.onnx'
    output_model_path = 'models/yolox_nano_ptq.onnx'
    calibration_dataset_path = 'dataset/coco128'

    if not os.path.exists(calibration_dataset_path):
        print(f"错误: 找不到校准数据集目录 {calibration_dataset_path}")
        return

    # --- 2. 自动获取排除节点 ---
    print("正在分析模型结构以提取排除节点...")
    exclude_nodes = get_nodes_to_exclude(input_model_path, DEFAULT_EXCLUDE_OP_TYPES)
    print(f"已自动排除 {len(exclude_nodes)} 个非计算/敏感算子 (类型: {DEFAULT_EXCLUDE_OP_TYPES})")

    # --- 3. 执行量化 ---
    dr = DataReader(calibration_dataset_path, input_model_path)

    print("开始静态 PTQ 量化...")
    quantize_static(
        model_input=input_model_path,
        model_output=output_model_path,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ,  # 推荐使用 QDQ 格式以获得更好的硬件兼容性
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,  # 如果在某些老旧 CPU 上报错，可尝试改为 True
        nodes_to_exclude=exclude_nodes,
        calibrate_method=CalibrationMethod.MinMax  # 默认为 MinMax，精度要求高可尝试 Entropy
    )

    print("-" * 30)
    print("量化完成！开始性能测试...")
    benchmark(input_model_path)
    benchmark(output_model_path)


if __name__ == "__main__":
    main()
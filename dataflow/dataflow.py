import onnx


def get_dims_from_dataflow(dataflow: onnx.ValueInfoProto):
    tensor_type = dataflow.type.tensor_type
    # 提取维度
    dims = []
    if tensor_type.HasField("shape"):
        for d in tensor_type.shape.dim:
            # 维度可能是具体的数值 (dim_value)
            # 也可能是动态的符号 (dim_param，例如 'batch_size')
            if d.HasField("dim_value"):
                dims.append(d.dim_value)
            elif d.HasField("dim_param"):
                dims.append(d.dim_param)  # 保存为字符串，如 "N"
            else:
                dims.append("?")  # 未知维度
    return dims

"""
将老版本的序列化文件转换为新版本
- 使用 temp.json 中的 name 和 view_name 替换 temp20.json.bak 中的 name
- 匹配规则：temp.json 中的 view_name 经过 to_cpp_name 转换后等于 temp20.json.bak 中的 name
- 如果 temp.json 中没有匹配，则保持原 name 不变
"""
import json
import re

def to_cpp_name(name: str) -> str:
    """将 view_name 转换为 cpp 名称格式"""
    clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)

    if clean_name[0].isdigit():
        clean_name = "_" + clean_name

    clean_name = re.sub(r'_+', '_', clean_name)

    return clean_name.strip('_')

def convert_json_file(temp_path, input_path, output_path):
    """转换JSON文件"""
    # 读取 temp.json（新名称来源）
    with open(temp_path, 'r', encoding='utf-8') as f:
        temp_data = json.load(f)
    
    # 读取 temp20.json.bak（待转换文件）
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 构建 view_name -> (name, view_name) 的映射
    # key: to_cpp_name(view_name), value: (name, view_name)
    name_mapping = {}
    temp_nodes = temp_data.get("nodes", [])
    for node in temp_nodes:
        view_name = node.get("view_name", "")
        cpp_name = to_cpp_name(view_name)
        new_name = node.get("name", "")
        name_mapping[cpp_name] = (new_name, view_name)
    
    # 转换节点
    nodes = data.get("nodes", [])
    matched_count = 0
    unmatched_nodes = []
    
    for node in nodes:
        old_name = node.get("name", "")
        
        # 尝试匹配
        if old_name in name_mapping:
            new_name, view_name = name_mapping[old_name]
            node["name"] = new_name
            node["view_name"] = view_name
            matched_count += 1
        else:
            # 没有匹配，保持原 name，view_name 设为原 name
            node["view_name"] = old_name
            unmatched_nodes.append(old_name)
    
    # 转换边：删除 base_name 和 extra_name 字段
    edges = data.get("edges", [])
    for edge in edges:
        if "base_name" in edge:
            del edge["base_name"]
        if "extra_name" in edge:
            del edge["extra_name"]
    
    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成: {output_path}")
    print(f"总节点数: {len(nodes)}")
    print(f"匹配成功: {matched_count}")
    print(f"未匹配: {len(unmatched_nodes)}")
    print(f"边数量: {len(edges)}")
    
    if unmatched_nodes:
        print(f"\n未匹配的节点名称 (前10个):")
        for name in unmatched_nodes[:10]:
            print(f"  {name}")
        if len(unmatched_nodes) > 10:
            print(f"  ... 还有 {len(unmatched_nodes) - 10} 个")

if __name__ == "__main__":
    convert_json_file("temp.json", "temp20.json.bak", "temp20.json")

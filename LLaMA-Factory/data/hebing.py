import json

# 文件路径
file1_path = '/root/user/zsy/LLaMA-Factory/eval_train/62/qwen2.5vl-3b/mvsa_fine_tuning_explain.json'
file2_path = '/root/user/zsy/LLaMA-Factory/data/mvsa_train.json'
output_path = '/root/user/zsy/LLaMA-Factory/data/mvsa_sc.json'

# 读取两个 JSON 文件
with open(file1_path, 'r', encoding='utf-8') as f1:
    data1 = json.load(f1)

with open(file2_path, 'r', encoding='utf-8') as f2:
    data2 = json.load(f2)

# 拼接两个列表
merged_data = data1 + data2

# 保存合并后的结果
with open(output_path, 'w', encoding='utf-8') as fout:
    json.dump(merged_data, fout, indent=2, ensure_ascii=False)

print(f"成功合并 {len(data1)} + {len(data2)} 条，共 {len(merged_data)} 条数据，保存为 {output_path}")

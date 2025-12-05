import json

# 读取合并后的数据集
with open('/root/user/case_study/2015.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 筛选出 predict 与 label 不一致的样本
disagreement_data = []
for item in data:
    predict = item.get('predict', '').strip()
    label = item.get('label', '').strip()
    if predict != label:
        disagreement_data.append(item)

# 保存到新文件
output_file = '/root/user/case_study/wrong_samples_15.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(disagreement_data, f, ensure_ascii=False, indent=2)

print(f"共找到 {len(disagreement_data)} 条 predict 与 label 不一致的样本。")
print(f"已保存到 {output_file}")
import json

# 读取两个 JSON 文件
with open('/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/1-text_aspect_results.json', 'r', encoding='utf-8') as f1:
    data1 = json.load(f1)

with open('/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/2-image_aspect_results.json', 'r', encoding='utf-8') as f2:
    data2 = json.load(f2)

# 检查两个数据集长度是否一致
if len(data1) != len(data2):
    raise ValueError("两个数据集的长度不一致，无法按顺序合并！")

# 按索引顺序合并
merged_result = []
for i in range(len(data1)):
    item1 = data1[i]
    item2 = data2[i]

    # 可选：验证关键字段是否匹配（防止错位）
    if item1['image_id'] != item2['image_id'] or item1['text'] != item2['text']:
        raise ValueError(f"索引 {i} 处的条目不匹配！\ndata1: {item1['image_id']}, {item1['text']}\ndata2: {item2['image_id']}, {item2['text']}")

    # 合并字段
    merged_item = {
        **item1,           # 包含 text_aspect_analysis
        'image_aspect_analysis': item2.get('image_aspect_analysis')  # 添加 image_aspect_analysis
    }
    merged_result.append(merged_item)

# 保存合并后的结果
with open('/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/12-dataset_merged.json', 'w', encoding='utf-8') as out_file:
    json.dump(merged_result, out_file, indent=2, ensure_ascii=False)

print("✅ 两个数据集已按顺序成功合并并保存为 'merged_dataset.json'")
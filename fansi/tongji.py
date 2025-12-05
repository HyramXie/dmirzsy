import json
import re

# 加载数据集
with open('/root/user/LLaMA-Factory/32B/2017/final_predict_all.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 用于提取 <target>...</target> 中内容的正则
pattern = re.compile(r"<target>(.*?)</target>")

# 存储处理后的结果
updated_data = []

for item in data:
    prompt = item["prompt"]
    match = pattern.search(prompt)
    aspect = match.group(1) if match else None  # 提取标签内文本，若无则为 None

    # 创建新字典或添加字段
    item["aspect"] = aspect
    updated_data.append(item)

# 保存回新文件
with open('predict_all_aspect.json', 'w', encoding='utf-8') as f:
    json.dump(updated_data, f, indent=2, ensure_ascii=False)

print(f"✅ 处理完成！共 {len(updated_data)} 条数据，已保存到 'predict_all_aspect.json'")

# import json
# from os.path import basename

# # 读取两个数据集
# with open('/root/user/LLaMA-Factory/32B/2017/predict_all_aspect.json', 'r', encoding='utf-8') as f:
#     dataset1 = json.load(f)

# with open('/root/user/LLaMA-Factory/32B/2017/3-fused_results.json', 'r', encoding='utf-8') as f:
#     dataset2 = json.load(f)

# # 构建 dataset2 的索引：(image_id, aspect) -> 所需字段
# dataset2_map = {}
# for item in dataset2:
#     key = (item['image_id'], item['aspect'])
#     dataset2_map[key] = {
#         'text_sentiment': item['text_sentiment'],
#         'image_sentiment': item['image_sentiment']
#     }

# # 准备新数据集
# merged_data = []

# for item1 in dataset1:
#     image_path = item1['image']
#     image_id = basename(image_path)  # 提取文件名，如 "1739565.jpg"
#     aspect = item1['aspect']
    
#     # 查找 dataset2 中匹配的项
#     key = (image_id, aspect)
#     if key in dataset2_map:
#         merged_item = {
#             "image": image_path,
#             "prompt": item1["prompt"],
#             "aspect": aspect,
#             "text_sentiment": dataset2_map[key]['text_sentiment'],
#             "image_sentiment": dataset2_map[key]['image_sentiment'],
#             "final_sentiment": item1["final_sentiment"],
#             "label": item1["label"]
#         }
#         merged_data.append(merged_item)
#     else:
#         # 可选：打印未匹配项用于调试
#         print(f"Warning: No match for image_id={image_id}, aspect={aspect}")

# # 保存合并后的数据集
# with open('tongji.json', 'w', encoding='utf-8') as f:
#     json.dump(merged_data, f, indent=2, ensure_ascii=False)

# print(f"✅ 合并完成！共 {len(merged_data)} 条数据已保存到 'tongji.json'")
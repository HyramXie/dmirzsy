# import json

# # 读取第一个数据集
# with open('/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/w2.json', 'r') as f:
#     dataset1 = json.load(f)

# # 读取第二个数据集
# with open('/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/test+image_context.json', 'r') as f:
#     dataset2 = json.load(f)

# # 创建一个以 image_id 为 key 的字典，方便查找
# dataset2_dict = {item['image_id']: item for item in dataset2}

# # 合并两个数据集
# merged_dataset = []

# for item1 in dataset1:
#     image_id = item1['image_id']
#     if image_id in dataset2_dict:
#         # 合并两个字典
#         merged_item = {**item1, **dataset2_dict[image_id]}
#         merged_dataset.append(merged_item)

# # 保存合并后的数据集
# with open('/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/merged_dataset_2.json', 'w') as f:
#     json.dump(merged_dataset, f, indent=2)

# print("合并完成，结果已保存为 merged_dataset.json")

import json

# 读取两个 JSON 文件
with open('/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2017/gemma3_results.json', 'r', encoding='utf-8') as f:
    data1 = json.load(f)

with open('/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2017/test+image_context.json', 'r', encoding='utf-8') as f:
    data2 = json.load(f)

# 构建 dataset2 的索引：以 image_id 为键（去掉路径，只取文件名）
data2_dict = {}
for item in data2:
    # 提取 image_id 的文件名部分，如 "74960.jpg"
    img_id = item['image_id']
    data2_dict[img_id] = item

# 合并数据
merged_dataset = []

for item1 in data1:
    # 从 image 字段提取文件名，如 "twitter2015_images/74960.jpg" -> "74960.jpg"
    image_path = item1['image']
    image_filename = image_path.split('/')[-1]

    # 查找对应的 data2 条目
    if image_filename in data2_dict:
        merged_item = {**item1, **data2_dict[image_filename]}
        merged_dataset.append(merged_item)
    else:
        print(f"Warning: No matching entry in dataset2 for {image_filename}")

# 保存合并后的结果
with open('/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2017/gemma3_eval.json', 'w', encoding='utf-8') as f:
    json.dump(merged_dataset, f, ensure_ascii=False, indent=4)

print("✅ 合并完成，结果已保存到 'merged_dataset.json'")
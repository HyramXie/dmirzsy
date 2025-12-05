import os
import shutil
import json

# 设置路径
input_json_file = '/root/user/zsy/LLaMA-Factory/data/mvsa_train.json'
source_image_folder = 'root/user/zsy/MVSA_Single/data'
output_folder = '/root/user/zsy/LLaMA-Factory/data/MVSA/train'

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 读取 JSON 文件（完整数组）
with open(input_json_file, 'r', encoding='utf-8') as f:
    try:
        data = json.load(f)  # 整个 JSON 数组一次性加载
    except json.JSONDecodeError as e:
        print("JSON 格式错误:", e)
        exit(1)

# 提取唯一的 image_id 集合
image_ids = set(item.get('image_id') for item in data if item.get('image_id'))

# 复制图片
for image_id in image_ids:
    source_path = os.path.join(source_image_folder, image_id)
    dest_path = os.path.join(output_folder, image_id)

    if os.path.exists(source_path):
        shutil.copy(source_path, dest_path)
        # print(f"✅ 已复制: {image_id}")
    else:
        print(f"❌ 图片不存在: {image_id}")

print("🎉 图片提取完成！")
count = sum(1 for name in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, name)))
print(f"{output_folder} 共 {count} 张图片")
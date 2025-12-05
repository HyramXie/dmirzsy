import json

# 读取原始数据
input_file = '/root/user/case_study/wrong_samples_17.json'  # 替换为你的文件名
output_file = '/root/user/case_study/wrong_samples_171.json'

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 添加序号（从1开始）
for idx, item in enumerate(data, start=1):
    item['serial_number'] = idx  # 插入新字段

# 保存到新文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"已成功为 {len(data)} 条数据添加序号，保存至 {output_file}")
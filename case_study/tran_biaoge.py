import json
import pandas as pd

# 1. 读取 JSON 数据
input_json = '/root/user/case_study/wrong_samples_17.json'  # 替换为你的文件路径
with open(input_json, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. 定义列顺序
columns = [
    "serial_number",
    "image_id",
    "image_context",
    "text",
    "aspect",
    "predict",
    "label"
]

# 3. 转换为 DataFrame，并按指定顺序排列列
df = pd.DataFrame(data)[columns]

# 4. 保存为 CSV（推荐 UTF-8 编码）
output_csv = 'wrong_samples_17.csv'
df.to_csv(output_csv, index=False, encoding='utf-8')
print(f"✅ 数据已保存为 CSV：{output_csv}")

# 5. 保存为 Excel (.xlsx)
output_xlsx = 'wrong_samples_17.xlsx'
df.to_excel(output_xlsx, index=False, engine='openpyxl')
print(f"✅ 数据已保存为 Excel：{output_xlsx}")
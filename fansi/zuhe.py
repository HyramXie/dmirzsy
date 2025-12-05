import json

# 指定四个 JSON 文件路径
# file_paths = ['/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/final_predict.json', '/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/final_predict1.json', '/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/final_predict2.json', '/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/final_predict3.json']
file_paths = ['/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/final_predict.json', '/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/final_predict1.json', '/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/final_predict4.json']

# 合并数据
merged_data = []

for path in file_paths:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # 若文件中是一个 dict 列表
        if isinstance(data, list):
            merged_data.extend(data)
        else:
            raise ValueError(f"{path} does not contain a list of JSON objects.")

# 保存合并后的结果
with open('/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/final_predict_all.json', 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, indent=2, ensure_ascii=False)

print(f"合并完成，共 {len(merged_data)} 条数据，已保存为 merged_data.json")

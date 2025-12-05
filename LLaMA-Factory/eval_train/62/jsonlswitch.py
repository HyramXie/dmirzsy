import json

# 输入的jsonl文件路径
input_file = '/root/user/LLaMA-Factory/eval_train/62/qwen2.5vl-3b/17_explain.jsonl'
# 输出的json文件路径
output_file = '/root/user/LLaMA-Factory/eval_train/62/qwen2.5vl-3b/17_explain.json'

# 用于存储所有json对象的列表
data = []

# 逐行读取jsonl文件
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        # 将每行转换为字典并添加到列表中
        data.append(json.loads(line.strip()))

# 将整个列表写入json文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"转换完成，已保存为 {output_file}")
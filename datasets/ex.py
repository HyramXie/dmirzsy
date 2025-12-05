import json

# # 假设你的数据存储在一个名为 data.json 的文件中
# with open('/root/user/datasets/twitter2015/test+image_context.json', 'r', encoding='utf-8') as f:
#     dataset = json.load(f)

# # 提取 sentiment 值并组成字符串
# sentiment_sequence = ','.join(str(entry['sentiment']) for entry in dataset)

# # 打印结果
# print(sentiment_sequence)

# 定义映射关系
sentiment_map = {
    "Positive": "1",
    "Neutral": "0",
    "Negative": "-1"
}

# 存储结果
result = []

# 读取 jsonl 文件
with open('/root/user/LLaMA-Factory/eval_o/qwen2.5vl-3b/15_t+c+i.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())
        predict = data.get("predict")
        if predict in sentiment_map:
            result.append(sentiment_map[predict])
        else:
            # 可选：处理未知的 sentiment 值
            result.append("")  # 或者抛出错误、跳过等

# 输出结果
print(','.join(result))
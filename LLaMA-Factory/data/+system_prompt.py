import json

# 指定系统提示词
system_prompt = (
    "You are a social media sentiment analysis expert, specialized in multimodal aspect-based sentiment understanding. "
    "Your task is to analyze an image, its description, and a text with a marked <target>, and determine the sentiment "
    "expressed towards the target. Always answer with exactly one word: \"positive\", \"neutral\", or \"negative\"."
)

# 输入输出文件路径
input_file = "/root/user/LLaMA-Factory/data/2015/t+i+c+target_15.json"   # 你的原始数据集
output_file = "/root/user/LLaMA-Factory/data/2015/t+i+c+target+sys_15.json"  # 加了system后的新数据集

# 读取原始数据
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 处理数据
for item in data:
    # 确保 messages 存在
    if "messages" in item:
        # 在开头插入 system prompt
        item["messages"].insert(0, {
            "role": "system",
            "content": system_prompt
        })

# 保存新数据
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"已处理完成，保存到 {output_file}")

import json
import re
from collections import Counter

# 加载已生成的结果文件
input_path = "/root/user/zsy/fansi/1-react.json"
output_path = "/root/user/zsy/fansi/2-inconsistent_predictions.json"

# 支持的情感类别
SENTIMENTS = {"Positive", "Neutral", "Negative"}

def extract_sentiment(text):
    """
    从字符串中提取第一个出现的 Positive/Neutral/Negative（不区分大小写）
    """
    match = re.search(r'\b(Positive|Neutral|Negative)\b', text, re.IGNORECASE)
    if match:
        return match.group(1)  # 返回首字母大写的正确形式
    return None

# 存储不一致的样本
inconsistent_samples = []

print("🔍 正在分析模型输出的一致性...")

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    outputs = item.get("model_outputs", [])
    
    # 提取每个输出中的 sentiment
    sentiments = []
    for out in outputs:
        sent = extract_sentiment(out)
        if sent:
            sentiments.append(sent)
        else:
            sentiments.append("Unknown")  # 无法识别也算作一种“不一致”

    # 只保留有效情感标签
    valid_sentiments = [s for s in sentiments if s in SENTIMENTS]

    # 如果少于 2 个有效标签，也视为不一致（或可选跳过）
    if len(valid_sentiments) < 2:
        inconsistent_samples.append({
            "original": item,
            "extracted_sentiments": sentiments,
            "reason": "Too few valid sentiments extracted"
        })
        continue

    # 检查是否所有情感都相同
    sentiment_counter = Counter(valid_sentiments)
    if len(sentiment_counter) > 1:
        # 存在多种不同情感 → 不一致
        inconsistent_samples.append({
            "original": item,
            "extracted_sentiments": sentiments,
            "distribution": dict(sentiment_counter)
        })

# 保存结果
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(inconsistent_samples, f, indent=2, ensure_ascii=False)

print(f"✅ 筛选完成！共找到 {len(inconsistent_samples)} 个输出不一致的样本。")
print(f"📁 已保存至: {output_path}")

# 可选：打印一些统计信息
if inconsistent_samples:
    print("\n📊 示例不一致情况:")
    for i, sample in enumerate(inconsistent_samples[:3]):
        img = sample["original"]["images"][0]
        sents = sample["extracted_sentiments"]
        print(f"  [{i+1}] {img} -> {sents}")
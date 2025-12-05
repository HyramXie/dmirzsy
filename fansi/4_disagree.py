import json
import re
from collections import Counter

# 输入文件：包含反思输出的结果
input_file = "/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/3-reflected_judgments.json"
# 输出文件：仅保留第二轮仍不一致的样本
output_file = "/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/4-reflected_inconsistent_again.json"

# 支持的情感标签
SENTIMENTS = {"Positive", "Neutral", "Negative"}

def extract_final_sentiment(text):
    """
    从反思输出中提取 Final Sentiment 后的极性
    示例输入: "Final Sentiment: Negative. Reason: ..."
    """
    match = re.search(r"Final\s+Sentiment\s*:\s*([a-zA-Z]+)", text, re.IGNORECASE)
    if match:
        word = match.group(1).capitalize()
        if word in SENTIMENTS:
            return word
    return None

# 存储第二轮仍不一致的样本
still_inconsistent = []

print("🔍 正在分析反思后的输出一致性...")

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    try:
        reflected_outputs = item.get("reflected_outputs", [])
        
        # 提取每条反思输出中的最终情感
        sentiments = []
        for output in reflected_outputs:
            sent = extract_final_sentiment(output)
            sentiments.append(sent)

        # 过滤掉 None（无法解析的）
        valid_sentiments = [s for s in sentiments if s is not None]

        # 如果有效标签少于2个，视为不一致
        if len(valid_sentiments) < 2:
            status = "inconsistent (low valid)"
        else:
            # 检查是否全部相同
            if len(set(valid_sentiments)) > 1:
                status = "inconsistent"
            else:
                status = "consistent"

        # 只保留仍不一致的
        if status == "inconsistent" or len(valid_sentiments) < 2:
            still_inconsistent.append({
                "original_inconsistent_sample": item["original_inconsistent_sample"],
                "first_round_votes": item["first_round_votes"],
                "first_round_outputs": item["first_round_outputs"],
                "distribution_after_first": item["distribution"],
                "reflected_outputs": reflected_outputs,
                "reflected_sentiments": sentiments,
                "reflected_valid_count": len(valid_sentiments),
                "reflected_distribution": dict(Counter(valid_sentiments)) if valid_sentiments else {}
            })

    except Exception as e:
        print(f"Error processing item: {e}")
        continue

# 保存结果
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(still_inconsistent, f, indent=2, ensure_ascii=False)

print(f"✅ 筛选完成！共找到 {len(still_inconsistent)} 个样本在反思后仍不一致。")
print(f"📁 已保存至: {output_file}")
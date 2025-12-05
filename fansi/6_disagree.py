import json
import re
from collections import Counter

# 输入文件：第三轮元反思结果
input_file = "/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/5-final_reflection_round.json"
# 输出文件：第三轮后仍不一致的样本
output_file = "/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/6-final_inconsistent_hallucinations.json"

# 支持的情感标签
SENTIMENTS = {"Positive", "Neutral", "Negative"}

def extract_final_decision(text):
    """
    从 final_meta_reflection 中提取 Final Decision 后的情感极性
    示例输入: "Final Decision: Neutral. Rationale: ..."
    """
    match = re.search(r"Final\s+Decision\s*:\s*([a-zA-Z]+)", text, re.IGNORECASE)
    if match:
        word = match.group(1).capitalize()
        if word in SENTIMENTS:
            return word
    return None

# 存储最终仍不一致的样本
final_inconsistent = []

print("🔍 正在分析第三轮元反思结果的一致性...")

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    try:
        meta_outputs = item.get("final_meta_reflection", [])
        
        # 提取每条中的 Final Decision
        decisions = []
        for output in meta_outputs:
            dec = extract_final_decision(output)
            decisions.append(dec)

        # 过滤有效标签
        valid_decisions = [d for d in decisions if d is not None]

        # 如果有效数量 < 2，视为不一致
        if len(valid_decisions) < 2:
            status = "inconsistent (low valid)"
        else:
            # 检查是否所有有效决策都相同
            if len(set(valid_decisions)) > 1:
                status = "inconsistent"
            else:
                status = "consistent"

        # 只保留仍不一致的
        if status == "inconsistent" or len(valid_decisions) < 2:
            final_inconsistent.append({
                "original_inconsistent_sample": item["original_inconsistent_sample"],
                "first_round_votes": item["first_round_votes"],
                "first_round_outputs": item["first_round_outputs"],
                "reflected_outputs": item["reflected_outputs"],
                "reflected_distribution": item["reflected_distribution"],
                "final_meta_reflection": meta_outputs,
                "final_decisions": decisions,
                "valid_decision_count": len(valid_decisions),
                "final_distribution": dict(Counter(valid_decisions)) if valid_decisions else {}
            })

    except Exception as e:
        print(f"Error processing item: {e}")
        continue

# 保存结果
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(final_inconsistent, f, indent=2, ensure_ascii=False)

print(f"✅ 筛选完成！共找到 {len(final_inconsistent)} 个样本在三轮反思后仍不一致。")
print(f"📁 已保存至: {output_file}")

# 可选：打印统计信息
if final_inconsistent:
    print("\n📊 最终不一致样本的情感分布示例：")
    for i, sample in enumerate(final_inconsistent[:3]):
        img = sample["original_inconsistent_sample"]["images"][0]
        dist = sample["final_distribution"]
        print(f"  [{i+1}] {img} -> {dist}")
import json
from collections import Counter

# 输入文件：包含 final_decisions 和 final_distribution 的数据（如 final_inconsistent_hallucinations.json 或 final_reflection_round.json）
input_file = "/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/7-reflected_inconsistent_again.json"
# 输出文件：多数投票后的最终结果
output_file = "/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/final_predict4.json"

# 存储结果
majority_voted_results = []

print("🗳️ 正在对 final_meta_reflection 进行多数投票...")

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    try:
        # 获取原始样本信息
        original_sample = item["original_inconsistent_sample"]
        image_path = original_sample["images"][0]

        # 提取 prompt（用户输入内容）
        user_msg = next(msg for msg in original_sample["messages"] if msg["role"] == "user")
        raw_prompt = user_msg["content"]
        # 清理：只保留文本部分
        prompt_clean = raw_prompt.split("\n\nBased on the image")[0]
        prompt_clean = prompt_clean.replace("Image: <image>\nText: ", "").strip().strip('"')

        # 提取 label（原始标注）
        assistant_msg = next(msg for msg in original_sample["messages"] if msg["role"] == "assistant")
        label = assistant_msg["content"].strip()

        # 验证 label 合法性
        if label not in {"Positive", "Neutral", "Negative"}:
            print(f"⚠️ Invalid label: {label}, skipping {image_path}")
            continue

        # 获取 final_decisions 投票分布
        final_distribution = item.get("reflected_distribution", {})
        if not final_distribution:
            continue  # 无有效分布

        # 找出票数最多的情感（多数投票）
        max_votes = 0
        final_sentiment = None
        for sent, count in final_distribution.items():
            if sent in {"Positive", "Neutral", "Negative"} and count > max_votes:
                max_votes = count
                final_sentiment = sent

        if final_sentiment is None:
            continue  # 无效

        # 保存结果
        majority_voted_results.append({
            "image": image_path,
            "prompt": prompt_clean,
            "final_sentiment": final_sentiment,
            "label": label
        })

    except Exception as e:
        print(f"Error processing item: {e}")
        continue

# 保存结果
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(majority_voted_results, f, indent=2, ensure_ascii=False)

print(f"✅ 多数投票完成！共处理 {len(majority_voted_results)} 个样本。")
print(f"📁 已保存至: {output_file}")
import json
import re
from tqdm import tqdm

# 文件路径
input_file = "/root/user/LLaMA-Factory/32B/2015/10-final_synthesis_judgment.json"
output_file = "/root/user/LLaMA-Factory/32B/2015/final_predict3.json"

# 支持的情感标签
SENTIMENTS = {"Positive", "Neutral", "Negative"}

def extract_final_decision(text):
    """
    从 final_meta_reflection 中提取 Final Decision 后的情感
    支持格式：Final Decision: Positive / **Positive** / 各种变体
    """
    # 先尝试从 Final Decision 提取
    match = re.search(r"Final\s+Decision\s*:\s*([a-zA-Z]+)", text, re.IGNORECASE)
    if match:
        word = match.group(1).strip(" *")
        if word in SENTIMENTS:
            return word.capitalize()

    # 如果没匹配到，尝试从粗体或直接关键词提取
    match = re.search(r"\*\*([a-zA-Z]+)\*\*", text)  # **Positive**
    if match:
        word = match.group(1)
        if word in SENTIMENTS:
            return word.capitalize()

    # 最后 fallback：找第一个出现的 Positive/Neutral/Negative
    match = re.search(r'\b(Positive|Neutral|Negative)\b', text, re.IGNORECASE)
    if match:
        return match.group(1)

    return None

# 存储结果
consistent_final = []

print("🔍 正在筛选 final_meta_reflection 五次预测一致的样本...")

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in tqdm(data):
    try:
        meta_outputs = item.get("final_synthesis_judgment", [])
        if len(meta_outputs) != 5:
            continue  # 必须正好5条

        # 提取每条中的情感
        decisions = [extract_final_decision(out) for out in meta_outputs]
        valid_decisions = [d for d in decisions if d is not None]

        # 要求：5条都有效，且完全一致
        if len(valid_decisions) != 5:
            continue
        if len(set(valid_decisions)) != 1:
            continue

        final_sentiment = valid_decisions[0]

        # 获取原始样本
        original_sample = item["original_inconsistent_sample"]

        # 提取 prompt（清理）
        user_msg = next(msg for msg in original_sample["messages"] if msg["role"] == "user")
        raw_prompt = user_msg["content"]
        prompt_clean = raw_prompt.split("\n\nBased on the image")[0]
        prompt_clean = prompt_clean.replace("Image: <image>\nText: ", "").strip().strip('"')

        # 提取 label（assistant 的 content）
        assistant_msg = next(msg for msg in original_sample["messages"] if msg["role"] == "assistant")
        label = assistant_msg["content"].strip()
        if label not in SENTIMENTS:
            print(f"⚠️ Invalid label: {label}, skipping {original_sample['images'][0]}")
            continue

        # 保存结果
        consistent_final.append({
            "image": original_sample["images"][0],
            "prompt": prompt_clean,
            "final_sentiment": final_sentiment,
            "label": label
        })

    except Exception as e:
        print(f"Error processing {item.get('image', 'unknown')}: {e}")
        continue

# 保存结果
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(consistent_final, f, indent=2, ensure_ascii=False)

print(f"✅ 筛选完成！共找到 {len(consistent_final)} 个在第三轮后五次预测一致的样本。")
print(f"📁 已保存至: {output_file}")
import json
import re
from tqdm import tqdm

# 输入文件：包含 reflected_outputs 的文件（如 reflected_judgments.json）
input_file = "/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/6-reflected_judgments.json"
# 输出文件：reflected_outputs 五次预测一致的样本
output_file = "/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/final_predict1.json"

# 支持的情感标签
SENTIMENTS = {"Positive", "Neutral", "Negative"}

def extract_sentiment_from_reflected(text):
    """
    从 reflected_outputs 中提取 Final Sentiment 后的情感
    示例："Final Sentiment: Negative. Reason: ..." → "Negative"
    """
    match = re.search(r"Final\s+Sentiment\s*:\s*([a-zA-Z]+)", text, re.IGNORECASE)
    if match:
        word = match.group(1).capitalize()
        if word in SENTIMENTS:
            return word
    return None

# 存储结果
consistent_after_reflection = []

print("🔍 正在筛选 reflected_outputs 五次预测一致的样本...")

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in tqdm(data):
    try:
        # 获取反思阶段的输出
        reflected_outputs = item.get("reflected_outputs", [])
        if len(reflected_outputs) != 5:
            continue  # 确保正好5条

        # 提取每条中的情感
        sentiments = []
        for output in reflected_outputs:
            sent = extract_sentiment_from_reflected(output)
            sentiments.append(sent)

        # 过滤有效标签
        valid_sentiments = [s for s in sentiments if s is not None]

        # 要求：5条都有效，且完全一致
        if len(valid_sentiments) != 5:
            continue
        if len(set(valid_sentiments)) != 1:
            continue

        final_sentiment = valid_sentiments[0]  # 唯一的情感

        # 获取原始样本数据
        original = item["original_inconsistent_sample"]

        # 提取 prompt（user 内容，清理）
        user_msg = next(msg for msg in original["messages"] if msg["role"] == "user")
        raw_prompt = user_msg["content"]
        # 清理：去掉 Image: <image> 和 image description
        prompt_clean = raw_prompt.split("\n\nBased on the image")[0]
        prompt_clean = prompt_clean.replace("Image: <image>\nText: ", "").strip().strip('"')

        # 提取 label（assistant 的 content）
        assistant_msg = next(msg for msg in original["messages"] if msg["role"] == "assistant")
        label = assistant_msg["content"].strip()

        # 验证 label 合法性
        if label not in SENTIMENTS:
            print(f"⚠️ Invalid label: {label}, skipping {original['images'][0]}")
            continue

        # 保存结果
        consistent_after_reflection.append({
            "image": original["images"][0],
            "prompt": prompt_clean,
            "final_sentiment": final_sentiment,
            "label": label
        })

    except StopIteration:
        print(f"⚠️ Missing user or assistant message in {original.get('images', ['unknown'])[0]}")
        continue
    except Exception as e:
        print(f"Error processing item: {e}")
        continue

# 保存结果
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(consistent_after_reflection, f, indent=2, ensure_ascii=False)

print(f"✅ 筛选完成！共找到 {len(consistent_after_reflection)} 个在反思后五次预测一致的样本。")
print(f"📁 已保存至: {output_file}")
import json
import re
from tqdm import tqdm

# 输入文件路径
input_file = "/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/1-react.json"
# 输出文件路径：包含一致预测 + 原始 label
output_file = "/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/final_predict.json"

# 支持的情感标签
SENTIMENTS = {"Positive", "Neutral", "Negative"}

def extract_sentiment(text):
    """
    从文本中提取第一个出现的 Positive/Neutral/Negative（不区分大小写）
    """
    match = re.search(r'\b(Positive|Neutral|Negative)\b', text, re.IGNORECASE)
    if match:
        return match.group(1)  # 返回标准首字母大写形式
    return None

# 存储结果
consistent_results = []

print("🔍 正在筛选 model_outputs 五次预测一致的样本，并提取原始 label...")

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in tqdm(data):
    try:
        # 提取模型五次输出
        model_outputs = item.get("model_outputs", [])
        if len(model_outputs) < 5:
            continue  # 确保有5条生成结果

        # 从 model_outputs 提取情感
        sentiments = [extract_sentiment(out) for out in model_outputs]
        valid_sentiments = [s for s in sentiments if s is not None]

        # 要求 5 条都有效且完全一致
        if len(valid_sentiments) != 5 or len(set(valid_sentiments)) != 1:
            continue

        final_sentiment = valid_sentiments[0]

        # 提取原始 prompt（user 内容，去除非文本部分）
        user_msg = next(msg for msg in item["messages"] if msg["role"] == "user")
        raw_prompt = user_msg["content"]
        # 清理：去掉 "Image: <image>" 和 image description，保留纯文本
        prompt_clean = raw_prompt.split("\n\nBased on the image")[0]
        prompt_clean = prompt_clean.replace("Image: <image>\nText: ", "").strip().strip('"')

        # 提取 label 并标准化
        assistant_msg = next(msg for msg in item["messages"] if msg["role"] == "assistant")
        label = assistant_msg["content"].strip().capitalize()  # 确保首字母大写

        if label not in SENTIMENTS:
            print(f"⚠️ Invalid label detected: {label}, skipping {item['images'][0]}")
            continue

        # 保存结果
        consistent_results.append({
            "image": item["images"][0],
            "prompt": prompt_clean,
            "final_sentiment": final_sentiment,
            "label": label
        })

    except StopIteration:
        print(f"⚠️ Missing user or assistant message in {item.get('images', ['unknown'])[0]}")
        continue
    except Exception as e:
        print(f"Error processing item: {e}")
        continue

# 保存结果
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(consistent_results, f, indent=2, ensure_ascii=False)

print(f"✅ 筛选完成！共找到 {len(consistent_results)} 个五次预测一致的样本。")
print(f"📁 已保存至: {output_file}")
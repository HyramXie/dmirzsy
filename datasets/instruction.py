import json

# 情感标签映射
sentiment_map = {
    "-1": "Negative",
    "0": "Neutral",
    "1": "Positive"
}

# 加载原始数据
with open("/root/user/datasets/twitter2017/train+image_context.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

converted_data = []

for item in raw_data:
    text = item.get("text", "").strip()
    aspect = item.get("aspect", "").strip()
    sentiment = item.get("sentiment", "").strip()
    image_context = (item.get("image_context") or "").strip()

    output = sentiment_map.get(sentiment, "Unknown")

    instruction = (
    f"Given the following text and image context, determine the sentiment polarity "
    f"(positive, neutral, or negative) specifically toward the aspect <target>{aspect}</target>. "
    f"Consider both the textual expression and the visual scene in your judgment. "
    f"Respond with only one word: Positive, Neutral, or Negative."
)

    input_text = f"Text: {text}\nImage context: {image_context}"

    converted_data.append({
        "instruction": instruction,
        "input": input_text,
        "output": output
    })

# 保存结果
with open("/root/user/LLaMA-Factory/data/instruction_train_17.json", "w", encoding="utf-8") as f:
    json.dump(converted_data, f, indent=2, ensure_ascii=False)

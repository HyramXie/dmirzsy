import json

label_map = {
    "1": "Positive",
    "0": "Neutral",
    "-1": "Negative"
}

# 读取数据集
with open("/home/cbf00006701/zsy/datasets/twitter2017/train+image_context+syn.json", "r", encoding="utf-8") as fin:
    data = json.load(fin)

llava_finetune_data = []

for item in data:
    text = item["text"]
    aspect = item["aspect"]
    sentiment = label_map.get(str(item["sentiment"]), "Neutral")
    image = item["image_id"]
    image_context = item.get("image_context", "")
    syntax_info = item.get("syntax_info", "")

    prompt = f"""Given the following information:

- Text: "{text}"
- Syntax Analysis: "{syntax_info}"
- Aspect: "{aspect}"
- Image Context: "{image_context}"

Please determine the sentiment polarity (positive, neutral, or negative) toward the aspect "{aspect}" by considering the text, its syntactic structure, and the image context.

Respond with only one word: Positive, Neutral, or Negative."""

    llava_finetune_data.append({
        "messages": [
            {
                "content": prompt,
                "role": "user"
            },
            {
                "content": sentiment,
                "role": "assistant"
            }
        ]
    })

# 保存为 LLaVA 微调格式的 JSON
with open("/home/cbf00006701/zsy/LLaMA-Factory/data/train+context+syn_17.json", "w", encoding="utf-8") as fout:
    json.dump(llava_finetune_data, fout, indent=4, ensure_ascii=False)

import json
from tqdm import tqdm
import re

# 文件路径
explain_jsonl_path = "/root/user/LLaMA-Factory/eval_train/llava1.5-7b/17_explain.jsonl"
image_lookup_path = "/root/user/LLaMA-Factory/data/t+c+i_17.json"
output_json_path = "/root/user/LLaMA-Factory/data/17_explain_dialog.json"

# 加载图像查找文件（prompt -> image_path）
with open(image_lookup_path, "r") as f:
    image_data = json.load(f)

prompt_to_image = {
    item["messages"][0]["content"].strip(): item["images"][0]
    for item in image_data if item["images"]
}

# 处理 explain.jsonl 数据
output_data = []

with open(explain_jsonl_path, "r") as f:
    for line in tqdm(f):
        item = json.loads(line)
        prompt = item["prompt"].replace("USER: ", "").replace("ASSISTANT:", "").strip()
        predict = item["predict"]
        label = item["label"]
        reflection = item["Reflection"]
        improvement = item["Improvement"]

        # # 查找图像路径
        def normalize_prompt(text):
            # 删除所有空白字符（空格、换行、制表符等）和标点空格差异
            return re.sub(r"\s+", "", text).strip()

        # 建索引：标准化 prompt -> image path
        prompt_to_image = {
            normalize_prompt(item["messages"][0]["content"]): item["images"][0]
            for item in image_data if item["images"]
        }

        # 查找图像路径
        normalized_prompt = normalize_prompt(prompt)
        image_path = prompt_to_image.get(normalized_prompt, "")



        # 构建多轮对话 messages
        messages = [
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "assistant",
                "content": predict
            },
            {
                "role": "user",
                "content": (
                    f"Your answer was incorrect. The correct sentiment is **{label}**.\n\n"
                    f"Reflection: {reflection}\n\n"
                    f"Improvement: {improvement}\n\n"
                    f"Now, based on the image, text, image description, and the above feedback, "
                    f"what should the corrected sentiment be? Respond with only one word: Positive, Neutral, or Negative."
                )
            },
            {
                "role": "assistant",
                "content": label
            }
        ]

        output_data.append({
            "messages": messages,
            "images": [image_path]
        })

# 保存为标准 JSON 文件（数组形式）
with open(output_json_path, "w") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"✅ 转换完成，共保存 {len(output_data)} 条样本，输出路径：  {output_json_path}")

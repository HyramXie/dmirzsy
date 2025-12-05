import json
import os

# 设置图片根目录（按实际路径修改）
image_root = "2017/twitter2017_images"

# 读取原始 JSON 数据
with open("/root/user/datasets/twitter2017/train+image_context.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

converted = []

for item in raw_data:
    text = (item.get("text") or "").strip()
    image_id = (item.get("image_id") or "").strip()
    image_path = os.path.join(image_root, image_id)
    image_context = (item.get("image_context") or "").strip()

    # 构造 prompt
    prompt = (
        f"<image> Given the following social media post: \"{text}\", describe the visual scene in the image. "
        f"Focus on the setting, people, objects, and their emotional or physical state. "
        f"Your description should be precise, human-like, and informative."
    )

    converted.append({
        "messages": [
            {
                "content": prompt,
                "role": "user"
            },
            {
                "content": image_context,
                "role": "assistant"
            }
            # assistant 回复待模型生成
        ],
        "images": [
            image_path
        ]
    })

# 保存生成的数据
with open("t+i-c_train_17.json", "w", encoding="utf-8") as f:
    json.dump(converted, f, indent=2, ensure_ascii=False)

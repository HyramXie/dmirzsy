import json
import os

# 路径前缀（根据你图像文件夹实际位置调整）
image_root = "twitter2015_images"

# 读取原始数据
with open("/root/user/datasets/twitter2015/test+image_context.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

converted = []

for item in raw_data:
    text = (item.get("text") or "").strip()
    aspect = (item.get("aspect") or "").strip()
    sentiment = (item.get("sentiment") or "").strip()
    image_id = (item.get("image_id") or "").strip()
    image_context = (item.get("image_context") or "").strip()

    # 构造图像路径
    image_path = os.path.join(image_root, image_id)

    # 构造用户输入提示
    prompt = (
        f"<image> Based on the image and the following visual context: '{image_context}', "
        f"generate a short and realistic social media post that reflects the sentiment or intent conveyed by the image. "
        f"The post should sound natural, emotionally aligned, and human-like."
    )

    # 构造多模态对话格式
    converted.append({
        "messages": [
            {
                "content": prompt,
                "role": "user"
            },
            {
                "content": text,
                "role": "assistant"
            }
        ],
        "images": [
            image_path
        ]
    })

# 保存结果
with open("i+c->t_test_15.json", "w", encoding="utf-8") as f:
    json.dump(converted, f, indent=2, ensure_ascii=False)

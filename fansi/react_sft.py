import json
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForVision2Seq

# 模型路径（根据你本地模型实际路径设置）
model_path = "/public/home/byxu_jsjxy/ywl/pretrained/Qwen/Qwen2.5-VL-32B-Instruct"

# 加载模型、tokenizer、processor
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
).eval()

# 读取推理结果（merged_dataset_15.json）
input_path = "/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/merged_dataset_15.json"
with open(input_path, "r") as f:
    dataset = json.load(f)

output_results = []

for example in tqdm(dataset):
    try:
        text = example["text"]
        aspect = example["aspect"]
        gold_sentiment = example["sentiment"]
        pred_sentiment = example["predict"]
        image_id = example["image_id"]
        image_context = example["image_context"]

        # 图像路径
        image_path = f"/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/twitter2015_images/{image_id}"
        image = Image.open(image_path).convert("RGB")

        # 构造 prompt
        prompt = f"""Please evaluate whether the following predicted sentiment is correct based on the image, text, and image description.

Text: "{text}"
Image description: "{image_context}"
Aspect: <target>{aspect}</target>
Predicted sentiment: {pred_sentiment}

Do you agree with the predicted sentiment? If yes, respond in this format:
`{pred_sentiment}: <brief explanation>`
If not, provide the corrected sentiment and a brief explanation in this format:
`<corrected sentiment>: <brief explanation>`
"""


        # 构造多模态输入（文本+图像）
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # 构造符合 Qwen 模板的文本输入
        text_input = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        # 构造输入
        inputs = processor(text=text_input, images=[image], return_tensors="pt").to(model.device)

        # 推理
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)

        # 解码
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if "assistant" in decoded:
            reflection = decoded.split("assistant")[-1].strip()
        else:
            reflection = decoded

        # 提取情感标签
        reflection_sentiment = None
        for label in ["Positive", "Neutral", "Negative"]:
            if reflection.startswith(label):
                reflection_sentiment = label
                break

        # 判断是否正确
        is_correct = (reflection_sentiment == pred_sentiment)

        # 写入结果
        output_results.append({
            "text": text,
            "aspect": aspect,
            "sentiment": gold_sentiment,
            "predict": pred_sentiment,
            "Reflection Results": reflection,
            "is_correct": is_correct,
            "image_id": image_id,
            "image_context": image_context
        })

    except Exception as e:
        print(f"Error processing {example.get('image_id', '')}: {e}")
        continue

# 保存反思结果
output_path = "/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/merged_dataset_15_reflected.json"
with open(output_path, "w") as f:
    json.dump(output_results, f, indent=2)

print(f"反思推理完成，已保存至：{output_path}")

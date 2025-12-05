from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
from tqdm import tqdm
import torch
import json

model_path = "/public/home/byxu_jsjxy/ywl/pretrained/google/gemma-3-27b-it"

# 加载模型与处理器
model = Gemma3ForConditionalGeneration.from_pretrained(model_path, device_map="auto").eval()
processor = AutoProcessor.from_pretrained(model_path)

# 加载数据集
with open("/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/t+c+i_15.json", "r") as f:
    dataset = json.load(f)

results = []

for example in tqdm(dataset):
    try:
        # 图像
        image_path = example["images"][0]
        image = Image.open(image_path).convert("RGB")

        # 文本 Prompt
        user_msg = next(msg for msg in example["messages"] if msg["role"] == "user")
        user_text = user_msg["content"].replace("<image>", "").strip()

        # 构造聊天模板
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_text}]}
        ]

        # 构造输入
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=50)
            output_ids = generation[0][input_len:]

        decoded = processor.decode(output_ids, skip_special_tokens=True).strip()

        results.append({
            "image": image_path,
            "prompt": user_text,
            "model_output": decoded
        })

    except Exception as e:
        print(f"Error: {e}")
        continue

# 保存结果
with open("/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/gemma3_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("推理完成，结果保存在 gemma3_results.json")

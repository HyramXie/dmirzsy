
# DATASET_PATH = "/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/t+c+i_15.json"            # 你的输入数据集路径
# IMAGE_BASE_DIR = "/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/twitter2015_images"          # 图像文件夹路径（相对于数据集中的 image 字段）
# MODEL_PATH = "/public/home/byxu_jsjxy/ywl/pretrained/Qwen/Qwen2.5-VL-32B-Instruct"           # 本地 Qwen2-VL-32B 模型路径
# OUTPUT_PATH = "/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/w0.json"               # 输出结果路径
# DEVICE = "cuda:0" 
import json
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForVision2Seq

# 模型路径（修改为你自己的路径）
model_path = "/public/home/byxu_jsjxy/ywl/pretrained/Qwen/Qwen2.5-VL-32B-Instruct"

# 加载模型
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,  # 如果你的 GPU 支持，可以用 float16 或 bfloat16
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# 加载数据集
with open("/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/re_15_1.json", "r") as f:
    dataset = json.load(f)

results = []

for example in tqdm(dataset):
    try:
        # 图像路径
        image_path = example["images"][0]
        image = Image.open(image_path).convert("RGB")

        # 获取 user prompt 内容
        user_msg = next(msg for msg in example["messages"] if msg["role"] == "user")
        user_text = user_msg["content"].replace("<image>", "").strip()  # 删除 <image> 占位符

        # 构造符合 Qwen 格式的聊天内容
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text}
                ]
            }
        ]


        # 使用 tokenizer 构造模板输入（自动插入 <img> token）
        prompt = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
)

        # processor 正确用法：text=..., images=[...]
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to(model.device)


        # 推理
        # 推理
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10)

        # 解码为字符串
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # 提取 assistant 的最后回复
        # 有些模型会输出形如：system\n...\nuser\n...\nassistant\nPositive
        # 所以我们提取最后一次出现 "assistant" 之后的部分
        if "assistant" in decoded:
            response = decoded.split("assistant")[-1].strip()
        else:
            response = decoded  # fallback
        

        results.append({
            "image": image_path,
            "prompt": user_text,
            "model_output": response
        })

    except Exception as e:
        print(f"Error processing example: {e}")
        continue

# 保存结果到 JSON 文件
with open("/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/w2.json", "w") as f:
    json.dump(results, f, indent=2)

print("推理完成，结果保存在 /public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/w2.json")

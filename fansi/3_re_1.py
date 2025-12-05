import os
import json
import gc
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForVision2Seq

# 防碎片设置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 模型路径（确保和之前一致）
model_path = "/public/home/byxu_jsjxy/ywl/pretrained/google/gemma-3-27b-it"

# 加载 tokenizer 和 processor
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# 加载模型（保持 float16 和 device_map）
model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# 输入文件：之前提取出的不一致样本
inconsistent_file = "/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/2-inconsistent_predictions.json"
output_reflect_path = "/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/3-reflected_judgments.json"

# 推理参数
max_new_tokens = 128
num_return_sequences = 5  # 再生成5条反思回答
reflected_results = []

# 加载不一致样本
with open(inconsistent_file, "r", encoding="utf-8") as f:
    inconsistent_samples = json.load(f)

print("🔄 开始进行反思式推理（Reflection over inconsistent predictions）...")

for item in tqdm(inconsistent_samples):
    try:
        original_data = item["original"]
        image_path = original_data["images"][0]
        image = Image.open(image_path).convert("RGB")

        # 提取原始 user 内容
        user_msg = next(msg for msg in original_data["messages"] if msg["role"] == "user")
        user_content = user_msg["content"]

        # 提取第一轮的5个 model_outputs（已有的生成结果）
        first_round_outputs = original_data.get("model_outputs", [])
        votes = item["extracted_sentiments"]  # 如 ["Negative", "Neutral", ...]

        # 构建反思 prompt
        reflection_prompt = (
            f"Original task:\n{user_content}\n\n"
            f"Based on the image, text, and description above, "
            f"you previously generated 5 responses with mixed sentiment judgments:\n"
        )
        for i, (out, vote) in enumerate(zip(first_round_outputs, votes)):
            reflection_prompt += f"{i+1}. {out} ({vote})\n"
        
        reflection_prompt += (
            "\nNow, please re-evaluate all these opinions and provide a final judgment.\n"
            "Answer in exactly this format: \"Final Sentiment: [Positive/Neutral/Negative]. Reason: [brief explanation].\"\n"
            "Be concise and focus on evidence from the image, text, and reasoning."
        )

        # 构造对话
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": reflection_prompt}
                ]
            }
        ]

        # 使用 tokenizer 构建模板
        full_prompt = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        # 处理输入
        inputs = processor(text=full_prompt, images=[image], return_tensors="pt").to(model.device, torch.float16)

        # 生成多条反思回答
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                num_return_sequences=num_return_sequences,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # 清洗输出：提取 assistant 回复
        reflected_responses = []
        for output in decoded_outputs:
            if "assistant" in output:
                response = output.split("assistant")[-1].strip()
            else:
                response = output.strip()

            # 简单清洗格式
            if not response.startswith("Final Sentiment:"):
                response = "Final Sentiment: " + response  # 尽量补全
            reflected_responses.append(response)

        # 保存结果（保留原始分歧信息 + 新增反思输出）
        reflected_results.append({
            "original_inconsistent_sample": original_data,
            "first_round_votes": votes,
            "first_round_outputs": first_round_outputs,
            "distribution": item["distribution"],
            "reflected_outputs": reflected_responses  # 第二轮反思结果
        })

    except Exception as e:
        print(f"Error during reflection for {original_data['images'][0]}: {e}")
        reflected_results.append({
            "error": str(e),
            "image": original_data.get("images", ["unknown"])[0]
        })
    finally:
        torch.cuda.empty_cache()
        gc.collect()

# 保存反思结果
with open(output_reflect_path, "w", encoding="utf-8") as f:
    json.dump(reflected_results, f, indent=2, ensure_ascii=False)

print(f"✅ 反思完成！共处理 {len(reflected_results)} 个不一致样本。")
print(f"📁 结果已保存至: {output_reflect_path}")
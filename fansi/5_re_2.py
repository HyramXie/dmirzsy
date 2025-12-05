import os
import json
import gc
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForVision2Seq

# 防碎片设置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 模型路径
model_path = "/public/home/byxu_jsjxy/ywl/pretrained/Qwen/Qwen2.5-VL-32B-Instruct"

# 加载组件
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# 输入文件：第二轮反思后仍不一致的样本
input_file = "/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/4-reflected_inconsistent_again.json"
# 输出文件：第三轮最终反思结果
output_file = "/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/5-final_reflection_round.json"

# 推理参数
max_new_tokens = 128
num_return_sequences = 5
final_results = []

# 加载数据
with open(input_file, "r", encoding="utf-8") as f:
    samples = json.load(f)

print("🧠 开始第三轮元反思（Meta-Reflection over second-round judgments）...")

for item in tqdm(samples):
    try:
        # 原始信息
        original_sample = item["original_inconsistent_sample"]
        image_path = original_sample["images"][0]
        image = Image.open(image_path).convert("RGB")

        # 原始用户问题
        user_msg = next(msg for msg in original_sample["messages"] if msg["role"] == "user")
        user_content = user_msg["content"]

        # 第二轮反思输出（即模型自己之前的“反思”）
        previous_reflections = item.get("reflected_outputs", [])

        # 构造第三轮 prompt
        meta_prompt = (
            f"### Task Recap:\n{user_content}\n\n"
            f"### Previous Round of Self-Reflection:\n"
            "In your own previous reflection, you generated the following 5 analyses:\n"
        )
        for i, out in enumerate(previous_reflections):
            meta_prompt += f"{i+1}. {out}\n"
        
        meta_prompt += (
            "\n### Instruction:\n"
            "Now, act as a meta-analyst. Review all 5 of your prior reflection responses critically.\n"
            "Identify patterns, contradictions, and strongest evidence.\n"
            "Then provide a final, well-reasoned sentiment judgment.\n\n"
            "Answer in this format:\n"
            "\"Final Decision: [Positive/Neutral/Negative]. Rationale: [Concise, evidence-based explanation].\"\n"
            "Do NOT just repeat one of the above. Synthesize and evaluate them."
        )

        # 构建对话
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": meta_prompt}
                ]
            }
        ]

        # 应用 chat template
        full_prompt = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        # 处理输入
        inputs = processor(text=full_prompt, images=[image], return_tensors="pt").to(model.device, torch.float16)

        # 生成 5 条最终综合判断
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

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # 清洗输出
        final_outputs = []
        for output in decoded:
            if "assistant" in output:
                response = output.split("assistant")[-1].strip()
            else:
                response = output.strip()
            final_outputs.append(response)

        # 保存结果
        final_results.append({
            "original_inconsistent_sample": original_sample,
            "first_round_votes": item["first_round_votes"],
            "first_round_outputs": item["first_round_outputs"],
            "reflected_outputs": previous_reflections,
            "reflected_distribution": item["reflected_distribution"],
            "final_meta_reflection": final_outputs  # 第三轮最终输出
        })

    except Exception as e:
        print(f"Error in meta-reflection for {original_sample['images'][0]}: {e}")
        final_results.append({
            "error": str(e),
            "image": original_sample["images"][0]
        })
    finally:
        torch.cuda.empty_cache()
        gc.collect()

# 保存最终结果
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=2, ensure_ascii=False)

print(f"✅ 第三轮元反思完成！共处理 {len(final_results)} 个高争议样本。")
print(f"📁 结果已保存至: {output_file}")
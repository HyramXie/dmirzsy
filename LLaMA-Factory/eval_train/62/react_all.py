import json
import os
from openai import OpenAI
from tqdm import tqdm

# 初始化 DeepSeek API
client = OpenAI(
    api_key="sk-100b432f23414ba8a71a21edd60f7a99",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 或你的代理地址
)

input_file = "/root/user/zsy/LLaMA-Factory/eval_train/62/qwen2.5vl-3b/15.jsonl"
output_file = "/root/user/zsy/LLaMA-Factory/eval_train/62/qwen2.5vl-3b/15_explain_all.jsonl"

# 获取已处理的 prompt 列表（用于断点续跑）
processed_prompts = set()
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                processed_prompts.add(data['prompt'])
            except:
                continue

# 加载全部输入数据
with open(input_file, 'r', encoding='utf-8') as f:
    all_data = [json.loads(line) for line in f]

for entry in tqdm(all_data):
    prompt = entry['prompt']
    predict = entry['predict']
    label = entry['label']

    # 跳过已经处理过的样本
    if prompt in processed_prompts:
        continue

    # 构建提问内容（提示词）
    system_prompt = "You are an expert at diagnosing and improving LLM predictions."

    user_prompt = f"""The following is a model prediction. Analyze it against the ground truth.

Text Prompt: {prompt}
Model Prediction: {predict}
Ground Truth Label: {label}

Please provide:
1. Reflection: If the prediction is incorrect, where did the reasoning go wrong and why? If the prediction is correct, explain why the reasoning is sound and what evidence supports it.
2. Improvement: How can the model improve its reasoning to consistently reach the correct answer (even if this case is already correct)?

Respond in this format:
Reflection: ...
Improvement: ...
"""

    try:
        # 调用 DeepSeek API
        response = client.chat.completions.create(
            model="deepseek-v3",  # 替换为你实际用的 deepseek-v3 模型名
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=256
        )

        reply = response.choices[0].message.content.strip()

        # 解析模型输出（建议格式清晰，防止出错）
        reflection = ""
        improvement = ""
        for line in reply.splitlines():
            if line.startswith("Reflection:"):
                reflection = line.replace("Reflection:", "").strip()
            elif line.startswith("Improvement:"):
                improvement = line.replace("Improvement:", "").strip()

        # 构造输出结构
        result = {
            "prompt": prompt,
            "predict": predict,
            "label": label,
            "Reflection": reflection,
            "Improvement": improvement
        }

        # 追加保存
        with open(output_file, 'a', encoding='utf-8') as out_f:
            out_f.write(json.dumps(result, ensure_ascii=False) + '\n')

        print(f"Processed: {prompt[:50]}...")

    except Exception as e:
        print(f"Error processing prompt: {prompt[:50]}... Error: {e}")
        continue

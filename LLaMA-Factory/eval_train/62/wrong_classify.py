import json
import os
import re
from openai import OpenAI
from tqdm import tqdm

# 初始化 DeepSeek API
client = OpenAI(
    api_key="sk-100b432f23414ba8a71a21edd60f7a99",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 注意无多余空格
)

input_file = "/home/cbf00006701/zsy/LLaMA-Factory/eval_train/qwen2.5vl-3b/inconsistent_predictions_17.jsonl"
output_file = "/home/cbf00006701/zsy/LLaMA-Factory/eval_train/qwen2.5vl-3b/17_wrong_classify.jsonl"

# 加载已处理的 prompt，支持断点续跑
processed_prompts = set()
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                processed_prompts.add(data['prompt'])
            except:
                continue

# 加载所有待处理数据
with open(input_file, 'r', encoding='utf-8') as f:
    all_data = [json.loads(line) for line in f]

# 错误类型定义（简洁明了）
ERROR_CATEGORIES = """
(A) Over-reliance on a single modality (e.g., only focusing on image background or text tone).
(B) Failure to distinguish the target aspect $A$ from the overall context.
(C) Misunderstanding rhetorical expressions such as sarcasm, metaphor, or irony.
"""

for entry in tqdm(all_data):
    prompt = entry['prompt']
    predict = entry['predict']
    label = entry['label']

    # 跳过已处理样本
    if prompt in processed_prompts:
        continue

    # 构造 prompt
    user_message = f"""
A multimodal model made a prediction error.

Input:
{prompt}

Model Prediction: {predict}
True Label: {label}

The possible error types are:
{ERROR_CATEGORIES}

Analyze the error and output ONLY one character: A, B, or C.
Do not explain. Just output the letter.
"""

    try:
        response = client.chat.completions.create(
            model="deepseek-v3",  # 或 "deepseek-chat-v1"，根据百炼平台实际名称
            messages=[
                {"role": "system", "content": "You are an expert in diagnosing multimodal sentiment model errors."},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,      # 降低随机性
            max_tokens=10,        # 只需一个字母
            stop=None
        )

        raw_output = response.choices[0].message.content.strip()

        # 使用正则提取 A/B/C
        match = re.search(r'\b([ABC])\b', raw_output, re.IGNORECASE)
        error_type = match.group(1).upper() if match else "C"  # 默认 C

        # 构建结果
        result = {
            "prompt": prompt,
            "predict": predict,
            "label": label,
            "ErrorType": error_type
        }

        # 追加写入文件
        with open(output_file, 'a', encoding='utf-8') as out_f:
            out_f.write(json.dumps(result, ensure_ascii=False) + '\n')

    except Exception as e:
        print(f"Error processing prompt: {prompt[:50]}... | Error: {e}")
        # 出错时写入默认值可选，这里跳过
        continue
# import json
# import os
# from openai import OpenAI
# from tqdm import tqdm

# # 初始化 DeepSeek API
# client = OpenAI(
#     api_key="sk-100b432f23414ba8a71a21edd60f7a99",
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 或你的代理地址
# )

# input_file = "/root/user/LLaMA-Factory/eval_train/62/qwen2.5vl-7b/inconsistent_predictions_17.jsonl"
# output_file = "/root/user/LLaMA-Factory/eval_train/62/qwen2.5vl-7b/17_explain.jsonl"

# # 获取已处理的 prompt 列表（用于断点续跑）
# processed_prompts = set()
# if os.path.exists(output_file):
#     with open(output_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             try:
#                 data = json.loads(line)
#                 processed_prompts.add(data['prompt'])
#             except:
#                 continue

# # 加载全部输入数据
# with open(input_file, 'r', encoding='utf-8') as f:
#     all_data = [json.loads(line) for line in f]

# for entry in tqdm(all_data):
#     prompt = entry['prompt']
#     predict = entry['predict']
#     label = entry['label']

#     # 跳过已经处理过的样本
#     if prompt in processed_prompts:
#         continue

#     # 构建提问内容（提示词）
#     system_prompt = "You are an expert at diagnosing and improving LLM predictions."

#     user_prompt = f"""The following is a mistaken prediction by a model:
# Text Prompt: {prompt}
# Model Prediction: {predict}
# Ground Truth Label: {label}

# Please provide:
# 1. Reflection: Where did the reasoning go wrong, and why?
# 2. Improvement: How can the model improve its reasoning to get the correct answer?

# Respond in this format:
# Reflection: ...
# Improvement: ...
# """

#     try:
#         # 调用 DeepSeek API
#         response = client.chat.completions.create(
#             model="deepseek-v3",  # 替换为你实际用的 deepseek-v3 模型名
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt}
#             ],
#             temperature=0.7,
#             max_tokens=256
#         )

#         reply = response.choices[0].message.content.strip()

#         # 解析模型输出（建议格式清晰，防止出错）
#         reflection = ""
#         improvement = ""
#         for line in reply.splitlines():
#             if line.startswith("Reflection:"):
#                 reflection = line.replace("Reflection:", "").strip()
#             elif line.startswith("Improvement:"):
#                 improvement = line.replace("Improvement:", "").strip()

#         # 构造输出结构
#         result = {
#             "prompt": prompt,
#             "predict": predict,
#             "label": label,
#             "Reflection": reflection,
#             "Improvement": improvement
#         }

#         # 追加保存
#         with open(output_file, 'a', encoding='utf-8') as out_f:
#             out_f.write(json.dumps(result, ensure_ascii=False) + '\n')

#         print(f"Processed: {prompt[:50]}...")

#     except Exception as e:
#         print(f"Error processing prompt: {prompt[:50]}... Error: {e}")
#         continue

import json
import os
from openai import OpenAI
from tqdm import tqdm
import time

# 初始化 DeepSeek API
client = OpenAI(
    api_key="sk-100b432f23414ba8a71a21edd60f7a99",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 或你的代理地址
)

input_file = "/root/user/zsy/LLaMA-Factory/eval_train/62/qwen3vl-4b/inconsistent_predictions_17.jsonl"
output_file = "/root/user/zsy/LLaMA-Factory/eval_train/62/qwen3vl-4b/17_explain.jsonl"

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

# 统计指标
total_time = 0
total_prompt_tokens = 0
total_completion_tokens = 0
total_tokens = 0
processed_count = 0
error_count = 0

start_time = time.time()

for entry in tqdm(all_data, desc="Processing entries"):
    prompt = entry['prompt']
    predict = entry['predict']
    label = entry['label']

    # 跳过已经处理过的样本
    if prompt in processed_prompts:
        continue

    # 构建提问内容（提示词）
    system_prompt = "You are an expert at diagnosing and improving LLM predictions."

    user_prompt = f"""The following is a mistaken prediction by a model:
Text Prompt: {prompt}
Model Prediction: {predict}
Ground Truth Label: {label}

Please provide:
1. Reflection: Where did the reasoning go wrong, and why?
2. Improvement: How can the model improve its reasoning to get the correct answer?

Respond in this format:
Reflection: ...
Improvement: ...
"""

    try:
        # 记录开始时间
        request_start_time = time.time()
        
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
        
        request_end_time = time.time()
        
        # 计算处理时间
        request_time = request_end_time - request_start_time
        total_time += request_time

        # 累计token统计
        usage = response.usage
        total_prompt_tokens += usage.prompt_tokens if usage else 0
        total_completion_tokens += usage.completion_tokens if usage else 0
        total_tokens += (usage.prompt_tokens + usage.completion_tokens) if usage else 0

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

        processed_count += 1
        print(f"Processed: {prompt[:50]}...")

    except Exception as e:
        error_count += 1
        print(f"Error processing prompt: {prompt[:50]}... Error: {e}")
        continue

end_time = time.time()
overall_time = end_time - start_time

# 计算未处理的原始数据量
total_input_count = len(all_data)
skipped_count = total_input_count - processed_count - error_count

# 输出效率指标
print(f"\n📊 Processing Summary:")
print(f"📁 Total input entries: {total_input_count}")
print(f"⏭️  Skipped (already processed): {skipped_count}")
print(f"✅ Successfully processed: {processed_count}")
print(f"❌ Errors occurred: {error_count}")
print(f"🎯 Actually processed in this run: {processed_count}")
print(f"⏱️  Total processing time: {total_time:.2f}s")
print(f"⏱️  Overall time (including setup): {overall_time:.2f}s")
if processed_count > 0:
    print(f"⚡ Average time per entry: {total_time/processed_count:.2f}s")
    print(f"📈 Average prompt tokens per entry: {total_prompt_tokens/processed_count:.2f}")
    print(f"📈 Average completion tokens per entry: {total_completion_tokens/processed_count:.2f}")
    print(f"📈 Average total tokens per entry: {total_tokens/processed_count:.2f}")
print(f"📝 Total prompt tokens: {total_prompt_tokens}")
print(f"📝 Total completion tokens: {total_completion_tokens}")
print(f"📝 Total tokens: {total_tokens}")
print(f"✅ Results saved to: {output_file}")
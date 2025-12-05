# import json
# import re
# from tqdm import tqdm

# # 输入文件：第二轮反思的完整输入（即 inconsistent_predictions.json）
# input_file = "/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/2-inconsistent_predictions.json"
# # 输出文件：包含每条的状态分析
# diagnosis_output = "/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/reflection_diagnosis.json"

# # 反思结果文件（第二轮输出）
# reflected_file = "/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/3-reflected_judgments.json"

# # 支持的情感标签
# SENTIMENTS = {"Positive", "Neutral", "Negative"}

# def extract_sentiment_from_reflected(text):
#     match = re.search(r"Final\s+Sentiment\s*:\s*([a-zA-Z]+)", text, re.IGNORECASE)
#     if match:
#         word = match.group(1).capitalize()
#         if word in SENTIMENTS:
#             return word
#     return None

# # 加载原始不一致样本（应为 157 条）
# with open(input_file, "r", encoding="utf-8") as f:
#     original_inconsistent = json.load(f)
# print(f"📁 原始不一致样本总数: {len(original_inconsistent)}")

# # 加载反思后结果（可能少于 157）
# try:
#     with open(reflected_file, "r", encoding="utf-8") as f:
#         reflected_results = json.load(f)
#     print(f"📁 反思后结果数量: {len(reflected_results)}")
# except FileNotFoundError:
#     reflected_results = []
#     print("❌ 未找到反思结果文件")

# # 建立映射：用 image 路径作为唯一 key
# reflected_dict = {}
# for item in reflected_results:
#     try:
#         img_path = item["original_inconsistent_sample"]["images"][0]
#         reflected_dict[img_path] = item
#     except:
#         continue

# # 存储每条的状态
# diagnosis = []

# missing_count = 0
# for item in tqdm(original_inconsistent, desc="Analyzing"):
#     try:
#         img_path = item["original"]["images"][0]
#     except:
#         img_path = "unknown_image"

#     status = {
#         "image": img_path,
#         "status": None,
#         "note": ""
#     }

#     if img_path not in reflected_dict:
#         status["status"] = "MISSING"
#         status["note"] = "No reflection result generated"
#         missing_count += 1
#         diagnosis.append(status)
#         continue

#     reflected_item = reflected_dict[img_path]
#     reflected_outputs = reflected_item.get("reflected_outputs", [])

#     if len(reflected_outputs) != 5:
#         status["status"] = "INVALID_OUTPUT_COUNT"
#         status["note"] = f"Generated {len(reflected_outputs)} responses, not 5"
#         diagnosis.append(status)
#         continue

#     # 提取情感
#     sentiments = [extract_sentiment_from_reflected(out) for out in reflected_outputs]
#     valid_sentiments = [s for s in sentiments if s is not None]

#     if len(valid_sentiments) != 5:
#         status["status"] = "PARSING_FAILED"
#         failed = 5 - len(valid_sentiments)
#         status["note"] = f"Failed to parse {failed}/5 sentiments"
#         diagnosis.append(status)
#         continue

#     if len(set(valid_sentiments)) == 1:
#         status["status"] = "CONSISTENT"
#         status["final_sentiment"] = valid_sentiments[0]
#     else:
#         status["status"] = "INCONSISTENT"
#         status["distribution"] = dict(zip(SENTIMENTS, [valid_sentiments.count(s) for s in SENTIMENTS]))

#     diagnosis.append(status)

# # 保存诊断结果
# with open(diagnosis_output, "w", encoding="utf-8") as f:
#     json.dump(diagnosis, f, indent=2, ensure_ascii=False)

# # 统计
# from collections import Counter
# stats = Counter(d["status"] for d in diagnosis)

# print("\n📊 最终统计:")
# for k, v in stats.items():
#     print(f"  {k}: {v}")

# print(f"\n🔍 缺失的 8 条数据很可能是以下情况之一:")
# if "MISSING" in stats:
#     print(f"  • {stats['MISSING']} 条：未生成反思结果（可能因 CUDA OOM、超时、程序中断）")
# if "INVALID_OUTPUT_COUNT" in stats:
#     print(f"  • {stats['INVALID_OUTPUT_COUNT']} 条：生成数量不为5")
# if "PARSING_FAILED" in stats:
#     print(f"  • {stats['PARSING_FAILED']} 条：无法解析情感标签")

# print(f"\n📁 详细诊断已保存至: {diagnosis_output}")


# import json
# from collections import Counter

# # 假设你的数据保存在一个名为 data.json 的文件中
# with open('/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/4-hebing.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)

# import json
# import re

# def extract_sentiment(text):
#     """从 model_output 中提取第一个出现的情感极性（Positive/Neutral/Negative），忽略大小写"""
#     # 按顺序搜索这三个词的首次出现
#     pattern = r'\b(positive|neutral|negative)\b'
#     match = re.search(pattern, text, re.IGNORECASE)
#     if match:
#         return match.group(1).lower()  # 返回小写形式便于比较
#     return None  # 未找到情感词

# consistent_count = 0
# inconsistent_count = 0

# for item in data:
#     model_outputs = item.get("model_outputs", [])
#     if len(model_outputs) < 5:
#         # 如果 model_outputs 不足5个，视为不一致（或可根据需求调整）
#         inconsistent_count += 1
#         continue

#     # 提取每个 model_output 的第一个情感极性
#     sentiments = []
#     for output in model_outputs:
#         sent = extract_sentiment(output)
#         sentiments.append(sent)

#     # 检查是否所有提取出的情感都相同且有效
#     if None in sentiments:
#         inconsistent_count += 1
#     elif len(set(sentiments)) == 1:  # 所有元素相同
#         consistent_count += 1
#     else:
#         inconsistent_count += 1

# print(f"五个极性预测一致的数据条数: {consistent_count}")
# print(f"五个极性预测不一致的数据条数: {inconsistent_count}")

import json
import re

input_file = "/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/4-hebing.json"

SENTIMENTS = {"Positive", "Neutral", "Negative"}

def extract_sentiment(text):
    match = re.search(r'\b(Positive|Neutral|Negative)\b', text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()  # 统一标准化
    return None

# 调试计数器
total_items = 0
valid_5_outputs = 0
all_5_extracted = 0
all_5_consistent = 0
four_consistent_one_none = 0
four_consistent_one_diff = 0

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

print("🔍 开始调试分析...")

for item in data:
    total_items += 1
    model_outputs = item.get("model_outputs", [])
    
    if len(model_outputs) < 5:
        continue
    valid_5_outputs += 1

    # 提取情感
    sentiments = [extract_sentiment(out) for out in model_outputs]
    valid_sentiments = [s for s in sentiments if s in SENTIMENTS]  # 确保在集合中

    if len(valid_sentiments) == 5:
        all_5_extracted += 1
        if len(set(valid_sentiments)) == 1:
            all_5_consistent += 1
        else:
            # 5 个都有效但不一致
            pass
    elif len(valid_sentiments) == 4:
        if len(set(valid_sentiments)) == 1:
            # 4 个一致，1 个失败
            four_consistent_one_none += 1
        else:
            # 4 个中有不同
            four_consistent_one_diff += 1
    

    # 在调试循环中加入
    if len(valid_sentiments) == 4 and len(set(valid_sentiments)) == 1:
        print(f"\n🟡 发现 4 个一致 + 1 个提取失败的样本:")
        print(f"Image: {item['images'][0]}")
        print(f"提取结果: {sentiments}")
        for i, out in enumerate(model_outputs):
            print(f"Output {i+1}:\n{out.strip()}")
            match = re.search(r'\b(Positive|Neutral|Negative)\b', out, re.IGNORECASE)
            print(f"  → 提取: {match.group(1).capitalize() if match else 'None'}")
        print("-" * 60)

print(f"📊 总样本数: {total_items}")
print(f"✅ model_outputs 长度 ≥5 的样本数: {valid_5_outputs}")
print(f"✅ 5 个情感全部成功提取的样本数: {all_5_extracted}")
print(f"✅ 5 个提取成功且情感一致的样本数: {all_5_consistent}")
print(f"🟡 4 个一致 + 1 个提取失败的样本数: {four_consistent_one_none}")
print(f"🔴 4 个一致 + 1 个不同情感的样本数: {four_consistent_one_diff}")
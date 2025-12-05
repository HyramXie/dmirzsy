import json
import re
import os

# 数据集路径
input_path = "/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2017/merged_dataset_17_reflected.json"
output_path = "/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2017/merged_dataset_17_reflected_with_sentiment.json"

# 定义情感关键词（按优先级顺序，取第一个匹配的）
sentiment_keywords = ['positive', 'neutral', 'negative']

# 编译正则表达式，匹配这些词（忽略大小写和边界）
pattern = re.compile(r'\b(' + '|'.join(sentiment_keywords) + r')\b', re.IGNORECASE)

def extract_sentiment(text):
    """从文本中提取第一个匹配的情感极性词"""
    match = pattern.search(text)
    if match:
        return match.group(1).lower()  # 返回小写形式
    return None  # 如果没找到，返回 None

# 读取原始数据
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 遍历每条数据，提取 reflect_sentiment
for item in data:
    reflection_text = item.get("Reflection Results", "")
    sentiment = extract_sentiment(reflection_text)
    item["reflect_sentiment"] = sentiment

# 保存新数据
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"处理完成，已保存到 {output_path}")


# from sklearn.metrics import accuracy_score, f1_score

# with open('/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2017/merged_dataset_17_reflected_with_sentiment.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)

# def clean_and_check_sentiment(s, default='neutral', warn=True, context=""):
#     valid_labels = {'positive', 'negative', 'neutral'}
    
#     if s is None:
#         if warn:
#             print(f"⚠️  {context} -> reflect_sentiment 是 None")
#         return default
    
#     if not isinstance(s, str):
#         if warn:
#             print(f"⚠️  {context} -> reflect_sentiment 类型错误: {type(s)}, 值为 {s}")
#         return default
    
#     cleaned = s.lower().strip()
    
#     if cleaned not in valid_labels:
#         if warn:
#             print(f"⚠️  {context} -> reflect_sentiment 无效值: '{s}' (清洗后 '{cleaned}')")
#         return default
    
#     return cleaned

# # 提取标签并检查问题
# y_true = []
# y_pred = []

# for i, item in enumerate(data):
#     truth = item["sentiment"]
#     pred = item.get("reflect_sentiment", None)  # 使用 get 更安全

#     # 上下文信息，方便定位
#     context_info = f"索引 {i}, image_id: {item.get('image_id', '未知')}"

#     # 清洗并检查
#     label_true = clean_and_check_sentiment(truth, default='', warn=False, context=context_info)
#     label_pred = clean_and_check_sentiment(pred, default='neutral', warn=True, context=context_info)

#     y_true.append(label_true)
#     y_pred.append(label_pred)

# # 计算指标
# acc = accuracy_score(y_true, y_pred)
# f1_macro = f1_score(y_true, y_pred, average='macro')

# print("\n✅ 分类指标：")
# print(f"Accuracy: {acc:.4f}")
# print(f"Macro-F1: {f1_macro:.4f}")
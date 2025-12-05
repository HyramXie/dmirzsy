import json
from sklearn.metrics import accuracy_score, f1_score

# 读取数据
with open('/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/merged_dataset_15_reflected_with_sentiment.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取预测值和真实标签

# 提取真实标签和预测标签
y_true = [item["sentiment"].lower() for item in data]
y_pred = [item["reflect_sentiment"].lower() for item in data]

# 计算准确率和宏观F1分数
accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average='macro')

print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")


# import json
# import re

# # 定义情感极性关键词（正则匹配用）
# sentiment_keywords = ['positive', 'neutral', 'negative']

# # 编译正则表达式，匹配任意位置的情感词（忽略大小写）
# pattern = re.compile(r'\b(' + '|'.join(sentiment_keywords) + r')\b', re.IGNORECASE)

# def extract_sentiment_from_reflection(reflection_text):
#     """
#     从 Reflection Results 中提取第一个出现的情感极性词。
#     """
#     match = pattern.search(reflection_text)
#     if match:
#         return match.group(1).lower()  # 返回小写的情感词
#     else:
#         return None  # 如果没找到，返回 None

# # 数据集路径
# file_path = "/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2015/merged_dataset_15_reflected.json"

# # 存储真实标签和预测标签
# y_true = []
# y_pred = []

# # 读取数据集
# with open(file_path, 'r', encoding='utf-8') as f:
#     data = json.load(f)

# # 遍历每条样本
# for item in data:
#     sentiment_label = item.get("sentiment", "").strip().lower()
#     reflection_text = item.get("Reflection Results", "")
    
#     # 只处理 sentiment 是有效值的情况
#     if sentiment_label not in ['positive', 'neutral', 'negative']:
#         continue  # 跳过无效标签
    
#     # 提取 Reflection Results 中的情感
#     extracted = extract_sentiment_from_reflection(reflection_text)
    
#     if extracted is None:
#         # 如果没提取到，可以选择跳过或设为 'neutral' 等默认值
#         # 这里选择跳过
#         continue
    
#     # 添加到列表
#     y_true.append(sentiment_label)
#     y_pred.append(extracted)

# # 计算准确率和宏观 F1
# from sklearn.metrics import accuracy_score, f1_score

# accuracy = accuracy_score(y_true, y_pred)
# macro_f1 = f1_score(y_true, y_pred, average='macro', labels=['positive', 'neutral', 'negative'])

# # 输出结果
# print(f"Total samples processed: {len(y_true)}")
# print(f"Accuracy: {accuracy:.4f}")
# print(f"Macro-F1: {macro_f1:.4f}")

# # 可选：输出分类报告
# from sklearn.metrics import classification_report
# print("\nClassification Report:")
# print(classification_report(y_true, y_pred, labels=['positive', 'neutral', 'negative']))
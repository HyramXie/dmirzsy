import json
from collections import Counter
import re
from sklearn.metrics import accuracy_score, f1_score

# 标签映射：根据你提供的
LABEL_MAP = {
    1: "Positive",
    -1: "Negative",
    0: "Neutral"
}

def extract_sentiment(output):
    """从 model_output 文本中提取 Sentiment 标签"""
    match = re.search(r'Sentiment:\s*([a-zA-Z]+)', output, re.IGNORECASE)
    if match:
        pred = match.group(1).capitalize()
        # 统一标准拼写
        if pred in ['Positive', 'Negative', 'Neutral']:
            return pred
    return None

# 假设 data 是一个列表，包含多个样本
# data = [...]  # 替换为你的实际数据加载方式
with open("/public/home/byxu_jsjxy/ywl/LLaMA-Factory/data/2017/fenmotai/3-fused_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

all_ground_truths = []
all_predictions = []

for item in data:
    # 获取真实标签（使用 'sentiment' 字段）
    try:
        raw_sentiment = item["sentiment"]
        int_label = int(raw_sentiment)
        gt = LABEL_MAP[int_label]
        all_ground_truths.append(gt)
    except (ValueError, KeyError) as e:
        print(f"Invalid sentiment label: {item['sentiment']}, skipping...")
        continue  # 跳过无效标签

    # 从 model_outputs 提取预测
    preds = []
    for output in item["model_outputs"]:
        pred = extract_sentiment(output)
        if pred is not None:
            preds.append(pred)

    # 多数投票
    if len(preds) == 0:
        final_pred = "Neutral"  # 默认回退
    else:
        vote_count = Counter(preds)
        final_pred = vote_count.most_common(1)[0][0]  # 得票最多

    all_predictions.append(final_pred)

# === 计算评估指标 ===
acc = accuracy_score(all_ground_truths, all_predictions)
f1_macro = f1_score(all_ground_truths, all_predictions, average='macro')

print(f"Accuracy: {acc:.4f}")
print(f"Macro-F1: {f1_macro:.4f}")

# （可选）详细分类报告
from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(all_ground_truths, all_predictions))
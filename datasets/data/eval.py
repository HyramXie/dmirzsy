import json
from sklearn.metrics import accuracy_score, f1_score

# 定义映射关系
label_map = {
    "positive": 1,
    "neutral": 0,
    "negative": -1
}

data = []
with open("/home/cbf00006701/zsy/LLaMA-Factory/eval_q/t+c_15.jsonl", "r") as f:
    for line in f:
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"解析错误，跳过该行: {line}, 错误: {e}")

# 提取真实值和预测值
y_true = []
y_pred = []
skipped = 0

for item in data:
    try:
        # 标准化 label 和 predict
        true_str = str(item["label"]).strip().lower()
        pred_str = str(item["predict"]).strip().lower()

        # 尝试将字符串转换为数字，如果是文字就映射
        true_label = int(true_str) if true_str.lstrip("-").isdigit() else label_map[true_str]
        pred_label = int(pred_str) if pred_str.lstrip("-").isdigit() else label_map[pred_str]

        y_true.append(true_label)
        y_pred.append(pred_label)
    except (ValueError, KeyError) as e:
        print(f"跳过无效数据: {item}, 错误: {e}")
        skipped += 1

# 计算指标
if y_true:
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"总样本数: {len(data)}, 有效样本: {len(y_true)}, 跳过: {skipped}")
    print(f"准确率: {accuracy:.4f}")
    print(f"F1 分数: {f1:.4f}")
else:
    print("没有有效样本。")

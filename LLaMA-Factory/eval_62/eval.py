# # python /home/cbf00006701/zsy/LLaMA-Factory/eval_c/eval.py
# import json
# from sklearn.metrics import accuracy_score, f1_score

# # 定义映射关系
# label_map = {
#     "positive": 1,
#     "neutral": 0,
#     "negative": -1
# }

# data = []
# with open("/home/cbf00006701/zsy/LLaMA-Factory/eval/qwen2.5vl-3b/sc_new_15.jsonl", "r") as f:
#     for line in f:
#         try:
#             data.append(json.loads(line))
#         except json.JSONDecodeError as e:
#             print(f"解析错误，跳过该行: {line}, 错误: {e}")

# # 提取真实值和预测值
# y_true = []
# y_pred = []
# skipped = 0

# for item in data:
#     try:
#         # 标准化 label 和 predict
#         true_str = str(item["label"]).strip().lower()
#         pred_str = str(item["predict"]).strip().lower()

#         # 尝试将字符串转换为数字，如果是文字就映射
#         true_label = int(true_str) if true_str.lstrip("-").isdigit() else label_map[true_str]
#         pred_label = int(pred_str) if pred_str.lstrip("-").isdigit() else label_map[pred_str]

#         y_true.append(true_label)
#         y_pred.append(pred_label)
#     except (ValueError, KeyError) as e:
#         print(f"跳过无效数据: {item}, 错误: {e}")
#         skipped += 1

# # 计算指标
# if y_true:
#     accuracy = accuracy_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred, average='macro')
#     print(f"总样本数: {len(data)}, 有效样本: {len(y_true)}, 跳过: {skipped}")
#     print(f"准确率: {accuracy:.4f}")
#     print(f"F1 分数: {f1:.4f}")
# else:
#     print("没有有效样本。")


# python /root/user/LLaMA-Factory/eval_62/eval.py
import json
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 定义映射关系
label_map = {
    "positive": 1,
    "neutral": 0,
    "negative": -1
}

# 反向映射，用于输出标签名称
reverse_label_map = {
    1: "positive",
    0: "neutral",
    -1: "negative"
}

data = []
with open("/root/user/zsy/LLaMA-Factory/eval_62/qwen2.5vl-7b/mvsa_test_1.jsonl", "r") as f:
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
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    # 计算每个类别的F1分数
    positive_f1 = f1_score(y_true, y_pred, average=None, labels=[1])[0]
    neutral_f1 = f1_score(y_true, y_pred, average=None, labels=[0])[0]
    negative_f1 = f1_score(y_true, y_pred, average=None, labels=[-1])[0]
    
    # 使用classification_report获取更详细的信息
    report = classification_report(y_true, y_pred, labels=[1, 0, -1], 
                                  target_names=["positive", "neutral", "negative"])
    correct_count = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    incorrect_count = len(y_true) - correct_count
    
    print(f"总样本数: {len(data)}, 有效样本: {len(y_true)}, 跳过: {skipped}")
    print(f"准确率: {accuracy:.4f}")
    print(f"宏F1: {macro_f1:.4f}")
    print(f"积极F1: {positive_f1:.4f}")
    print(f"中立F1: {neutral_f1:.4f}")
    print(f"消极F1: {negative_f1:.4f}")
    print(f"预测正确样本: {correct_count}")
    print(f"预测错误样本: {incorrect_count}")
    print("\n详细分类报告:")
    print(report)
else:
    print("没有有效样本。")
# python /home/cbf00006701/zsy/LLaMA-Factory/eval_c/eval.py
import json
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict

# 定义映射关系
label_map = {
    "positive": 1,
    "neutral": 0,
    "negative": -1
}

# 读取整个JSON数组
try:
    with open("/root/user/LLaMA-Factory/eval_train/merged_dataset_sc_new_17.json", "r") as f:
        data = json.load(f)
    print(f"成功读取JSON数组，包含 {len(data)} 个对象")
except json.JSONDecodeError as e:
    print(f"解析JSON数组错误: {e}")
    # 如果失败，尝试逐行读取（JSONL格式）
    data = []
    with open("eval/qwen2.5vl-7b/merged_dataset_sc_new_15.json", "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    print(f"使用JSONL格式读取，包含 {len(data)} 个对象")

# 按image_id分组统计aspect数量
image_aspect_count = defaultdict(int)
for item in data:
    image_id = item.get("image_id", "")
    if image_id:
        image_aspect_count[image_id] += 1

# 提取真实值和预测值，并按aspect数量分类
y_true_all = []
y_pred_all = []
y_true_single_aspect = []  # aspect = 1
y_pred_single_aspect = []  # aspect = 1
y_true_multi_aspect = []   # aspect ≥ 2
y_pred_multi_aspect = []   # aspect ≥ 2

skipped = 0

for item in data:
    try:
        # 使用 "sentiment" 作为真实标签，"predict" 作为预测标签
        true_str = str(item.get("sentiment", "")).strip().lower()
        pred_str = str(item.get("predict", "")).strip().lower()

        if not true_str or not pred_str:
            skipped += 1
            continue

        # 尝试将字符串转换为数字，如果是文字就映射
        true_label = int(true_str) if true_str.lstrip("-").isdigit() else label_map[true_str]
        pred_label = int(pred_str) if pred_str.lstrip("-").isdigit() else label_map[pred_str]

        image_id = item.get("image_id", "")
        aspect_count = image_aspect_count.get(image_id, 0)
        
        # 添加到所有数据
        y_true_all.append(true_label)
        y_pred_all.append(pred_label)
        
        # 按aspect数量分类
        if aspect_count == 1:
            y_true_single_aspect.append(true_label)
            y_pred_single_aspect.append(pred_label)
        elif aspect_count >= 2:
            y_true_multi_aspect.append(true_label)
            y_pred_multi_aspect.append(pred_label)
            
    except (ValueError, KeyError) as e:
        print(f"跳过无效数据: {item}, 错误: {e}")
        skipped += 1

# 计算指标
if y_true_all:
    # 总体指标
    accuracy = accuracy_score(y_true_all, y_pred_all)
    macro_f1 = f1_score(y_true_all, y_pred_all, average='macro')
    
    print(f"总样本数: {len(data)}, 有效样本: {len(y_true_all)}, 跳过: {skipped}")
    print(f"准确率: {accuracy:.4f}")
    print(f"宏F1: {macro_f1:.4f}")
    print()
    
    # aspect = 1 时的指标
    if y_true_single_aspect:
        single_accuracy = accuracy_score(y_true_single_aspect, y_pred_single_aspect)
        single_f1 = f1_score(y_true_single_aspect, y_pred_single_aspect, average='macro')
        print(f"aspect = 1 的样本数: {len(y_true_single_aspect)}")
        print(f"aspect = 1 准确率: {single_accuracy:.4f}")
        print(f"aspect = 1 F1: {single_f1:.4f}")
    else:
        print("没有aspect = 1的有效样本")
    print()
    
    # aspect ≥ 2 时的指标
    if y_true_multi_aspect:
        multi_accuracy = accuracy_score(y_true_multi_aspect, y_pred_multi_aspect)
        multi_f1 = f1_score(y_true_multi_aspect, y_pred_multi_aspect, average='macro')
        print(f"aspect ≥ 2 的样本数: {len(y_true_multi_aspect)}")
        print(f"aspect ≥ 2 准确率: {multi_accuracy:.4f}")
        print(f"aspect ≥ 2 F1: {multi_f1:.4f}")
    else:
        print("没有aspect ≥ 2的有效样本")
    print()
    
    # 打印统计信息
    print("图像统计信息:")
    single_image_count = sum(1 for count in image_aspect_count.values() if count == 1)
    multi_image_count = sum(1 for count in image_aspect_count.values() if count >= 2)
    print(f"单aspect图像数量: {single_image_count}")
    print(f"多aspect图像数量: {multi_image_count}")
    print(f"总图像数量: {len(image_aspect_count)}")
    
    # 打印每个类别的详细F1分数
    print("\n各类别F1分数:")
    labels = [1, 0, -1]
    label_names = ["positive", "neutral", "negative"]
    f1_scores = f1_score(y_true_all, y_pred_all, average=None, labels=labels)
    
    for label, name, score in zip(labels, label_names, f1_scores):
        print(f"{name} F1: {score:.4f}")
    
else:
    print("没有有效样本。")
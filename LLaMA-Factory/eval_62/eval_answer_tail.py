import json
import re
import argparse
from sklearn.metrics import accuracy_score, f1_score, classification_report

label_map = {
    "positive": 1,
    "neutral": 0,
    "negative": -1
}

def extract_answer_label(text):
    matches = re.findall(r"Answer:\s*(positive|neutral|negative)", str(text), flags=re.IGNORECASE)
    if not matches:
        return None
    return matches[-1].strip().lower()

def main(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    y_true = []
    y_pred = []
    skipped = 0

    skipped_indices = []
    for idx, item in enumerate(data):
        try:
            true_str = str(item["label"]).strip().lower()
            pred_field = item.get("predict", "")
            pred_str_extracted = extract_answer_label(pred_field)
            if pred_str_extracted is None:
                raise ValueError("no answer label")

            true_label = int(true_str) if true_str.lstrip("-").isdigit() else label_map[true_str]
            pred_label = int(pred_str_extracted) if pred_str_extracted.lstrip("-").isdigit() else label_map[pred_str_extracted]

            y_true.append(true_label)
            y_pred.append(pred_label)
        except Exception:
            skipped += 1
            skipped_indices.append(idx)

    if y_true:
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        positive_f1 = f1_score(y_true, y_pred, average=None, labels=[1])[0]
        neutral_f1 = f1_score(y_true, y_pred, average=None, labels=[0])[0]
        negative_f1 = f1_score(y_true, y_pred, average=None, labels=[-1])[0]
        report = classification_report(y_true, y_pred, labels=[1, 0, -1], target_names=["positive", "neutral", "negative"])
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
        if skipped_indices:
            print(f"跳过样本索引: {skipped_indices}")
        print("\n详细分类报告:")
        print(report)
    else:
        print("没有有效样本。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=False, default="/root/user/zsy/LLaMA-Factory/eval_62/qwen2.5vl-3b/mvsa_sc.jsonl")
    args = parser.parse_args()
    main(args.path)
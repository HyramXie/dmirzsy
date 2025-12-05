import json
import argparse
from sklearn.metrics import f1_score, accuracy_score

def to_binary(value):
    s = str(value).strip().lower()
    if s.lstrip("-").isdigit():
        v = int(s)
        if v > 0:
            return 1
        if v < 0:
            return 0
        raise KeyError("中立标签被移除")
    mapping = {"positive": 1, "negative": 0}
    if s in mapping:
        return mapping[s]
    if s == "neutral":
        raise KeyError("中立标签被移除")
    raise KeyError(f"未知标签: {value}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    data = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    y_true = []
    y_pred = []
    skipped = 0

    for item in data:
        try:
            t = to_binary(item.get("label"))
            p = to_binary(item.get("predict"))
            y_true.append(t)
            y_pred.append(p)
        except (ValueError, KeyError, TypeError):
            skipped += 1

    if y_true:
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        f1s = f1_score(y_true, y_pred, average=None, labels=[1, 0])
        positive_f1, negative_f1 = f1s[0], f1s[1]
        print(f"总样本数: {len(data)}, 有效样本: {len(y_true)}, 跳过: {skipped}")
        print(f"准确率: {acc:.4f}")
        print(f"宏F1: {macro_f1:.4f}")
        print(f"积极F1: {positive_f1:.4f}")
        print(f"消极F1: {negative_f1:.4f}")
    else:
        print("没有有效样本。")

if __name__ == "__main__":
    main()
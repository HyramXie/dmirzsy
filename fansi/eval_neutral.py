import json
import sys

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_true_neutral(item):
    msgs = item.get("messages") or []
    label = None
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "assistant":
            c = m.get("content")
            if isinstance(c, str):
                label = c.strip().lower()
    return label == "neutral"

def get_pred_neutral(item):
    p = item.get("predict——sentiment")
    return isinstance(p, str) and p.strip().lower() == "neutral"

def eval_neutral(items):
    tp = fp = fn = tn = 0
    for it in items:
        y = 1 if get_true_neutral(it) else 0
        yhat = 1 if get_pred_neutral(it) else 0
        if y == 1 and yhat == 1:
            tp += 1
        elif y == 0 and yhat == 1:
            fp += 1
        elif y == 1 and yhat == 0:
            fn += 1
        else:
            tn += 1
    n = tp + fp + fn + tn
    acc = (tp + tn) / n if n else 0.0
    f1_neu = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    f1_non = (2 * tn) / (2 * tn + fn + fp) if (2 * tn + fn + fp) else 0.0
    macro_f1 = (f1_neu + f1_non) / 2
    return {
        "count": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": acc,
        "f1_neutral": f1_neu,
        "f1_non_neutral": f1_non,
        "macro_f1": macro_f1,
    }

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/root/user/zsy/fansi/1-react.json"
    items = load(path)
    res = eval_neutral(items)
    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
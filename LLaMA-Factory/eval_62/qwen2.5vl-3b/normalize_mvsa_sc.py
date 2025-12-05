import argparse
import json
import os
import re

def extract_answer(text):
    if not isinstance(text, str) or not text:
        return None
    m = re.search(r"Answer\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    tail = m.group(1)
    tail = tail.strip()
    parts = re.split(r"[\r\n]", tail)
    first = parts[0].strip()
    first = re.sub(r"^[-\s]*", "", first)
    first = re.sub(r"\s+$", "", first)
    first = first.strip()
    if not first:
        return None
    return first

def extract_correct(text):
    if not isinstance(text, str) or not text:
        return None
    m = re.search(r"The\s+correct\s+sentiment\s+is\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    tail = m.group(1)
    tail = tail.strip()
    tail = re.sub(r"^[-\s]*", "", tail)
    # 仅捕获常见单词/数值，避免抓到后续解释
    m2 = re.match(r"^([A-Za-z+-]?\d+|positive|neutral|negative)\b", tail, flags=re.IGNORECASE)
    if m2:
        return m2.group(1)
    # 兜底：按换行/句号截断
    first = re.split(r"[\r\n\.]+", tail)[0].strip()
    return first or None

def normalize_jsonl(in_path: str, out_path: str) -> tuple[int, int]:
    changed = 0
    total = 0
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            obj = json.loads(line)
            pred = obj.get("predict", "")
            ans = extract_answer(pred)
            if ans is None:
                ans = extract_correct(pred)
            if ans is not None and ans != pred:
                obj["predict"] = ans
                changed += 1
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return total, changed

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=False)
    args = p.parse_args()
    in_path = args.in_path
    out_path = args.out_path or os.path.splitext(in_path)[0] + ".normalized.jsonl"
    total, changed = normalize_jsonl(in_path, out_path)
    print(json.dumps({"input": in_path, "output": out_path, "total": total, "changed": changed}, ensure_ascii=False))

if __name__ == "__main__":
    main()
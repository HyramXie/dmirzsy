import json
import os


def _norm(x: str) -> str:
    s = (x or "").strip().lower()
    if s in {"+", "pos", "1"}:
        return "positive"
    if s in {"0", "neu"}:
        return "neutral"
    if s in {"-", "neg", "-1"}:
        return "negative"
    return s


def build_single_turn_dataset(input_jsonl: str, output_json: str) -> None:
    items = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue

    result = []
    for s in items:
        prompt = s.get("prompt", "")
        predict = _norm(s.get("predict", ""))
        label = _norm(s.get("label", ""))
        reflection = s.get("Reflection", "").strip()
        improvement = s.get("Improvement", "").strip()

        if predict != label:
            assistant = (
                f"{predict}. Sorry, I made a mistake. {reflection} {improvement} "
                f"The correct sentiment is {label}. Answer: {label}"
            )
        else:
            assistant = (
                f"{predict}. The prediction is correct. {reflection} {improvement} "
                f"The sentiment is {label}. Answer: {label}"
            )

        conversation = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant},
            ]
        }
        result.append(conversation)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    in_15 = "/root/user/zsy/LLaMA-Factory/eval_train/62/qwen2.5vl-3b/15_explain_all.jsonl"
    out_15 = "/root/user/zsy/LLaMA-Factory/eval_train/62/qwen2.5vl-3b/15_fine_tuning_explain_all.json"
    if os.path.exists(in_15):
        build_single_turn_dataset(in_15, out_15)
        print(out_15)
    in_17 = "/root/user/zsy/LLaMA-Factory/eval_train/62/qwen2.5vl-3b/17_explain_all.jsonl"
    out_17 = "/root/user/zsy/LLaMA-Factory/eval_train/62/qwen2.5vl-3b/17_fine_tuning_explain_all.json"
    if os.path.exists(in_17):
        build_single_turn_dataset(in_17, out_17)
        print(out_17)
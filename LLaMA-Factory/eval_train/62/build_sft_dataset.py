import json
import os
import re


def _extract_system_and_user(prompt: str) -> tuple[str, str]:
    system_match = re.search(r"system\n([\s\S]*?)\nuser", prompt)
    user_match = re.search(r"user\n([\s\S]*?)\nassistant", prompt)
    system = system_match.group(1).strip() if system_match else ""
    user = user_match.group(1).strip() if user_match else prompt.strip()
    return system, user


def _norm_label(label: str) -> str:
    s = (label or "").strip().lower()
    if s in {"positive", "+", "pos"}:
        return "positive"
    if s in {"neutral", "neu", "0"}:
        return "neutral"
    if s in {"negative", "neg", "-", "-1"}:
        return "negative"
    return s


def build_answer_only_dataset(input_file: str, output_file: str) -> None:
    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except Exception:
                continue

    data = []
    for s in samples:
        system, user = _extract_system_and_user(s.get("prompt", ""))
        label = _norm_label(s.get("label", ""))
        convo = {"messages": []}
        if system:
            convo["messages"].append({"role": "system", "content": system})
        convo["messages"].append({"role": "user", "content": user})
        convo["messages"].append({"role": "assistant", "content": label})
        data.append(convo)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_correction_dataset(input_file: str, output_file: str) -> None:
    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except Exception:
                continue

    data = []
    for s in samples:
        predict = _norm_label(s.get("predict", ""))
        label = _norm_label(s.get("label", ""))
        if not predict or not label or predict == label:
            continue
        system, user = _extract_system_and_user(s.get("prompt", ""))
        correction_instruction = "请反思并纠正你的答案，仅输出一个词：positive、neutral 或 negative。"
        convo = {"messages": []}
        if system:
            convo["messages"].append({"role": "system", "content": system})
        convo["messages"].append({"role": "user", "content": user})
        convo["messages"].append({"role": "assistant", "content": predict})
        convo["messages"].append({"role": "user", "content": correction_instruction})
        convo["messages"].append({"role": "assistant", "content": label})
        data.append(convo)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    in_file = "/root/user/zsy/LLaMA-Factory/eval_train/62/qwen2.5vl-3b/15_explain_all.jsonl"
    base_dir = os.path.dirname(in_file)
    out_answer = os.path.join(base_dir, "15_sft_answer_only.json")
    out_correction = os.path.join(base_dir, "15_sft_correction.json")
    build_answer_only_dataset(in_file, out_answer)
    build_correction_dataset(in_file, out_correction)
    print(json.dumps({"answer_only": out_answer, "correction": out_correction}, ensure_ascii=False))
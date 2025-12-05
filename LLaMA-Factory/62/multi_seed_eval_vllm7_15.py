import argparse
import json
import os
import re
import random
import importlib.util
from statistics import mean, stdev

def load_vllm_infer(module_path):
    spec = importlib.util.spec_from_file_location("vllm_infer7", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.vllm_infer

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows.append((obj.get("predict", ""), obj.get("label", "")))
    return rows

def normalize_text(s):
    s = str(s).strip()
    m = re.search(r"(positive|negative|neutral|pos|neg|neu)", s, re.IGNORECASE)
    if m:
        t = m.group(1).lower()
        if t in ("pos", "neg", "neu"):
            t = {"pos": "positive", "neg": "negative", "neu": "neutral"}[t]
        return t
    return s.lower()

def compute_metrics(rows):
    labels = [normalize_text(r[1]) for r in rows]
    preds = [normalize_text(r[0]) for r in rows]
    classes = sorted(set(labels))
    correct = [int(p == y) for p, y in zip(preds, labels)]
    acc = sum(correct) / len(correct) if correct else 0.0
    f1s = []
    for c in classes:
        tp = sum(1 for p, y in zip(preds, labels) if p == c and y == c)
        fp = sum(1 for p, y in zip(preds, labels) if p == c and y != c)
        fn = sum(1 for p, y in zip(preds, labels) if p != c and y == c)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    macro_f1 = mean(f1s) if f1s else 0.0
    return {"accuracy": acc, "macro_f1": macro_f1, "per_sample_correct": correct}

def paired_permutation_test(a, b, iters=10000, rng_seed=12345):
    rng = random.Random(rng_seed)
    diffs = [float(x) - float(y) for x, y in zip(a, b)]
    obs = abs(sum(diffs) / (len(diffs) if diffs else 1))
    if not diffs:
        return 1.0
    count = 0
    for _ in range(iters):
        s = 0.0
        for d in diffs:
            s += d if rng.random() < 0.5 else -d
        stat = abs(s / len(diffs))
        if stat >= obs:
            count += 1
    return count / iters

def mcnemar_p_value(a, b):
    b01 = 0
    b10 = 0
    for x, y in zip(a, b):
        if x and not y:
            b10 += 1
        elif y and not x:
            b01 += 1
    n = b01 + b10
    if n == 0:
        return 1.0
    from math import comb
    p = 0.0
    for k in range(0, min(b01, b10) + 1):
        p += comb(n, k) * (0.5 ** n)
    p *= 2.0
    if p > 1.0:
        p = 1.0
    return p

def run_once(vllm_infer_fn, args, seed, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    vllm_infer_fn(
        model_name_or_path=args.model_path,
        adapter_name_or_path=args.adapter_path,
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        template=args.template,
        cutoff_len=args.cutoff_len,
        max_samples=args.max_samples,
        vllm_config=args.vllm_config,
        save_name=save_path,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        skip_special_tokens=True,
        default_system=None,
        enable_thinking=True,
        seed=seed,
        pipeline_parallel_size=args.pipeline_parallel_size,
        image_max_pixels=args.image_max_pixels,
        image_min_pixels=args.image_min_pixels,
        video_fps=args.video_fps,
        video_maxlen=args.video_maxlen,
        batch_size=args.batch_size,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/root/user/zsy/LLaMA-Factory/output_62/qwen2_vl-7b/lora/sft_15_sc_new")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="mllm_demo")
    parser.add_argument("--dataset_dir", type=str, default="data/2015")
    parser.add_argument("--template", type=str, default="qwen2_vl")
    parser.add_argument("--cutoff_len", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--vllm_config", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--top_p", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--image_max_pixels", type=int, default=384*384)
    parser.add_argument("--image_min_pixels", type=int, default=32*32)
    parser.add_argument("--video_fps", type=float, default=2.0)
    parser.add_argument("--video_maxlen", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 123, 3407, 777, 2025])
    parser.add_argument("--module_path", type=str, default=os.path.join(os.path.dirname(__file__), "vllm_infer7.py"))
    parser.add_argument("--save_dir", type=str, default=os.path.join(os.path.dirname(__file__), "..", "eval_62", "qwen2vl-7b"))
    parser.add_argument("--prefix", type=str, default="sc_15")
    args = parser.parse_args()

    vllm_infer_fn = load_vllm_infer(args.module_path)

    all_metrics = []
    per_seed_correct = []
    seed_ids = []

    for s in args.seeds:
        save_path = os.path.join(args.save_dir, f"{args.prefix}_seed{s}.jsonl")
        run_once(vllm_infer_fn, args, s, save_path)
        rows = read_jsonl(save_path)
        m = compute_metrics(rows)
        all_metrics.append({"seed": s, **m})
        per_seed_correct.append(m["per_sample_correct"])
        seed_ids.append(s)

    accs = [m["accuracy"] for m in all_metrics]
    f1s = [m["macro_f1"] for m in all_metrics]
    acc_mean = mean(accs) if accs else 0.0
    acc_std = stdev(accs) if len(accs) > 1 else 0.0
    f1_mean = mean(f1s) if f1s else 0.0
    f1_std = stdev(f1s) if len(f1s) > 1 else 0.0

    best_idx = max(range(len(accs)), key=lambda i: accs[i]) if accs else 0
    best_seed = seed_ids[best_idx] if seed_ids else None
    pvals = []
    for i, s in enumerate(seed_ids):
        if i == best_idx:
            continue
        p_perm = paired_permutation_test(per_seed_correct[best_idx], per_seed_correct[i])
        p_mcn = mcnemar_p_value(per_seed_correct[best_idx], per_seed_correct[i])
        pvals.append((best_seed, s, p_perm, p_mcn))

    print("| Seed | Accuracy | Macro-F1 |")
    print("| --- | --- | --- |")
    for m in all_metrics:
        print(f"| {m['seed']} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} |")
    print(f"| Mean±Std | {acc_mean:.4f}±{acc_std:.4f} | {f1_mean:.4f}±{f1_std:.4f} |")

    if pvals:
        print("Significance vs best:")
        print("method=permutation and mcnemar")
        for a, b, p1, p2 in pvals:
            sig1 = "YES" if p1 < 0.05 else "NO"
            sig2 = "YES" if p2 < 0.05 else "NO"
            print(f"best={a} vs seed={b}: perm p={p1:.6f} ({sig1}), mcnemar p={p2:.6f} ({sig2})")

if __name__ == "__main__":
    main()
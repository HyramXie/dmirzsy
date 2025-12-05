import json
import re

def normalize_text(text):
    """去除标点、空格、换行，转小写，便于比较"""
    return re.sub(r'[\s\"\'\.\,\!\?\:\;\(\)]', '', text).lower()

def extract_text(content):
    match = re.search(r"Based on the image and the text:\s*'([^']+)'", content)
    if match:
        return match.group(1).strip()
    return None

# 读取两个数据集
with open('/root/user/zsy/LLaMA-Factory/data/mvsa_train.json', 'r', encoding='utf-8') as f:
    dataset1 = json.load(f)

with open('/root/user/zsy/LLaMA-Factory/eval_train/62/qwen2.5vl-3b/mvsa_fine_tuning_explain.json', 'r', encoding='utf-8') as f:
    dataset2 = json.load(f)

text_to_image = {}
for item in dataset1:
    if 'messages' not in item or len(item['messages']) == 0:
        continue
    user_content = None
    for msg in item['messages']:
        if msg['role'] == 'user':
            user_content = msg['content']
            break
    if not user_content:
        continue
    t = extract_text(user_content)
    if t:
        text_to_image[t] = item['images'][0]

matched = 0
unmatched = 0
new_dataset2 = []
for item in dataset2:
    new_item = {k: v for k, v in item.items()}
    user_content = None
    for msg in item['messages']:
        if msg['role'] == 'user':
            user_content = msg['content']
            break
    if not user_content:
        new_item['images'] = []
        new_dataset2.append(new_item)
        unmatched += 1
        continue
    t = extract_text(user_content)
    if not t:
        new_item['images'] = []
        new_dataset2.append(new_item)
        unmatched += 1
        continue
    if t in text_to_image:
        new_item['images'] = [text_to_image[t]]
        matched += 1
    else:
        new_item['images'] = []
        unmatched += 1
    new_dataset2.append(new_item)

with open('/root/user/zsy/LLaMA-Factory/eval_train/62/qwen2.5vl-3b/mvsa_fine_tuning_explain.json.bak', 'w', encoding='utf-8') as f:
    json.dump(dataset2, f, indent=2, ensure_ascii=False)
with open('/root/user/zsy/LLaMA-Factory/eval_train/62/qwen2.5vl-3b/mvsa_fine_tuning_explain.json', 'w', encoding='utf-8') as f:
    json.dump(new_dataset2, f, indent=2, ensure_ascii=False)
print(f"matched: {matched}, unmatched: {unmatched}")

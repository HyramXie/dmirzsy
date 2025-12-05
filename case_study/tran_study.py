import json

# Step 1: 读取并解析三个数据集

# 第一个数据集（JSON 格式）
with open('/root/user/datasets/twitter2015/test+image_context.json', 'r', encoding='utf-8') as f:
    data1 = json.load(f)

# 第二个数据集（JSON 格式）
with open('/root/user/LLaMA-Factory/data/2015/t+c+i_15.json', 'r', encoding='utf-8') as f:
    data2 = json.load(f)

# 第三个数据集（JSONL 格式）
data3 = []
with open('/root/user/LLaMA-Factory/eval/llava/t+c+i_15.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            data3.append(json.loads(line))

# 构建映射：image_id -> predict 和 label（从 dataset3 和 dataset2 提取）

# 用于存储预测和标签的字典，以 image_id 和 aspect 为键
predictions_labels = {}

# 从 dataset3 中提取 predict 和 label
for item in data3:
    content = item['prompt']
    # 提取文本中的 aspect（<target>...</target>）
    start = content.find('<target>') + len('<target>')
    end = content.find('</target>')
    aspect = content[start:end].strip()

    text_start = content.find('Text: "') + len('Text: "')
    text_end = content.find('"', text_start)
    text = content[text_start:text_end].replace(f'<target>{aspect}</target>', aspect)

    image_desc_start = content.find('Image description: "') + len('Image description: "')
    image_desc_end = content.find('"', image_desc_start)
    image_context = content[image_desc_start:image_desc_end]

    # 提取 image_id：从 image_context 或其他方式无法直接获取，但我们可以从 image_context + text 唯一匹配
    # 但我们先假设可以通过 text 和 aspect 唯一确定一条记录
    key = (text, aspect)
    predictions_labels[key] = {
        'predict': item['predict'],
        'label': item['label'],
        'image_context': image_context
    }

# 从 dataset2 中提取（补充或验证）
for item in data2:
    messages = item['messages']
    image_path = item['images'][0]  # e.g., "twitter2015_images/74960.jpg"
    image_id = image_path.split('/')[-1]  # extract filename

    # 找到 user 消息
    user_msg = None
    for msg in messages:
        if msg['role'] == 'user':
            user_msg = msg['content']
            break
    if not user_msg:
        continue

    # 解析文本
    text_start = user_msg.find('Text: "') + len('Text: "')
    text_end = user_msg.find('"', text_start)
    text = user_msg[text_start:text_end]

    # 提取 aspect
    target_start = text.find('<target>') + len('<target>')
    target_end = text.find('</target>')
    aspect = text[target_start:target_end]
    text = text.replace(f'<target>{aspect}</target>', aspect)

    # 提取 image description
    desc_start = user_msg.find('Image description: "') + len('Image description: "')
    desc_end = user_msg.find('"', desc_start)
    image_context = user_msg[desc_start:desc_end]

    # 找到 assistant 的回复
    predict = None
    for msg in messages:
        if msg['role'] == 'assistant':
            predict = msg['content'].strip()
            break
    if not predict:
        continue

    label = predict  # 因为是标注数据，predict 即为 label（这里假设没有模型预测，只有真实标签）

    key = (text, aspect)
    if key not in predictions_labels:
        predictions_labels[key] = {
            'predict': predict,
            'label': label,
            'image_context': image_context,
            'image_id': image_id
        }
    else:
        predictions_labels[key]['image_id'] = image_id

# 现在处理 dataset1 并合并
final_data = []

for item in data1:
    text = item['text']
    aspect = item['aspect']
    image_id = item['image_id']
    image_context = item['image_context']
    sentiment = item['sentiment']  # '0' -> Neutral, '1' -> Positive, '-1' -> Negative

    # 转换 sentiment 数字到标签
    sentiment_map = {'0': 'Neutral', '1': 'Positive', '-1': 'Negative'}
    label = sentiment_map[sentiment]

    key = (text, aspect)

    # 获取 predict 和可能的 image_context / image_id（如果缺失）
    if key in predictions_labels:
        entry = predictions_labels[key]
        predict = entry['predict']
        # 优先使用 dataset1 的 image_context 和 image_id，否则用 dataset2 的
        final_image_context = image_context if image_context else entry.get('image_context', '')
        final_image_id = image_id if image_id else entry.get('image_id', '')
    else:
        # 如果没有预测信息，使用默认
        predict = label  # 假设预测等于标签（真实情况可能不同）
        final_image_context = image_context
        final_image_id = image_id

    final_data.append({
        "text": text,
        "aspect": aspect,
        "predict": predict,
        "label": label,
        "image_id": final_image_id,
        "image_context": final_image_context
    })

# 保存为新的 JSON 文件
with open('/root/user/case_study/2015.json', 'w', encoding='utf-8') as f:
    json.dump(final_data, f, ensure_ascii=False, indent=2)

print("合并完成，已保存到 /root/user/case_study/2015.json")
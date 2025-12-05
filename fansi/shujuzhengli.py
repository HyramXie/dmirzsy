import json

# 读取数据
with open('/root/user/LLaMA-Factory/32B/2015/tongji.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 初始化计数器
count_total = len(data)
count_match = 0          # final_sentiment == label
count_mismatch = 0       # final_sentiment != label

# 子统计：在 mismatch 中
count_text_image_consistent = 0   # 文图情感一致
count_text_image_inconsistent = 0 # 文图情感不一致

# 在 mismatch 且 文图不一致 的情况下
count_label_match_text = 0        # label 与文本情感一致
count_label_match_image = 0       # label 与图像情感一致

# 情感类别映射（防止大小写问题）
def normalize(sentiment):
    return str(sentiment).strip().lower()

# 遍历每条数据
for item in data:
    final = normalize(item['final_sentiment'])
    label = normalize(item['label'])
    text = normalize(item['text_sentiment'])
    image = normalize(item['image_sentiment'])

    if final == label:
        count_match += 1
    else:
        count_mismatch += 1

        # 判断图文是否一致
        if text == image:
            count_text_image_consistent += 1
        else:
            count_text_image_inconsistent += 1

            # 进一步判断 label 更接近 text 还是 image
            if label == text:
                count_label_match_text += 1
            if label == image:
                count_label_match_image += 1

# 输出结果
print("🔍 数据集情感一致性分析结果")
print("="*50)
print(f"总数据量: {count_total}")
print(f"① final_sentiment 与 label 一致的数量: {count_match}")
print(f"① final_sentiment 与 label 不一致的数量: {count_mismatch}")

print(f"\n② 在不一致样本中：")
print(f"   - 文本与图像情感一致: {count_text_image_consistent}")
print(f"   - 文本与图像情感不一致: {count_text_image_inconsistent}")

print(f"\n③ 在 final ≠ label 且 文图情感不一致 的样本中：")
print(f"   - label 与文本情感一致: {count_label_match_text}")
print(f"   - label 与图像情感一致: {count_label_match_image}")
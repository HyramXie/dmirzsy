import matplotlib.pyplot as plt
from collections import Counter
import json

error_types = []

with open('/home/cbf00006701/zsy/LLaMA-Factory/eval_train/qwen2.5vl-3b/15_wrong_classify.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        error_types.append(data['ErrorType'])

# 统计各类别的数量
counter = Counter(error_types)

# 准备绘图数据
labels = list(counter.keys())
sizes = list(counter.values())

# 绘制饼图并保存为 JPG 文件
plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
plt.title("Distribution of Error Types", fontsize=14)
plt.axis('equal')  # 确保饼图是圆形的

# 保存为 JPG 文件
plt.savefig('/home/cbf00006701/zsy/LLaMA-Factory/eval_train/qwen2.5vl-3b/error_type_distribution15.jpg', format='jpg', dpi=300, bbox_inches='tight')
plt.close()  # 关闭图形以释放内存

print("饼图已保存为 error_type_distribution.jpg")
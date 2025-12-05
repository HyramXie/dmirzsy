# import json

# # 读取第一个数据集
# with open('/root/user/datasets/twitter2017/w0.json', 'r') as f:
#     dataset1 = json.load(f)

# # 读取第二个数据集
# with open('/root/user/datasets/twitter2017/test+image_context.json', 'r') as f:
#     dataset2 = json.load(f)

# # 创建一个以 image_id 为 key 的字典，方便查找
# dataset2_dict = {item['image_id']: item for item in dataset2}

# # 合并两个数据集
# merged_dataset = []

# for item1 in dataset1:
#     image_id = item1['image_id']
#     if image_id in dataset2_dict:
#         # 合并两个字典
#         merged_item = {**item1, **dataset2_dict[image_id]}
#         merged_dataset.append(merged_item)

# # 保存合并后的数据集
# with open('/root/user/datasets/twitter2017/merged_dataset.json', 'w') as f:
#     json.dump(merged_dataset, f, indent=2)

# print("合并完成，结果已保存为 merged_dataset.json")


# -----------------------------------------------------------------------
import json
from sklearn.metrics import accuracy_score, f1_score

# 加载数据集
with open('/root/user/datasets/twitter2017/merged_dataset_2.json', 'r') as f:
    data = json.load(f)

# 定义转换函数
def model_output_to_label(output):
    mapping = {
        "Positive": 1,
        "Neutral": 0,
        "Negative": -1
    }
    return mapping[output]

# 提取真实标签和预测标签
y_true = []
y_pred = []

for item in data:
    # 真实标签（注意：原始数据是字符串，需要转为整数）
    true_label = int(item["sentiment"])
    # 模型预测标签
    pred_label = model_output_to_label(item["model_output"])
    
    y_true.append(true_label)
    y_pred.append(pred_label)

# 计算准确率和宏F1
acc = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average='macro')

# 输出结果
print(f"Accuracy: {acc:.4f}")
print(f"Macro F1 Score: {macro_f1:.4f}")
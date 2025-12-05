# import json

# with open("/home/cbf00006701/zsy/datasets/twitter2015/test.json", "r", encoding="utf-8") as f:
#     data = json.load(f)  # ← 正确地把 JSON 字符串转为 Python 列表

# converted_data = []

# # 构建目标格式（不在问题中暴露 aspect）
# for item in data:
#     text = item["text"]
#     aspect = item["aspect"]
#     sentiment = item["sentiment"]
#     image_id = item["image_id"]
    
#     question = "This post contains the following text: '{text}'. Based on both the image and this sentence, extract all (aspect, sentiment) pairs expressed. An'aspect' is an entity or topic mentioned, and 'sentiment' can be -1, 0, or 1 for negative, neutral, or positive attitudes respectively."
#     question = question.replace('{text}', text)
#     new_item = {
#         "messages": [
#             {
#                 "content": f"<image> {question}",
#                 "role": "user"
#             },
#             {
#                 "content": f"({aspect},{sentiment})",
#                 "role": "assistant"
#             }
#         ],
#         "images": [
#             f"twitter2015_images/{image_id}"
#         ]
#     }
    
#     converted_data.append(new_item)


# # 输出结果
# with open("/home/cbf00006701/zsy/LLaMA-Factory/data/2015/test.json", "w", encoding="utf-8") as f:
#     json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
# import json
# from collections import defaultdict

# with open("/home/cbf00006701/zsy/SIEVE-main/results_test/2015_cap_test.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# # 使用defaultdict自动分组相同text的条目
# text_groups = defaultdict(list)
# for item in data:
#     text_groups[item["text"]].append(item)
# #     text_groups[item["generatedtext"]].append(item)
# # for item in data:
# #     text = item["text"]
#     generatedtext = item["generatedtext"]
# #     aspect = item["aspect"]
# #     sentiment = item["sentiment"]
# #     image_id = item["image_id"]

# converted_data = []

# for text, items in text_groups.items():
#     # 收集所有aspect-sentiment对
#     pairs = [f"({item['aspect']},{item['sentiment']})" for item in items]
    
#     # 收集所有关联的image_id（去重）
#     unique_images = list({f"twitter2015_images/{item['image_id']}" for item in items})
    
#     # 构建问答模板
#     # question = f"This post contains the following text: '{text}'. Based on both the image and this sentence, extract all (aspect, sentiment) pairs expressed. An'aspect' is an entity or topic mentioned, and 'sentiment' can be -1, 0, or 1 for negative, neutral, or positive attitudes respectively."
#     question = f"You are an assistant for multimodal sentiment analysis. Given an image, its caption, and an associated text, extract all (aspect, sentiment) pairs expressed. An'aspect' is an entity or topic mentioned, and 'sentiment' can be -1, 0, or 1 for negative, neutral, or positive attitudes respectively. Inputs:Image: <image>; Caption:'{generatedtext}'; Text:'{text}'" # type: ignore
   

#     new_item = {
#         "messages": [
#             {
#                 "content": f"{question}",
#                 "role": "user"
#             },
#             {
#                 "content": ",".join(pairs),
#                 "role": "assistant"
#             }
#         ],
#         "images": unique_images
#     }
    
#     converted_data.append(new_item)

# # 保存结果
# with open("/home/cbf00006701/zsy/LLaMA-Factory/data/2015/test+cap.json", "w", encoding="utf-8") as f:
#     json.dump(converted_data, f, indent=2, ensure_ascii=False)



import json

# 假设你从一个 JSON 文件中读取原始数据
with open("/home/cbf00006701/zsy/datasets/twitter2015/test.json", "r", encoding="utf-8") as f:
    data = json.load(f)

converted_data = []

for item in data:
    text = item["text"]
    aspect = item["aspect"]
    sentiment = item["sentiment"]
    image_id = item["image_id"]
    
    # 构建包含原始文本的问题
    # question = f"Based on the image and the text: '{text}', what is the sentiment expressed towards '{aspect}'? The 'sentiment' can be -1, 0, or 1 for negative, neutral, or positive attitudes respectively."
    # user_prompt = (
    #     f"<image>Given the image and text: \"{text}\", "
    #     f"what is the sentiment toward the aspect \"{aspect}\"? Please analyze step by step."
    # )
    user_prompt = f'<image>What is the sentiment toward the aspect "{item["aspect"]}" in the text: {item["text"]}?'

    # 构建目标格式
    new_item = {
        "messages": [
            {
                "content": f"<image> {user_prompt}",
                "role": "user"
            },
            {
                "content": f"{sentiment}",
                "role": "assistant"
            }
        ],
        "images": [
            f"twitter2015_images/{image_id}"
        ]
    }
    
    converted_data.append(new_item)

# 将转换后的数据保存到新文件
with open("/home/cbf00006701/zsy/LLaMA-Factory/data/2015/test_con.json", "w", encoding="utf-8") as f:
    json.dump(converted_data, f, indent=2, ensure_ascii=False)

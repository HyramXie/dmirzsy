import json
import re

def convert_sentiment(sentiment_num):
    """转换情感标签数值到文本"""
    sentiment_map = {
        "1": "positive",
        "0": "neutral", 
        "-1": "negative"
    }
    return sentiment_map.get(str(sentiment_num), "neutral")

def extract_predict_from_jsonl(jsonl_line):
    """从JSONL格式中提取predict字段"""
    try:
        data = json.loads(jsonl_line)
        return data.get("predict", "")
    except json.JSONDecodeError:
        return ""

def process_datasets(jsonl_file_path, json_file_path, output_file_path):
    """
    批量处理两个数据集并合并
    
    参数:
    jsonl_file_path: 数据集1的JSONL文件路径
    json_file_path: 数据集2的JSON文件路径  
    output_file_path: 输出文件路径
    """
    
    # 读取JSONL文件（数据集1）
    jsonl_data = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                jsonl_data.append(json.loads(line))
    
    # 读取JSON文件（数据集2）
    with open(json_file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # 确保两个数据集长度相同
    if len(jsonl_data) != len(json_data):
        print(f"警告: 两个数据集长度不同 (JSONL: {len(jsonl_data)}, JSON: {len(json_data)})")
    
    # 合并数据
    merged_data = []
    min_length = min(len(jsonl_data), len(json_data))
    
    for i in range(min_length):
        json_item = json_data[i]
        jsonl_item = jsonl_data[i]
        
        # 创建新的合并项
        merged_item = {
            "text": json_item.get("text", ""),
            "aspect": json_item.get("aspect", ""),
            "sentiment": convert_sentiment(json_item.get("sentiment", "0")),
            "predict": jsonl_item.get("predict", ""),
            "image_id": json_item.get("image_id", ""),
            "image_context": json_item.get("image_context", "")
        }
        
        merged_data.append(merged_item)
    
    # 保存到输出文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)
    
    print(f"处理完成！共合并 {len(merged_data)} 条数据，保存到 {output_file_path}")

# 使用示例
if __name__ == "__main__":
    # 请替换为实际文件路径
    jsonl_file = "/home/cbf00006701/zsy/LLaMA-Factory/eval/qwen2.5vl-7b/sc_new_17_2.jsonl"  # 数据集1文件
    json_file = "/home/cbf00006701/zsy/datasets/twitter2017/test+image_context.json"    # 数据集2文件
    output_file = "/home/cbf00006701/zsy/LLaMA-Factory/eval/qwen2.5vl-7b/merged_dataset_sc_new_17.json"  # 输出文件
    
    process_datasets(jsonl_file, json_file, output_file)
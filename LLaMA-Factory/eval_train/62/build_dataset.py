import json
import re

def build_self_correction_samples(input_file, output_file):
    """
    构建自我纠正样本并转换为微调数据集格式
    
    Args:
        input_file: 输入的JSONL文件路径
        output_file: 输出的JSON文件路径
    """
    
    # 读取JSONL文件
    samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))
    
    # 构建微调数据集
    fine_tuning_data = []
    
    for sample in samples:
        # 构建用户消息
        user_message = sample["prompt"]
        
        # 构建自我纠正的助手消息
        # 格式: [错误预测] + 反思 + 改进 + 正确答案
        correction_message = (
            f"{sample['predict']}. Sorry, I made a mistake. {sample['Reflection']} "
            f"{sample['Improvement']} The correct sentiment is {sample['label']}. "
            f"Answer: {sample['label']}"
        )
        
        # 创建对话格式
        conversation = {
            "messages": [
                {
                    "role": "user",
                    "content": user_message
                },
                {
                    "role": "assistant",
                    "content": correction_message
                }
            ]
        }
        
        fine_tuning_data.append(conversation)
    
    # 保存为JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(fine_tuning_data, f, indent=2, ensure_ascii=False)
    
    print(f"成功转换 {len(fine_tuning_data)} 条样本")
    print(f"已保存到: {output_file}")

def add_loss_mask_to_dataset(input_file, output_file, tokenizer=None):
    """
    为数据集添加loss mask信息（可选）
    如果不提供tokenizer，则只添加mask位置的文本标记
    
    Args:
        input_file: 输入的JSON文件路径
        output_file: 输出的JSON文件路径
        tokenizer: 可选的分词器，用于计算确切的token位置
    """
    
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    for item in dataset:
        assistant_content = item["messages"][1]["content"]
        
        # 找到错误预测部分的结束位置（第一个句号后）
        wrong_part_end = assistant_content.find('.') + 1
        if wrong_part_end == 0:  # 如果没有找到句号
            wrong_part_end = len(assistant_content.split()[0])  # 使用第一个单词
        
        # 添加mask信息
        item["loss_mask"] = {
            "wrong_part_end": wrong_part_end,
            "wrong_part_text": assistant_content[:wrong_part_end]
        }
        
        # 如果需要使用tokenizer计算确切的token位置
        if tokenizer:
            tokens = tokenizer.tokenize(assistant_content)
            wrong_tokens = tokenizer.tokenize(assistant_content[:wrong_part_end])
            item["loss_mask"]["wrong_token_count"] = len(wrong_tokens)
            item["loss_mask"]["total_token_count"] = len(tokens)
    
    # 保存带有mask信息的数据集
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"已添加loss mask信息到: {output_file}")

# 使用示例
if __name__ == "__main__":
    input_jsonl = "/root/user/zsy/LLaMA-Factory/eval_train/62/qwen2.5vl-3b/mvsa_explain.jsonl"  # 替换为您的输入文件路径
    output_json = "/root/user/zsy/LLaMA-Factory/eval_train/62/qwen2.5vl-3b/mvsa_fine_tuning_explain.json"  # 输出文件路径
    # masked_output_json = "/root/user/LLaMA-Factory/eval_train/15_fine_tuning_explain_with_mask.json"  # 带有mask信息的输出文件路径
    
    # 构建自我纠正样本
    build_self_correction_samples(input_jsonl, output_json)
    
    # 添加loss mask信息（可选）
    # 如果您有分词器，可以传入tokenizer参数
    # 例如: from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    add_loss_mask_to_dataset(output_json, masked_output_json)
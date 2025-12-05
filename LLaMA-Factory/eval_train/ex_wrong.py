import json

def normalize(s):
    """忽略大小写和空格进行标准化"""
    return s.strip().lower() if s else ""

# 输入文件路径
input_file = '/root/user/LLaMA-Factory/eval_train/llava1.5-7b/17.jsonl'  # 替换为你的输入文件名
# 输出文件路径（存储 predict 和 label 不一致的样本）
output_file = '/root/user/LLaMA-Factory/eval_train/llava1.5-7b/inconsistent_predictions_17.jsonl'

inconsistent_count = 0

with open(input_file, 'r', encoding='utf-8') as f_in, \
     open(output_file, 'w', encoding='utf-8') as f_out:
    
    for line in f_in:
        line = line.strip()
        if not line:
            continue
        
        try:
            data = json.loads(line)
            predict = data.get("predict")
            label = data.get("label")
            
            # 标准化并比较
            if normalize(predict) != normalize(label):
                # 写入不一致的数据
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                inconsistent_count += 1
                
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error parsing line: {line[:50]}... -> {e}")
            continue

print(f"Done! Found {inconsistent_count} inconsistent entries. Saved to {output_file}")
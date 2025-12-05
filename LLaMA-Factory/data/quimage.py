import json

def remove_images_from_json(input_file, output_file):
    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 如果是单个对象
    if isinstance(data, dict) and 'images' in data:
        del data['images']
    # 如果是对象列表
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and 'images' in item:
                del item['images']
    
    # 保存修改后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 使用示例
remove_images_from_json('/root/user/LLaMA-Factory/data/2015/t+i+c+target_15.json', '/root/user/LLaMA-Factory/data/2015/t+c+target_15.json')
import json
import os
from collections import defaultdict

def count_aspects_in_dataset(dataset_name, split):
    """
    统计指定数据集和划分中的单方面和多方面样本数量
    """
    file_path = f"/root/user/datasets/{dataset_name}/{split}.json"
    
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在")
        return 0, 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 使用字典统计每个文本出现的次数
    text_count = defaultdict(int)
    for item in data:
        text = item.get('text')
        text_count[text] += 1
    
    one_aspect_count = 0
    multi_aspect_count = 0
    
    for text, count in text_count.items():
        if count == 1:
            one_aspect_count += 1
        else:
            multi_aspect_count += 1
    
    return one_aspect_count, multi_aspect_count

def main():
    datasets = ["twitter2015", "twitter2017"]
    splits = ["train", "dev", "test"]
    
    print("Dataset Statistics for Twitter-2015 and Twitter-2017:")
    print("-" * 80)
    print(f"{'Dataset':<15} {'Split':<6} {'One Aspect':<12} {'Multi Aspects':<13}")
    print("-" * 80)
    
    for dataset in datasets:
        for split in splits:
            one_aspect, multi_aspect = count_aspects_in_dataset(dataset, split)
            print(f"{dataset:<15} {split:<6} {one_aspect:<12} {multi_aspect:<13}")
    
    # 计算总计
    print("-" * 80)
    print("Total Counts:")
    print("-" * 80)
    
    total_one_aspect_2015 = total_multi_aspect_2015 = 0
    total_one_aspect_2017 = total_multi_aspect_2017 = 0
    
    for split in splits:
        one_2015, multi_2015 = count_aspects_in_dataset("twitter2015", split)
        one_2017, multi_2017 = count_aspects_in_dataset("twitter2017", split)
        
        total_one_aspect_2015 += one_2015
        total_multi_aspect_2015 += multi_2015
        total_one_aspect_2017 += one_2017
        total_multi_aspect_2017 += multi_2017
    
    print(f"Twitter-2015 Total: {total_one_aspect_2015:<12} {total_multi_aspect_2015:<13}")
    print(f"Twitter-2017 Total: {total_one_aspect_2017:<12} {total_multi_aspect_2017:<13}")

if __name__ == "__main__":
    main()
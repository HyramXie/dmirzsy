import json
import os
from openai import OpenAI
from tqdm import tqdm

# 初始化 DeepSeek API
client = OpenAI(
    api_key="sk-100b432f23414ba8a71a21edd60f7a99",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 或你的代理地址
)

input_file = "/home/cbf00006701/zsy/LLaMA-Factory/eval_train/qwen2.5vl-3b/inconsistent_predictions_15.jsonl"
output_file = "/home/cbf00006701/zsy/LLaMA-Factory/eval_train/qwen2.5vl-3b/15_explain_new.jsonl"

# 获取已处理的 prompt 列表（用于断点续跑）
processed_prompts = set()
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                processed_prompts.add(data['prompt'])
            except:
                continue

# 加载全部输入数据
with open(input_file, 'r', encoding='utf-8') as f:
    all_data = [json.loads(line) for line in f]

# Few-Shot 示例（从你提供的 JSON 数据中提取并格式化）
few_shot_examples = """
Here are some examples of correct Reflection and Improvement responses:

Example 1:
Text Prompt: <image> Given the image and its description: "A large crowd gathers at a concert, illuminated by stage lights, with an uplifting message displayed on a screen, creating a sense of shared joy and community.", analyze the following text: 'Life is what happens while you are busy planning other things <target>John Lennon</target>'. What is the sentiment expressed towards the entity marked with <target>?
Please answer with one word: 'positive', 'neutral', or 'negative'.
Model Prediction: neutral
Ground Truth Label: positive
Reflection: The model incorrectly labeled the sentiment as 'neutral' by taking the text literally and missing the positive, playful comparison in text The peace sign, round glasses, and relaxed vibe reference John Lennon’s iconic, peace-associated image—indicating admiration.
Improvement: The model should better recognize figurative language and cultural references. Comparing oneself to a positively viewed figure like Lennon typically signals positive sentiment. Leveraging contextual knowledge and detecting metaphorical or humorous tone can improve accuracy beyond literal interpretation.

Example 2:
Text Prompt: <image> Given the image and its description: "A focused athlete in a blue jersey and headband stands on a sports field, appearing attentive and ready for action.", analyze the following text: 'RT @ CIothesPorn : USA defender <target>Julie Johnston</target> is perfection'. What is the sentiment expressed towards the entity marked with <target>?
Please answer with one word: 'positive', 'neutral', or 'negative'.
Model Prediction: positive
Ground Truth Label: positive
Reflection: The model erred by attributing the positive sentiment of 'Julie Johnston is perfection' to the target **USA**, though the praise is directed solely at the player. It failed to distinguish sentiment toward the individual from the neutral mention of her team/nationality.
Improvement: The model should better disambiguate sentiment targets in multimodal content, recognizing that mentioning an entity (USA) as an attribute does not imply sentiment toward it. Separating sentiment toward individuals from associated groups will prevent misattribution and ensure accurate neutral classification for non-targeted entities.

Example 3:
Text Prompt: <image> Given the image and its description: "The image is a table showing the results of the last 10 league games between Stoke City and Chelsea at Stoke, highlighting Chelsea's dominance with six wins, one draw, and three losses.", analyze the following text: 'RT @ chelseafc : Our last 10 league games away at <target>Stoke City</target> . . . # CFC'. What is the sentiment expressed towards the entity marked with <target>?
Please answer with one word: 'positive', 'neutral', or 'negative'.
Model Prediction: neutral
Ground Truth Label: positive
Reflection: The model erred by focusing only on the neutral tweet text, ignoring the image’s stats showing Chelsea’s dominance (6 wins, 1 draw vs. Stoke), which frame #CFC positively.
Improvement: Integrate multimodal cues—like win counts and performance visuals—to correctly infer positive sentiment toward #CFC, not neutral.
"""

for entry in tqdm(all_data):
    prompt = entry['prompt']
    predict = entry['predict']
    label = entry['label']

    # 跳过已经处理过的样本
    if prompt in processed_prompts:
        continue

    # 构建提问内容（提示词）
    system_prompt = "You are an expert at diagnosing and improving LLM predictions."

    user_prompt = f"""
{few_shot_examples}

Now consider the following case:
Text Prompt: {prompt}
Model Prediction: {predict}
Ground Truth Label: {label}

Please provide:
1. Reflection: Where did the reasoning go wrong, and why?
2. Improvement: How can the model improve its reasoning to get the correct answer?

Respond in this format:
Reflection: ...
Improvement: ...
"""

    try:
        # 调用 DeepSeek API
        response = client.chat.completions.create(
            model="deepseek-v3",  # 替换为你实际用的 deepseek-v3 模型名
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=256
        )

        reply = response.choices[0].message.content.strip()

        # 解析模型输出（建议格式清晰，防止出错）
        reflection = ""
        improvement = ""
        for line in reply.splitlines():
            if line.startswith("Reflection:"):
                reflection = line.replace("Reflection:", "").strip()
            elif line.startswith("Improvement:"):
                improvement = line.replace("Improvement:", "").strip()

        # 构造输出结构
        result = {
            "prompt": prompt,
            "predict": predict,
            "label": label,
            "Reflection": reflection,
            "Improvement": improvement
        }

        # 追加保存
        with open(output_file, 'a', encoding='utf-8') as out_f:
            out_f.write(json.dumps(result, ensure_ascii=False) + '\n')

        print(f"Processed: {prompt[:50]}...")

    except Exception as e:
        print(f"Error processing prompt: {prompt[:50]}... Error: {e}")
        continue
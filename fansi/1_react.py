import os
import json
import base64
import mimetypes
import time
from tqdm import tqdm
from openai import OpenAI

# 加载数据集
with open("/root/user/zsy/LLaMA-Factory/data/2017/t+i+c+target_17.json", "r") as f:
    dataset = json.load(f)

results = []
max_new_tokens = 64
model_name = "qwen3-vl-plus"
client = OpenAI(
    api_key="sk-775a41fcc6cd4ed0909e25a7148be224",
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    timeout=60,
    max_retries=3,
)

def chat_with_retry(client, model_name, system_prompt, user_contents, max_new_tokens, temperature, top_p, retries=3, backoff=2):
    last_exc = None
    for i in range(retries):
        try:
            return client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_contents}
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens
            )
        except Exception as e:
            last_exc = e
            time.sleep(backoff ** i)
    raise last_exc

for example in tqdm(dataset):
    try:
        user_msg = next(m for m in example["messages"] if m["role"] == "user")
        user_text = user_msg["content"]

        image_path = None
        if isinstance(example.get("images"), list) and len(example["images"]) > 0:
            image_path = example["images"][0]
        
        image_content = None
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as img_f:
                b64 = base64.b64encode(img_f.read()).decode("utf-8")
            mime, _ = mimetypes.guess_type(image_path)
            if not mime:
                mime = "image/jpeg"
            image_content = {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"}
            }

        prompt = (
            f"{user_text}\n\n"
            "First, state your sentiment judgment as one word: Positive, Neutral, or Negative. "
            "Then briefly explain why. "
            "Format your answer exactly as: \"Sentiment: Explanation\"."
        )

        system_prompt = "You are a helpful assistant."

        user_contents = [{"type": "text", "text": prompt}]
        if image_content:
            user_contents.insert(0, image_content)

        response = chat_with_retry(
            client,
            model_name,
            system_prompt,
            user_contents,
            max_new_tokens,
            0.8,
            0.9,
            retries=3,
            backoff=2,
        )

        reply = response.choices[0].message.content.strip()

        results.append({
            "messages": example["messages"],
            "images": example.get("images", []),
            "model_outputs": [reply]
        })

    except Exception as e:
        print(f"Error: {e}")
        results.append({
            "messages": example.get("messages", []),
            "images": example.get("images", []),
            "model_outputs": [f"Error: {e}"]
        })

# 保存结果
save_path = "/root/user/zsy/fansi/1-react.json"
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✅ 推理完成，保存至 {save_path}")

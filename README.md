import os
import glob
import requests

API_KEY = os.getenv('DEEPSEEK_API_KEY', 'sk-be29bde69c8d441bae27a6df578b4c44')  # 请替换为你的API Key

POLISH_PROMPT = """
你是一名中国法院资深法官，请对以下判决书草稿进行润色和规范化，只做表达和格式优化，不得增加任何原文中未出现的事实内容。可以适当补充司法文书常用套话，使判决书更正式、更详细且更规范。

【判决书草稿】：
{content}

【输出要求】：
- 只做润色和规范化，不得虚构或补充事实
- 可补充司法惯用语和标准格式
- 输出完整判决书文本
"""

def polish_judgement(text, api_key=API_KEY):
    prompt = POLISH_PROMPT.format(content=text)
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是中国法院法官，擅长撰写规范判决书。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 3000
    }
    try:
        response = requests.post(url, json=data, headers=headers, timeout=120)
        resp_json = response.json()
        print("[DEBUG] API response:", resp_json)  # 新增debug输出
        content = resp_json["choices"][0]["message"]["content"]
        if content.strip().startswith('```'):
            content = content.strip().lstrip('`').rstrip('`').strip()
            content = content.split('```')[-1] if '```' in content else content
        # 对比原文和润色结果
        if content.strip() == text.strip():
            print("[WARNING] 润色结果与原文无差异！")
        return content
    except Exception as e:
        print(f"润色API调用出错: {str(e)}")
        return text

def polish_all_judgements(input_dir, output_dir=None):
    if output_dir is None:
        output_dir = input_dir
    files = glob.glob(os.path.join(input_dir, '*_judgement_draft.md'))
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"正在润色: {os.path.basename(file)}")
        polished = polish_judgement(text)
        out_path = os.path.join(output_dir, os.path.basename(file).replace('_draft.md', '_polished.md'))
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(polished)
        print(f"已保存润色后判决书: {out_path}")

if __name__ == "__main__":
    # 默认处理 src/result 目录下所有 *_judgement_draft.md
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../result'))
    polish_all_judgements(base_dir)

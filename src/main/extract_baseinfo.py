import os
import json
from langchain_community.document_loaders import PyPDFLoader
import requests

def extract_party_info_llm(text, api_key):
    prompt = f"""
请从以下起诉状内容中仅提取原告和被告的详细信息，输出JSON格式：
- 原告和被告均为数组，每个人包含：姓名、性别、民族、生日、住址、身份证号、联系电话。
- 字段必须齐全，无则填空字符串。
- 只输出JSON，不要解释。

起诉状内容：
{text}

输出格式：
{{
  "原告": [{{"姓名": "", "性别": "", "民族": "", "生日": "", "住址": "", "身份证号": "", "联系电话": ""}}, ...],
  "被告": [{{...}}, ...]
}}
"""
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 1024
    }
    try:
        response = requests.post(url, json=data, headers=headers, timeout=60)
        content = response.json()["choices"][0]["message"]["content"]
        if content.strip().startswith('```'):
            content = content.strip().lstrip('`json').lstrip('`').rstrip('`').strip()
            content = content.split('```')[-1] if '```' in content else content
        return json.loads(content)
    except Exception as e:
        return {"error": str(e)}

def main():
    # 使用相对路径访问 code/data/data2
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/data2'))
    api_key = "sk-c1d6cb5fc75c4de5ba19fa2b3f1143a1"  # 可替换为你的API Key
    result = {}
    for case in os.listdir(base_dir):
        case_path = os.path.join(base_dir, case)
        pdf_path = os.path.join(case_path, "起诉状.pdf")
        if os.path.exists(pdf_path):
            try:
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                text = "\n".join([p.page_content for p in docs])
                info = extract_party_info_llm(text, api_key)
                result[case] = info
            except Exception as e:
                result[case] = {"error": str(e)}
        else:
            result[case] = {"error": "起诉状.pdf不存在"}
    # 结果保存到 src/result/baseinformation.json
    result_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../result'))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    output_path = os.path.join(result_dir, "baseinformation.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"已用大模型结构化抽取并保存到 {output_path}")

if __name__ == "__main__":
    main()
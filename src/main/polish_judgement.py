import os
import glob
import requests
from docx import Document  # 新增依赖

def read_case_summary_from_docx(docx_path):
    try:
        doc = Document(docx_path)
        return '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        print(f"[WARNING] 读取案情摘要失败: {docx_path}, {e}")
        return None

API_KEY = os.getenv('DEEPSEEK_API_KEY', 'sk-be29bde69c8d441bae27a6df578b4c44')  # 请替换为你的API Key

POLISH_PROMPT = """
你是一名中国法院资深法官，请对以下判决书草稿进行润色和规范化，只做表达和格式优化，不得增加任何未出现的事实内容。可以适当补充司法文书常用套话，使判决书更正式、更详细且更规范。

【案情摘要】：
{summary}

【判决书草稿】：
{content}

【输出要求】：
- 重要：只做润色和规范化，不得虚构或补充事实
- 可补充司法惯用语和标准格式
- 输出完整判决书文本
"""

def polish_judgement(text, case_summary=None, api_key=API_KEY):
    if case_summary:
        prompt = POLISH_PROMPT.format(summary=case_summary, content=text)
    else:
        prompt = POLISH_PROMPT.format(summary="（无案情摘要）", content=text)
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
        print("[DEBUG] API response:", resp_json)
        content = resp_json["choices"][0]["message"]["content"]
        if content.strip().startswith('```'):
            content = content.strip().lstrip('`').rstrip('`').strip()
            content = content.split('```')[-1] if '```' in content else content
        if content.strip() == text.strip():
            print("[WARNING] 润色结果与原文无差异！")
        return content
    except Exception as e:
        print(f"润色API调用出错: {str(e)}")
        return text

def polish_all_judgements(input_dir, attachment_dir=None, output_dir=None):
    if output_dir is None:
        output_dir = input_dir
    files = glob.glob(os.path.join(input_dir, '*_judgement_draft.md'))
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
        case_name = os.path.basename(file).replace('_judgement_draft.md', '')
        case_summary = None
        # 只查找案情摘要.docx
        if attachment_dir:
            # 只在二级子目录下查找案情摘要.docx
            for subdir in os.listdir(attachment_dir):
                sub_path = os.path.join(attachment_dir, subdir)
                if os.path.isdir(sub_path) and case_name in subdir:
                    docx_path = os.path.join(sub_path, '案情摘要.docx')
                    if os.path.exists(docx_path):
                        case_summary = read_case_summary_from_docx(docx_path)
                        break
        print(f"正在润色: {os.path.basename(file)}")
        polished = polish_judgement(text, case_summary=case_summary)
        out_path = os.path.join(output_dir, os.path.basename(file).replace('_draft.md', '_polished.md'))
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(polished)
        print(f"已保存润色后判决书: {out_path}")

if __name__ == "__main__":
    # 默认处理 src/result 目录下所有 *_judgement_draft.md，并自动查找 data/附件1 下案情摘要.docx
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../result'))
    attachment_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/附件1'))
    polish_all_judgements(base_dir, attachment_dir=attachment_dir)

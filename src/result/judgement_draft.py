import json
import os

def json_to_markdown(json_path, md_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lines = []
    for case_type, content in data.items():
        lines.append(f"# {case_type} 判决书初稿\n")
        if isinstance(content, str):
            lines.append(content.strip() + "\n\n---\n")
        else:
            # 若内容为结构化数据
            lines.append(json.dumps(content, ensure_ascii=False, indent=2) + "\n\n---\n")
    with open(md_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"已生成 {md_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    json_path = os.path.join(base_dir, "judgement_draft.json")
    md_path = os.path.join(base_dir, "judgement_draft.md")
    json_to_markdown(json_path, md_path)

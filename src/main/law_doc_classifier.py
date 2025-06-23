import os
import shutil
from docx import Document

def extract_law_names(doc_path, keywords):
    """
    提取文档中出现的法条关键词（如“合同法”“婚姻法”等）。
    """
    doc = Document(doc_path)
    text = '\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
    found = set()
    for kw in keywords:
        if kw in text:
            found.add(kw)
    return found

def classify_docs_by_law(case_dir, out_dir, keywords):
    """
    按法条关键词对docx文档分类，包含某关键词的文档复制到对应子文件夹。
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for fname in os.listdir(case_dir):
        if fname.endswith('.docx'):
            fpath = os.path.join(case_dir, fname)
            found_laws = extract_law_names(fpath, keywords)
            for law in found_laws:
                target_dir = os.path.join(out_dir, law)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                shutil.copy2(fpath, os.path.join(target_dir, fname))
    print('分类完成。')

def main():
    # 你可以自定义需要分类的法条关键词
    keywords = ['合同法', '婚姻法', '民法典', '劳动法', '公司法']
    case_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/附件2'))
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/分类结果'))
    classify_docs_by_law(case_dir, out_dir, keywords)

if __name__ == '__main__':
    main()

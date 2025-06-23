from docx import Document
import os

def read_all_laws(law_dir, max_length=20000):
    """
    读取law_dir下所有docx法规文件，拼接为一段大文本，超长时按文件均分截断。
    """
    law_files = [fname for fname in os.listdir(law_dir) if fname.endswith('.docx')]
    file_count = len(law_files)
    if file_count == 0:
        return ''
    # 为每个文件分配最大可用长度
    per_file_max = max_length // file_count
    all_texts = []
    for fname in law_files:
        fpath = os.path.join(law_dir, fname)
        try:
            doc = Document(fpath)
            text = '\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
            # 截断每个文件的内容
            if len(text) > per_file_max:
                text = text[:per_file_max] + '\n...（已截断）'
            all_texts.append(f"【{fname}】\n" + text)
        except Exception:
            continue
    full_text = '\n'.join(all_texts)
    # 最终整体截断，防止极端情况下超长
    if len(full_text) > max_length:
        full_text = full_text[:max_length] + '\n...（已截断）'
    return full_text

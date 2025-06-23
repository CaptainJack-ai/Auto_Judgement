import os
import re
import pandas as pd
import numpy as np
from collections import defaultdict
from docx import Document

def extract_law_citations_from_judgement(text):
    """
    从判决书结尾“依照……判决如下”段落，提取所有法条及其条号，支持同一法条多个条号归并。
    返回：{法条名称: [条号, ...], ...}
    """
    # 匹配“依照……判决如下”段
    match = re.search(r'依照(.+?)(判决如下|裁定如下|判决为|判决：|裁定：)', text, re.S)
    if not match:
        return {}
    law_block = match.group(1)
    # 匹配所有“《法条名称》条号”组合，支持多个条号
    law_pattern = r'([《<][^》>]+[》>])((第[一二三四五六七八九十百千万0-9]+条(?:、|,|，)?)+)'
    article_pattern = r'第[一二三四五六七八九十百千万0-9]+条'
    result = defaultdict(list)
    for m in re.finditer(law_pattern, law_block):
        law_name = m.group(1)
        articles = re.findall(article_pattern, m.group(2))
        result[law_name].extend(articles)
    # 去重并排序
    for k in result:
        result[k] = sorted(set(result[k]), key=result[k].index)
    return result

def extract_law_articles_from_docx(doc_path):
    """
    优先从判决书结尾“依照……判决如下”段落提取法条及条号，若无则回退全文正则提取。
    返回：['《法条名称》第x条', ...]
    """
    doc = Document(doc_path)
    text = '\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
    # 优先判决书结尾归并提取
    law_dict = extract_law_citations_from_judgement(text)
    results = set()
    if law_dict:
        for law, articles in law_dict.items():
            if articles:
                results.add(f"{law}{'、'.join(articles)}")
            else:
                results.add(law)
        print(f"[{os.path.basename(doc_path)}] 判决结尾提取法条：", results)
        return list(results)
    # 若无“依照”段，则回退全文正则提取
    law_name_pattern = r'[《<][^》>]+[》>]'
    article_pattern = r'第[一二三四五六七八九十百千万0-9]+条(?:第[一二三四五六七八九十百千万0-9]+款)?(?:第[一二三四五六七八九十百千万0-9]+项)?'
    pattern = rf'({law_name_pattern})(（[^）]*）)?((?:{article_pattern}[、,，]*)+)'
    for m in re.finditer(pattern, text):
        law_name = m.group(1)
        if m.group(2):
            law_name += m.group(2)
        articles = re.findall(article_pattern, m.group(3))
        for art in articles:
            results.add(f'{law_name}{art}')
    print(f"[{os.path.basename(doc_path)}] 全文正则提取法条：", results)
    return list(results)

def build_case_law_matrix(case_dir):
    """
    遍历case_dir下所有docx案例，构建案例-法条二值矩阵。
    返回：DataFrame（行=案例文件名，列=法条，值=0/1）
    """
    law_set = set()
    case_law_dict = {}
    for fname in os.listdir(case_dir):
        if fname.endswith('.docx'):
            fpath = os.path.join(case_dir, fname)
            articles = extract_law_articles_from_docx(fpath)
            law_set.update(articles)
            case_law_dict[fname] = set(articles)
    law_list = sorted(law_set)
    data = []
    for fname in case_law_dict:
        row = [1 if law in case_law_dict[fname] else 0 for law in law_list]
        data.append(row)
    df = pd.DataFrame(data, index=case_law_dict.keys(), columns=law_list)
    return df

def compute_law_prob_and_corr(df):
    """
    计算每个法条出现概率和法条相关系数矩阵。
    返回：prob(Series), corr(DataFrame)
    """
    prob = df.sum(axis=0) / df.shape[0]
    corr = df.corr(method='pearson')
    return prob, corr

def main():
    case_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/附件2'))
    result_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../result'))
    os.makedirs(result_dir, exist_ok=True)
    df = build_case_law_matrix(case_dir)
    prob, corr = compute_law_prob_and_corr(df)
    # 保存结果到src/result目录
    df.to_csv(os.path.join(result_dir, 'case_law_matrix.csv'), encoding='utf-8-sig')
    prob.to_csv(os.path.join(result_dir, 'law_article_prob.csv'), encoding='utf-8-sig')
    corr.to_csv(os.path.join(result_dir, 'law_article_corr.csv'), encoding='utf-8-sig')
    print('案例-法条矩阵、法条概率、法条相关系数已保存到src/result目录。')

if __name__ == '__main__':
    main()

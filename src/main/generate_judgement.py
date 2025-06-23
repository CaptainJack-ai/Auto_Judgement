import json
from openai import OpenAI
import os
from law_utils import read_all_laws
import pandas as pd

def call_llm_generate_judgement(case_info, similar_cases, law_text, model_api_key="sk-be29bde69c8d441bae27a6df578b4c44"):
    client = OpenAI(api_key=model_api_key or os.getenv('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")
    # 证据明细与案例要点整理
    evidence_list = []
    for key in ["案情摘要关键信息", "起诉状关键信息", "庭审笔录关键信息"]:
        if key in case_info and "证据" in case_info[key]:
            for ev in case_info[key]["证据"]:
                evidence_list.append(ev)
    evidence_str = '\n'.join([f"- {ev}" for ev in evidence_list]) if evidence_list else "无明确证据明细。"
    # 案例要点
    similar_points = '\n'.join([f"- {c.get('裁判要点', c.get('案情摘要', ''))}" for c in similar_cases if c])
    # 法条推理
    law_reasoning = '\n'.join([f"{item['条文名称']}：{item['条文内容']}\n推理：{item.get('逻辑推断','')}" for item in case_info.get("相关法律条文及依据",[])])
    prompt = f"""
你是一名中国法院法官，请根据以下案件材料，严格按照中国法院判决书的专业格式和写作规范，分结构、分要素、分条理地详细撰写判决书。要求：
1. 结构分明，分段输出：案由、当事人信息、诉讼请求、争议焦点、查明事实、举证责任分析、法律适用与裁判理由、判决结果。
2. 每一部分内容详实、逻辑清晰，引用法律条文要准确，判决用语规范。
3. 争议焦点要明确列举，举证责任要结合事实和证据分析。
4. 查明事实部分请结合如下证据明细分条列举：\n{evidence_str}
5. 法律适用部分要结合案件事实，逐条引用相关法条并说明理由。可参考如下法条推理：\n{law_reasoning}
6. 可参考如下相似案例裁判要点：\n{similar_points}
7. 判决结果要有法律依据，格式规范。

【案件关键信息】：\n{json.dumps(case_info, ensure_ascii=False, indent=2)}\n
【相关案例摘要】：\n{json.dumps(similar_cases, ensure_ascii=False, indent=2)}\n
【可参考法律法规条文】：\n{law_text}\n
请严格分结构输出：
一、案由
二、当事人信息
三、诉讼请求
四、争议焦点
五、查明事实
六、举证责任分析
七、法律适用与裁判理由
八、判决结果
"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": "你是中国法院法官，擅长撰写规范判决书。"},
                  {"role": "user", "content": prompt}],
        temperature=0.2,
        stream=False
    )
    content = response.choices[0].message.content
    # 自动去除 markdown 代码块标记
    if content.strip().startswith('```'):
        content = content.strip().lstrip('`').rstrip('`').strip()
        content = content.split('```')[-1] if '```' in content else content
    return content

def merge_party_info(case_info, baseinfo):
    # 合并详细当事人信息到案件关键信息
    for key in ["案情摘要关键信息", "起诉状关键信息", "庭审笔录关键信息"]:
        if key in case_info and "当事人信息" in case_info[key]:
            for role in ["原告", "被告"]:
                # 如果baseinfo有详细信息则覆盖
                if baseinfo and role in baseinfo and baseinfo[role]:
                    case_info[key]["当事人信息"][role] = baseinfo[role]
    # 也可直接在case_info顶层加一份详细当事人信息
    case_info["详细当事人信息"] = baseinfo
    return case_info

def get_high_freq_laws_from_csv(similar_cases, matrix_path, topn=5):
    """
    根据相似案例编号，统计高频法条。
    """
    df = pd.read_csv(matrix_path, index_col=0)
    # 只取前topn个相似案例编号
    case_ids = [c["编号"].replace('.docx','') for c in similar_cases[:topn] if "编号" in c]
    # 兼容csv索引可能不带.docx
    case_ids = [cid for cid in case_ids if cid in df.index]
    if not case_ids:
        return []
    sub_df = df.loc[case_ids]
    freq = sub_df.sum(axis=0).sort_values(ascending=False)
    return list(freq[freq>0].index)

def main():
    # 读取最终版case_law_results.json（实际在src/background目录）
    case_law_path = os.path.join(os.path.dirname(__file__), "..", "background", "最终版case_law_results.json")
    with open(case_law_path, "r", encoding="utf-8") as f:
        case_law_results = json.load(f)
    # 读取result.json、similar_cases.json、baseinformation.json、case_law_matrix.csv都从src/result目录
    result_dir = os.path.join(os.path.dirname(__file__), "..", "result")
    with open(os.path.join(result_dir, "result.json"), "r", encoding="utf-8") as f:
        case_infos = json.load(f)
    with open(os.path.join(result_dir, "similar_cases.json"), "r", encoding="utf-8") as f:
        similar_cases = json.load(f)
    with open(os.path.join(result_dir, "baseinformation.json"), "r", encoding="utf-8") as f:
        baseinfo_all = json.load(f)
    matrix_path = os.path.join(result_dir, "case_law_matrix.csv")
    results = {}
    for case_type in case_law_results:
        case_info = dict(case_law_results[case_type])
        if case_type in case_infos and "案情摘要关键信息" in case_infos[case_type]:
            case_info["案情摘要关键信息"] = case_infos[case_type]["案情摘要关键信息"]
        sim_cases = similar_cases.get(f"{case_type}.docx", []) or similar_cases.get(case_type, [])
        baseinfo = baseinfo_all.get(case_type, {})
        case_info_with_party = merge_party_info(case_info, baseinfo)
        law_names = get_high_freq_laws_from_csv(sim_cases, matrix_path, topn=5)
        law_text = '\n'.join(law_names)
        judgement = call_llm_generate_judgement(case_info_with_party, sim_cases[:5], law_text)
        results[case_type] = judgement
        print(f"[{case_type}] 判决书初稿生成完毕\n")
    # 结果保存到src/result/judgement_draft.json
    output_path = os.path.join(result_dir, 'judgement_draft.json')
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"全部判决书初稿已保存为 {output_path}")

if __name__ == "__main__":
    main()

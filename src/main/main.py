import subprocess
import sys
import os
    
# 依次自动运行司法案例分析全流程的各个脚本
# 脚本顺序：信息抽取 -> 向量检索 -> 相似案例匹配 -> 法条分析 -> 文档分类 -> 判决书生成 -> Markdown导出
SCRIPTS = [
    # 1. 案件关键信息抽取
   os.path.abspath(os.path.join(os.path.dirname(__file__), '../main/extract_baseinfo.py')),
    # 2. PDF/Word批量向量化与检索
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../main/FaissExtractor.py')),
    # 3. 相似案例向量匹配
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../main/search_similar_cases.py')),
    # 4. 法条提取与矩阵分析
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../main/law_article_analysis.py')),
    # 5. 按法条关键词分类文档
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../main/law_doc_classifier.py')),
    # 6. 判决书自动生成
    os.path.join("..", "main", "generate_judgement.py"), 
    # 7. 判决书初稿导出为Markdown
    os.path.join("..", "result", "judgement_draft.py"),
    # 8. 判决书润色
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../main/polish_judgement.py'))
]

def run_script(script_path):
    """
    调用子进程运行单个Python脚本，并输出运行信息。
    """
    print(f'>>> 正在运行: {os.path.basename(script_path)}')
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"{script_path} 执行失败")

def main():
    """
    依次运行全流程各脚本，自动串联司法案例分析与判决书生成。
    """
    for script in SCRIPTS:
        run_script(script)
    print('全部流程已顺利完成！')

if __name__ == '__main__':
    main()

# 基于RAG结构的信息比对与检索算法

# 项目简介
- 本程序提供一种Faiss相关库对用户提供的文件(pdf)进行分块索引化，然后调用DeepSeek API根据用户需求加以分析的模式。通过更改提供给DeepSeek的提示词，可对不同的附件进行不同深度的检索与分析。

# 使用方法
- 修改search_relevant_laws.py主函数中的案件摘要文件夹和法律条文文件夹
- 运行search_relevant_laws.py，结果输出为case_law_results.json

# 环境要求
- Python 3.8+
- requirements.txt


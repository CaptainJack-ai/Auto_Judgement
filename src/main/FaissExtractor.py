import os
import json
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import requests

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'



# 加载PDF文件并构建faiss向量索引
def build_faiss_index(folder_path):

    # 加载PDF
    loader = DirectoryLoader(
        path=folder_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        use_multithreading=True
    )
    
    documents = loader.load()
    print(f"共加载 {len(documents)} 页PDF文档")
    
    # 分块处理
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？"]  # 根据中文标点分隔
    )
    texts = text_splitter.split_documents(documents)
    print(f"生成 {len(texts)} 个文本块")
    
    # 创建嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        cache_folder="./models/text2vec-base-chinese",
    )
    
    # 构建FAISS索引
    vector_store = FAISS.from_documents(texts, embeddings)
    vector_store.save_local("faiss_index")
    print(f"FAISS索引构建完成，包含 {vector_store.index.ntotal} 个向量")
    
    return vector_store, embeddings


# 通过faiss索引检索案件信息
def retrieve_case_context(vector_store, case_name, top_k=10):

    # 使用案件名称作为查询
    docs = vector_store.similarity_search(case_name, k=top_k)

    # 获取文件基本名（不含路径和扩展名）
    base_name = os.path.splitext(os.path.basename(case_name))[0]
    
    # 过滤出属于该案件的文本块
    case_context = []
    for doc in docs:
        # 从元数据获取源文件名
        source_path = doc.metadata.get('source', '')
        source_name = os.path.splitext(os.path.basename(source_path))[0]
        
        # 使用文件名基本部分匹配
        if source_name == base_name:
            case_context.append(doc.page_content)
    
    return "\n\n".join(case_context)


# 调用DeepSeek API生成结构化JSON
def generate_case_json(context, api_key):

    prompt = f"""
你是一名资深法律专家，请从以下法律文书中提取关键信息并输出JSON格式：
1. 案件类型（必须属于：劳动纠纷、离婚、民间借贷三类之一）
2. 案由
3. 当事人信息（包括原告和被告）
4. 争议焦点
5. 诉讼请求
6. 裁判要点
7. 其他关键信息

### 案件文书内容：
{context}

### 输出要求：
- 严格使用JSON格式
- 案件类型必须是"劳动纠纷"、"离婚"或"民间借贷"
- 当事人信息格式：{{"原告": ["姓名1", "姓名2"], "被告": ["姓名1", "姓名2"]}}
- 其他字段使用数组格式
- 不要包含任何解释性文字
"""
    
    # 调用DeepSeek API
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 2000
    }
    
    try:
        response = requests.post(url, json=data, headers=headers, timeout=90)
        content = response.json()["choices"][0]["message"]["content"]
        
        # 去除可能的Markdown标记
        if content.strip().startswith('```'):
            content = content.strip().lstrip('`json').lstrip('`').rstrip('`').strip()
            content = content.split('```')[-1] if '```' in content else content
        
        # 解析JSON
        return json.loads(content)
    except Exception as e:
        print(f"API调用出错: {str(e)}")
        return {"error": str(e)}

# 主处理函数
def process_all_cases_with_faiss(folder_path, api_key, output_file="result.json"):
    """使用FAISS索引处理所有案件"""
    # 构建索引
    vector_store, _ = build_faiss_index(folder_path)
    
    # 获取所有案件文件
    case_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                case_files.append({
                    "path": os.path.join(root, file),
                    "name": os.path.splitext(file)[0]
                })
    
    print(f"找到 {len(case_files)} 个案件PDF文件")
    
    # 处理每个案件
    results = {}
    for i, case in enumerate(case_files):
        case_name = case["name"]
        print(f"处理中 ({i+1}/{len(case_files)}): {case_name}")
        
        # 通过索引检索案件上下文
        context = retrieve_case_context(vector_store, case_name)
        
        if not context:
            print(f"未找到 {case_name} 的相关内容")
            continue
            
        # 生成结构化JSON
        case_info = generate_case_json(context, api_key)
        
        # 添加到结果
        results[case_name] = case_info
    
    # 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"成功处理 {len(results)} 个案件，结果已保存至 {output_file}")
    return results

def process_case_folders_with_summary(folder_path, output_file="result.json"):
    """遍历每个案件文件夹，提取案情摘要、起诉状、庭审笔录，生成结构化result.json"""
    results = {}
    for case_name in os.listdir(folder_path):
        case_path = os.path.join(folder_path, case_name)
        if not os.path.isdir(case_path):
            continue
        case_info = {}
        # 读取案情摘要、起诉状、庭审笔录
        for key, fname in zip(["案情摘要", "起诉状", "庭审笔录"], ["案情摘要.txt", "起诉状.txt", "庭审笔录.txt"]):
            fpath = os.path.join(case_path, fname)
            if os.path.exists(fpath):
                with open(fpath, "r", encoding="utf-8") as f:
                    case_info[key] = f.read().strip()
            else:
                case_info[key] = ""
        results[case_name] = case_info
    # 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"已生成结构化案件摘要，结果保存至 {output_file}")

from langchain_community.document_loaders import PyPDFLoader

def process_case_folders_with_summary_pdf(folder_path, output_file="result.json"):
    """遍历每个案件文件夹，提取案情摘要、起诉状、庭审笔录（PDF），生成结构化result.json"""
    results = {}
    for case_name in os.listdir(folder_path):
        case_path = os.path.join(folder_path, case_name)
        if not os.path.isdir(case_path):
            continue
        case_info = {}
        # 读取案情摘要、起诉状、庭审笔录（PDF）
        for key, fname in zip(["案情摘要", "起诉状", "庭审笔录"], ["案情摘要.pdf", "起诉状.pdf", "庭审笔录.pdf"]):
            fpath = os.path.join(case_path, fname)
            if os.path.exists(fpath):
                try:
                    loader = PyPDFLoader(fpath)
                    docs = loader.load()
                    text = "\n".join([p.page_content for p in docs])
                    case_info[key] = text.strip()
                except Exception as e:
                    case_info[key] = f"读取失败: {e}"
            else:
                case_info[key] = ""
        results[case_name] = case_info
    # 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"已生成结构化案件摘要（PDF），结果保存至 {output_file}")

def process_all_cases_with_faiss_and_extract(folder_path, api_key, output_file="result.json"):
    """遍历每个案件文件夹，分别对案情摘要、起诉状、庭审笔录PDF抽取关键信息，合并输出result.json"""
    results = {}
    for case_name in os.listdir(folder_path):
        case_path = os.path.join(folder_path, case_name)
        if not os.path.isdir(case_path):
            continue
        case_info = {}
        for key, fname in zip(["案情摘要关键信息", "起诉状关键信息", "庭审笔录关键信息"], ["案情摘要.pdf", "起诉状.pdf", "庭审笔录.pdf"]):
            fpath = os.path.join(case_path, fname)
            if os.path.exists(fpath):
                try:
                    loader = PyPDFLoader(fpath)
                    docs = loader.load()
                    text = "\n".join([p.page_content for p in docs])
                    # 用大模型API抽取关键信息
                    info = generate_case_json(text, api_key)
                    case_info[key] = info
                except Exception as e:
                    case_info[key] = {"error": str(e)}
            else:
                case_info[key] = {}
        results[case_name] = case_info
    # 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"已完成所有案件三类PDF的关键信息抽取，结果保存至 {output_file}")

def extract_party_details(info):
    """从关键信息中提取详细当事人信息（支持dict或list）"""
    details = {"原告": [], "被告": []}
    if not isinstance(info, dict):
        return details
    party_info = info.get("当事人信息", {})
    for role in ["原告", "被告"]:
        val = party_info.get(role, [])
        if isinstance(val, dict):
            details[role].append(val)
        elif isinstance(val, list):
            for v in val:
                if isinstance(v, dict):
                    details[role].append(v)
    return details

def merge_party_details(*details_list):
    """合并多个详细当事人信息，去重"""
    merged = {"原告": [], "被告": []}
    for role in ["原告", "被告"]:
        seen = set()
        for details in details_list:
            for d in details.get(role, []):
                key = tuple(sorted(d.items()))
                if key not in seen:
                    merged[role].append(d)
                    seen.add(key)
    return merged

def add_party_details_to_result(result_path):
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for case_name, case_info in data.items():
        details_list = []
        for key in ["案情摘要关键信息", "起诉状关键信息", "庭审笔录关键信息"]:
            info = case_info.get(key, {})
            details = extract_party_details(info)
            details_list.append(details)
        merged_details = merge_party_details(*details_list)
        case_info["详细当事人信息"] = merged_details
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("已补全详细当事人信息到 result.json")

if __name__ == "__main__":
    # 配置参数
    PDF_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/data2'))  # 用相对路径访问code/data/data2
    API_KEY = "sk-c1d6cb5fc75c4de5ba19fa2b3f1143a1" 
    # 结果保存到src/result/result.json
    result_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../result'))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_path = os.path.join(result_dir, "result.json")
    # 执行处理
    # cases_data = process_all_cases_with_faiss(PDF_FOLDER, API_KEY, output_file=result_path)
    # process_case_folders_with_summary(PDF_FOLDER, output_file=result_path)
    # process_case_folders_with_summary_pdf(PDF_FOLDER, output_file=result_path)
    # process_all_cases_with_faiss_and_extract(PDF_FOLDER, API_KEY, output_file=result_path)
    add_party_details_to_result(result_path)
    # 打印示例输出
    # if cases_data:
    #     first_case = next(iter(cases_data.values()))
    #     print("\n示例案件输出:")
    #     print(json.dumps(first_case, ensure_ascii=False, indent=2))
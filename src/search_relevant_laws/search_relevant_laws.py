import os
import json
import time
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import requests

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 加载PDF文件并构建faiss向量索引
def build_faiss_index(folder_path, index_name):
    """
    构建FAISS向量索引
    :param folder_path: PDF文件夹路径
    :param index_name: 索引名称（用于区分不同类型的内容）
    :return: 向量存储对象
    """
    # 加载PDF
    loader = DirectoryLoader(
        path=folder_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        use_multithreading=True
    )
    documents = loader.load()
    print(f"[{index_name}] 共加载 {len(documents)} 页PDF文档")
    
    # 分块处理
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？"]  # 根据中文标点分隔
    )
    texts = text_splitter.split_documents(documents)
    print(f"[{index_name}] 生成 {len(texts)} 个文本块")
    
    # 创建嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        model_kwargs={'device': 'cpu'},
        cache_folder="./models/text2vec-base-chinese",
    )
    
    # 构建FAISS索引
    vector_store = FAISS.from_documents(texts, embeddings)
    print(f"[{index_name}] FAISS索引构建完成，包含 {vector_store.index.ntotal} 个向量")
    
    return vector_store

# 通过faiss索引检索案件信息
def retrieve_case_context(vector_store, case_name, top_k=10):
    """
    检索特定案件的相关内容
    :param vector_store: 案件摘要向量库
    :param case_name: 案件名称
    :param top_k: 返回结果数量
    :return: 案件相关文本内容
    """
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

# 检索相关法律条文
def retrieve_related_laws(law_vector_store, case_context, top_k=5):
    """
    检索与案件相关的法律条文
    :param law_vector_store: 法律条文向量库
    :param case_context: 案件相关内容
    :param top_k: 返回结果数量
    :return: 相关法律条文列表
    """
    # 使用案件内容作为查询
    docs = law_vector_store.similarity_search(case_context, k=top_k)
    
    # 提取法律条文内容
    laws = []
    for doc in docs:
        # 获取法律条文来源信息
        source_path = doc.metadata.get('source', '')
        law_name = os.path.splitext(os.path.basename(source_path))[0]
        
        # 组合法律条文信息
        law_info = {
            "来源": law_name,
            "内容": doc.page_content
        }
        laws.append(law_info)
    
    return laws

# 调用DeepSeek API生成结构化JSON
def generate_case_json(context, related_laws, api_key):
    """
    生成案件结构化信息
    :param context: 案件相关内容
    :param related_laws: 相关法律条文
    :param api_key: DeepSeek API密钥
    :return: 结构化案件信息
    """
    # 格式化相关法律条文
    laws_text = "\n".join([f"{idx+1}. 《{law['来源']}》: {law['内容']}" 
                          for idx, law in enumerate(related_laws)])
    
    prompt = f"""
你是一名资深法律专家，请从以下法律文书中提取关键信息并输出JSON格式：
1. 案件类型（必须属于：劳动纠纷、离婚、民间借贷三类之一）
2. 案由
3. 当事人信息（包括原告和被告）
4. 争议焦点
5. 诉讼请求
6. 裁判要点
7. 相关法律条文原文（基于提供的条文列表，不可修改）
8. 如何依据法律条文对案件加以判决（与相关法律条文原文一一对应）
9. 其他关键信息

### 案件文书内容：
{context}

### 相关法律条文：
{laws_text}

### 输出要求：
- 严格使用JSON格式
- 案件类型必须是"劳动纠纷"、"离婚"或"民间借贷"
- 当事人信息格式：{{"原告": ["姓名1", "姓名2"], "被告": ["姓名1", "姓名2"]}}
- 相关法律条文及依据条文的逻辑推断格式：[{{"条纹名称"："名称"，"条文内容": "内容"，"逻辑推断"："推断"}}]
- 其他字段使用数组格式
- 相关法律条文尽可能多，每个案件至少六条
- 相关法律条文尽可能来自不同的法律文件，如“民法典”与“劳动合同法”是两个不同的法律文件
- 不要包含任何解释性文字
- 不许偷懒！！！！！！！！！！！！
"""
    
    # 调用DeepSeek API
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 3000  # 增加token限制以容纳法律条文
    }
    
    try:
        response = requests.post(url, json=data, headers=headers, timeout=120)
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
def process_cases_with_laws(case_folder, law_folder, api_key, output_file="case_law_results.json"):
    """
    处理案件并关联相关法律条文
    :param case_folder: 案件摘要文件夹路径
    :param law_folder: 法律条文文件夹路径
    :param api_key: DeepSeek API密钥
    :param output_file: 输出文件名
    :return: 处理结果
    """
    start_time = time.time()
    
    # 构建案件摘要索引
    print("\n" + "="*60)
    print("⚖️ 开始构建案件摘要索引...")
    case_vector_store = build_faiss_index(case_folder, "案件摘要")
    
    # 构建法律条文索引
    print("\n" + "="*60)
    print("📜 开始构建法律条文索引...")
    law_vector_store = build_faiss_index(law_folder, "法律条文")
    
    # 获取所有案件文件
    case_files = []
    for root, _, files in os.walk(case_folder):
        for file in files:
            if file.lower().endswith(".pdf"):
                case_files.append({
                    "path": os.path.join(root, file),
                    "name": os.path.splitext(file)[0]
                })
    
    print(f"\n🔍 找到 {len(case_files)} 个案件PDF文件")
    
    # 处理每个案件
    results = {}
    for i, case in enumerate(case_files):
        case_name = case["name"]
        print(f"\n{'='*60}")
        print(f"处理案件 ({i+1}/{len(case_files)}): {case_name}")
        
        # 检索案件上下文
        context = retrieve_case_context(case_vector_store, case_name)
        
        if not context:
            print(f"⚠️ 未找到 {case_name} 的相关内容")
            continue
            
        # 检索相关法律条文
        print("🔍 检索相关法律条文...")
        related_laws = retrieve_related_laws(law_vector_store, context)
        print(f"✅ 找到 {len(related_laws)} 条相关法律条文")
        
        # 生成结构化JSON（包含法律条文）
        print("🧠 生成结构化案件信息...")
        case_info = generate_case_json(context, related_laws, api_key)
        
        # 添加到结果
        results[case_name] = case_info
        print(f"✅ 案件处理完成")
    
    # 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 计算处理时间
    processing_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ 成功处理 {len(results)} 个案件")
    print(f"⏱️ 总耗时: {processing_time:.2f}秒")
    print(f"📄 结果已保存至 {output_file}")
    
    return results

if __name__ == "__main__":
    # 配置参数
    CASE_FOLDER = r"C:\code\rag_framed\faiss_RAGextractor\data2"  # 案件摘要文件夹
    LAW_FOLDER = r"C:\code\rag_framed\data1\laws"    # 法律条文文件夹
    API_KEY = "sk-c1d6cb5fc75c4de5ba19fa2b3f1143a1" 
    
    # 执行处理
    results = process_cases_with_laws(CASE_FOLDER, LAW_FOLDER, API_KEY)
    
    # 打印示例输出
    if results:
        first_case = next(iter(results.values()))
        print("\n" + "="*60)
        print("示例案件输出:")
        print(json.dumps(first_case, ensure_ascii=False, indent=2))
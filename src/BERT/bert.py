from transformers import BertTokenizer, BertModel

# 加载支持中文的tokenizer和model
model_name = "bert-base-chinese"  # 使用中文预训练模型
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

import os
import re
import jieba
import numpy as np
import torch
from pathlib import Path
from docx import Document
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import json
import pickle

# 1. 安全加载BERT模型
def load_bert_model():
    """加载BERT模型和分词器，支持多种模型选项"""
    MODEL_NAMES = [
        "bert-base-chinese",
        "hfl/chinese-bert-wwm-ext",
        "shibing624/text2vec-base-chinese"
    ]
    
    for model_name in MODEL_NAMES:
        try:
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertModel.from_pretrained(model_name)
            print(f"✅ 成功加载模型: {model_name}")
            return tokenizer, model, model_name
        except Exception as e:
            print(f"⚠️ 无法加载模型 {model_name}: {str(e)}")
    
    try:
        print("⚠️ 尝试从本地缓存加载模型...")
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", local_files_only=True)
        model = BertModel.from_pretrained("bert-base-chinese", local_files_only=True)
        return tokenizer, model, "bert-base-chinese (本地缓存)"
    except Exception as e:
        print(f"❌ 无法加载任何BERT模型: {str(e)}")
        raise RuntimeError("无法加载BERT模型，请检查网络连接或安装状态")

# 2. 从DOCX文件中提取文本
def extract_text_from_docx(docx_path):
    """从DOCX文件中提取纯文本内容"""
    try:
        doc = Document(docx_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        print(f"❌ 读取文件 {docx_path} 时出错: {str(e)}")
        return ""

# 3. 文本预处理和分块
def preprocess_and_chunk(text, tokenizer, max_chunk_length=400):
    """预处理文本并分块，返回文本块列表"""
    # 清洗文本
    text = re.sub(r'\s+', ' ', text)  # 合并空白字符
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)  # 保留中文字符和常用标点
    
    # 使用jieba分词
    words = jieba.lcut(text)
    
    # 分块处理
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if not word.strip():
            continue
            
        word_tokens = tokenizer.tokenize(word)
        word_length = len(word_tokens)
        
        if current_length + word_length > max_chunk_length and current_chunk:
            chunks.append("".join(current_chunk))
            current_chunk = []
            current_length = 0
        
        current_chunk.append(word)
        current_length += word_length
    
    if current_chunk:
        chunks.append("".join(current_chunk))
    
    return chunks

# 4. 计算文档向量
def compute_document_vector(text, tokenizer, model, strategy="mean-pooling"):
    """计算整个文档的向量表示"""
    # 处理空文本
    if not text.strip():
        return np.zeros(model.config.hidden_size)
    
    # 分块处理长文本
    chunks = preprocess_and_chunk(text, tokenizer)
    
    # 处理无有效内容的情况
    if not chunks:
        return np.zeros(model.config.hidden_size)
    
    chunk_embeddings = []
    
    # 批处理所有分块
    inputs = tokenizer(
        chunks,
        max_length=512,
        truncation=True,
        padding='max_length',
        return_tensors="pt"
    )
    
    # 使用GPU加速（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
    
    # 为每个分块生成向量
    for i in range(len(chunks)):
        if strategy == "mean-pooling":
            # 平均池化（去除padding部分）
            mask = inputs['attention_mask'][i].unsqueeze(-1)
            chunk_embedding = (last_hidden_states[i] * mask).sum(dim=0) / mask.sum()
        elif strategy == "max-pooling":
            # 最大池化
            chunk_embedding = last_hidden_states[i].max(dim=0)[0]
        elif strategy == "cls":
            # 使用[CLS]标记
            chunk_embedding = last_hidden_states[i][0]
        else:
            # 默认使用平均池化
            mask = inputs['attention_mask'][i].unsqueeze(-1)
            chunk_embedding = (last_hidden_states[i] * mask).sum(dim=0) / mask.sum()
        
        chunk_embeddings.append(chunk_embedding.cpu().numpy())
    
    # 聚合所有分块向量（按长度加权平均）
    weights = [len(chunk) for chunk in chunks]
    total_weight = sum(weights)
    weighted_embeddings = [emb * weight for emb, weight in zip(chunk_embeddings, weights)]
    
    return sum(weighted_embeddings) / total_weight

# 5. 处理文件夹中的所有DOCX文件
def process_docx_folder(folder_path, output_dir="document_vectors"):
    """
    处理文件夹中的所有DOCX文件，计算并保存文本向量
    :param folder_path: 包含DOCX文件的文件夹路径
    :param output_dir: 输出向量文件的目录
    """
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 加载BERT模型
    tokenizer, model, model_name = load_bert_model()
    
    # 获取所有DOCX文件
    docx_files = [f for f in Path(folder_path).glob("**/*.docx") if f.is_file()]
    print(f"📂 找到 {len(docx_files)} 个DOCX文件")
    
    # 结果存储
    results = {
        "model_name": model_name,
        "vector_size": model.config.hidden_size,
        "documents": []
    }
    
    # 向量存储（文件名 -> 向量）
    vectors = {}
    
    # 处理每个文件
    for docx_path in tqdm(docx_files, desc="处理文档"):
        try:
            # 提取文本
            text = extract_text_from_docx(docx_path)
            if not text.strip():
                print(f"⚠️ 文件 {docx_path.name} 内容为空，跳过")
                continue
                
            # 计算文档向量
            vector = compute_document_vector(text, tokenizer, model)
            
            # 存储结果
            doc_info = {
                "file_name": docx_path.name,
                "file_path": str(docx_path),
                "vector": vector.tolist(),  # 转换为列表便于JSON序列化
                "vector_norm": float(np.linalg.norm(vector))
            }
            
            results["documents"].append(doc_info)
            vectors[docx_path.name] = vector
            
            print(f"✅ 完成处理: {docx_path.name} (向量范数: {doc_info['vector_norm']:.2f})")
        except Exception as e:
            print(f"❌ 处理文件 {docx_path.name} 时出错: {str(e)}")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_prefix = f"document_vectors_{timestamp}"
    
    # 保存为JSON（包含元数据）
    json_path = Path(output_dir) / f"{output_prefix}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"💾 元数据保存至: {json_path}")
    
    # 保存为Pickle（便于后续加载使用）
    pkl_path = Path(output_dir) / f"{output_prefix}.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(vectors, f)
    print(f"💾 向量数据保存至: {pkl_path}")
    
    # 保存为NumPy格式（适用于机器学习）
    np_path = Path(output_dir) / f"{output_prefix}.npz"
    np.savez_compressed(
        np_path, 
        filenames=list(vectors.keys()),
        vectors=np.array(list(vectors.values()))
    )
    print(f"💾 NumPy格式保存至: {np_path}")
    
    return vectors

# 6. 示例使用
if __name__ == "__main__":
    from datetime import datetime
    
    # 设置输入文件夹和输出目录
    input_folder = r"../data/附件2"  # 替换为你的DOCX文件夹路径
    output_dir = r"../background"  # 替换为你的输出目录路径
    
    print(f"🕒 开始处理: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    vectors = process_docx_folder(input_folder, output_dir)
    print(f"✅ 处理完成! 共计算 {len(vectors)} 个文档向量")
    print(f"🕒 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

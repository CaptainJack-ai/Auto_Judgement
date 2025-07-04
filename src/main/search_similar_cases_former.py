import json
import numpy as np
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def cosine_sim(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def is_valid_vector(vec):
    return isinstance(vec, list) and all(isinstance(x, (float, int)) for x in vec)

def load_faiss_index(doc1_vectors):
    """构建FAISS索引并加载向量"""
    embeddings = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        cache_folder="./models/text2vec-base-chinese",
    )
    vector_store = FAISS(embeddings)
    vector_store.index.add(np.array(doc1_vectors))
    return vector_store

def main():
    # 只读取src/background/document_vectors.json中documents下的vector
    doc1_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../background/document_vectors.json'))
    with open(doc1_path, "r", encoding="utf-8") as f:
        doc1_raw = json.load(f)
    doc1_docs = doc1_raw["documents"] if "documents" in doc1_raw else []
    doc1_names = [item["file_name"] for item in doc1_docs if "file_name" in item and is_valid_vector(item.get("vector"))]
    doc1_vectors = [item["vector"] for item in doc1_docs if "file_name" in item and is_valid_vector(item.get("vector"))]
    print(f"document_vectors.json有效文档数: {len(doc1_names)}")

    # 构建FAISS索引
    vector_store = load_faiss_index(doc1_vectors)

    # 只读取src/background/附件1 vec.json中documents下的vector
    doc2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../background/附件1 vec.json'))
    with open(doc2_path, "r", encoding="utf-8") as f:
        doc2_raw = json.load(f)
    doc2_docs = doc2_raw["documents"] if "documents" in doc2_raw else []
    doc2_names = [item["file_name"] for item in doc2_docs if "file_name" in item and is_valid_vector(item.get("vector"))]
    doc2_vectors = [item["vector"] for item in doc2_docs if "file_name" in item and is_valid_vector(item.get("vector"))]
    print(f"附件1 vec.json有效文档数: {len(doc2_names)}")

    # 反向：对每个附件1 vec.json中的文档，找出与document_vectors.json中最相似的前20个文档
    results = {}
    for idx, (doc2_name, doc2_vec) in enumerate(zip(doc2_names, doc2_vectors)):
        # 使用FAISS检索
        distances, indices = vector_store.index.search(np.array([doc2_vec]), k=20)
        similar = []
        for dist, i in zip(distances[0], indices[0]):
            similar.append({"编号": doc1_names[i], "相似度": 1 - dist})  # FAISS返回的是距离，需转为相似度
        results[doc2_name] = similar

        if idx < 3:
            print(f"[{doc2_name}] FAISS前5: {similar[:5]}")

    # 保存结果到src/result
    result_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../result'))
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, "similar_cases.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("附件1 vec.json与document_vectors.json向量相似度检索完成，结果已保存为 src/result/similar_cases.json")

if __name__ == "__main__":
    main()

import sys
import os
import json
import torch

# 配置 sys.stdout/stderr 以在 Windows 下输出 utf-8，防止 Unicode 编码报错
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# 1. 注入当前项目路径和旧 RAG 路径
current_project_path = "E:/project/advanced-rag"
if current_project_path not in sys.path:
    sys.path.insert(0, current_project_path)

rag_path = "E:/project/rag"
if rag_path not in sys.path:
    sys.path.append(rag_path)


# 配置 HuggingFace 相关的环境变量（确保直接使用下载好的本地缓存）
os.environ['HF_HOME'] = r'D:\my_huggingface_cache'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = r'D:\my_huggingface_cache'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from rag_engine import RAGEngine, RAGConfig

# 2. 目标文档路径
DOC_CHINESE = "E:/desktop/code/New folder/paper song.docx"
DOC_ENGLISH = "E:/project/DeepSeek-OCR/ocr_results/44221625_LI LEI/44221625_LI LEI_merged.docx"

def align_naive_rag(engine):
    print("Checking / Aligning documents in Naive RAG...")
    for path in [DOC_CHINESE, DOC_ENGLISH]:
        filename = os.path.basename(path)
        existing = engine.collection.get(where={"filename": filename})
        if existing and existing.get('documents') and len(existing.get('documents')) > 0:
            print(f"Naive RAG: Document '{filename}' already exists in database. Skipping indexing.")
        else:
            print(f"Naive RAG: Document '{filename}' not found. Adding file...")
            engine.add_file(path)

def align_advanced_rag(coordinator):
    print("Checking / Aligning documents in Advanced RAG...")
    for path in [DOC_CHINESE, DOC_ENGLISH]:
        filename = os.path.basename(path)
        existing = coordinator.db_adapter.collection.get(where={"filename": filename})
        if existing and existing.get('documents') and len(existing.get('documents')) > 0:
            print(f"Advanced RAG: Document '{filename}' already exists in database. Skipping indexing.")
        else:
            print(f"Advanced RAG: Document '{filename}' not found. Adding file...")
            coordinator.add_file(path)

def main():
    # 检测运行设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Detected and using device for models: {device}")
    
    # 实例化 Naive RAG Engine
    naive_config = RAGConfig()
    naive_config.EMBEDDING_DEVICE = device
    naive_config.RERANKER_DEVICE = device
    naive_engine = RAGEngine(naive_config)
    
    # 导入新版 RAG 模块并实例化 RAGCoordinator
    from src.loader import DocumentLoader
    from src.splitter import SemanticParentChildSplitter
    from src.embedding import LocalEmbeddingService
    from src.database import ChromaAdapter
    from src.reranker import RerankerService
    from src.coordinator import RAGCoordinator

    db_dir = "E:/project/advanced-rag/vector_db"
    loader = DocumentLoader()
    embedding_service = LocalEmbeddingService(device=device)
    splitter = SemanticParentChildSplitter(embedding_service=embedding_service, threshold=None, child_size=150)
    db_adapter = ChromaAdapter(db_dir=db_dir)
    reranker_service = RerankerService(device=device)

    coordinator = RAGCoordinator(
        loader=loader,
        splitter=splitter,
        embedding_service=embedding_service,
        db_adapter=db_adapter,
        reranker=reranker_service
    )

    # 对齐数据（索引缺失文档）
    align_naive_rag(naive_engine)
    align_advanced_rag(coordinator)

    # 读取测试数据集
    dataset_path = "E:/project/advanced-rag/tests/test_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        questions_data = json.load(f)

    print(f"Loaded {len(questions_data)} questions from dataset.")

    results = []
    # 遍历每个问题，执行检索
    for i, item in enumerate(questions_data):
        question = item["question"]
        ground_truth = item["ground_truth"]
        print(f"[{i+1}/{len(questions_data)}] Querying for: {question}")
        
        # Naive RAG 检索 (top_k=5)
        naive_context = naive_engine.search_with_context(question, top_k=5)
        
        # Advanced RAG 检索 (通过 coordinator.query 检索并返回 top 5 的拼接上下文)
        advanced_context = coordinator.query(question)
        
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "naive_context": naive_context,
            "advanced_context": advanced_context
        })

    # 保存结果到 tests/retrieval_results.json
    output_path = "E:/project/advanced-rag/tests/retrieval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Successfully saved {len(results)} retrieval results to {output_path}")

if __name__ == "__main__":
    main()

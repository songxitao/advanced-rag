import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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

import re

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

def align_sanguo_dataset(naive_engine, coordinator):
    print("Cleaning existing database and aligning with disguised Sanguo Dataset...")
    
    # 1. 彻底清空 Naive RAG
    naive_engine.clear_db()
    
    # 2. 彻底清空 Advanced RAG Chroma 与 NetworkX 内存图
    try:
        coordinator.db_adapter.client.delete_collection(coordinator.db_adapter.collection.name)
    except Exception as e:
        print(f"Warning deleting collection: {e}")
    coordinator.db_adapter.collection = coordinator.db_adapter.client.get_or_create_collection(
        name=coordinator.db_adapter.collection.name,
        metadata={"hnsw:space": "cosine"}
    )
    import networkx as nx
    coordinator.db_adapter.bm25 = None
    coordinator.db_adapter.bm25_docs = []
    coordinator.db_adapter.graph = nx.Graph()
    
    # 3. 仅且只导入全量伪装好的新书
    disguised_path = "tests/temp_data/三国演义白话文_disguised.txt"
    if os.path.exists(disguised_path):
        filename = os.path.basename(disguised_path)
        print(f"Adding disguised book '{filename}' to Advanced RAG...")
        coordinator.add_file(disguised_path)
        print(f"Adding disguised book '{filename}' to Naive RAG...")
        naive_engine.add_file(disguised_path)
    else:
        raise FileNotFoundError(f"未找到全量伪装文本: {disguised_path}")

def main():
    # 开启 CUDA 运行并释放线程锁
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device for RAG models: {device}")
    
    # 导入新版 RAG 模块并实例化 RAGCoordinator
    from src.loader import DocumentLoader
    from src.splitter import SemanticParentChildSplitter
    from src.embedding import LocalEmbeddingService
    from src.database import ChromaAdapter
    from src.reranker import RerankerService
    from src.coordinator import RAGCoordinator
    from src.graph_search import GraphPostRetriever

    # 1. 先实例化高级 RAG 相关的 embedding 和 reranker 服务（模型共享源头）
    embedding_service = LocalEmbeddingService(device=device)
    reranker_service = RerankerService(device=device)

    # 2. 注入 Mock 方法，防止 RAGEngine 实例化时重复加载模型产生死锁
    original_init_emb = RAGEngine._init_embedding_model
    original_init_rerank = RAGEngine._init_reranker_model
    
    def mock_init_emb(self):
        print("🔄 Sharing embedding model instance with Advanced RAG...")
        self.embedding_model = embedding_service.model
        print("✅ Shared Embedding 就绪")
        
    def mock_init_rerank(self):
        print("🔄 Sharing reranker model instance with Advanced RAG...")
        self.reranker_model = reranker_service.model
        print("✅ Shared Reranker 就绪")
        
    RAGEngine._init_embedding_model = mock_init_emb
    RAGEngine._init_reranker_model = mock_init_rerank

    # 3. 实例化 Naive RAG Engine（此时将秒级安全加载）
    naive_config = RAGConfig()
    naive_config.EMBEDDING_DEVICE = device
    naive_config.RERANKER_DEVICE = device
    naive_engine = RAGEngine(naive_config)
    
    # 恢复原函数，避免污染
    RAGEngine._init_embedding_model = original_init_emb
    RAGEngine._init_reranker_model = original_init_rerank

    db_dir = "E:/project/advanced-rag/vector_db"
    loader = DocumentLoader()
    splitter = SemanticParentChildSplitter(embedding_service=embedding_service, threshold=None, child_size=150)
    db_adapter = ChromaAdapter(db_dir=db_dir)

    coordinator = RAGCoordinator(
        loader=loader,
        splitter=splitter,
        embedding_service=embedding_service,
        db_adapter=db_adapter,
        reranker=reranker_service
    )

    # 注入“大海背景噪声”和“伪装情节文本”
    align_sanguo_dataset(naive_engine, coordinator)

    # 实例化图检索器
    retriever = GraphPostRetriever(
        embedding_service=embedding_service,
        db_adapter=db_adapter,
        reranker=reranker_service
    )

    # 读取测试数据集
    dataset_path = "E:/project/advanced-rag/tests/temp_data/test_sanguo_dataset.json"
    if not os.path.exists(dataset_path):
        print(f"❌ 数据集文件不存在: {dataset_path}。请先生成数据集。")
        sys.exit(1)
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        questions_data = json.load(f)

    print(f"Loaded {len(questions_data)} questions from dataset.")

    results = []
    # 遍历每个问题，执行多轨检索
    for i, item in enumerate(questions_data):
        question = item["question"]
        ground_truth = item["ground_truth"]
        print(f"[{i+1}/{len(questions_data)}] Querying for: {question}")
        
        # 1. Naive RAG 检索 (top_k=5)
        naive_context = naive_engine.search_with_context(question, top_k=5)
        
        # 2. 传统 Advanced RAG 检索 (none 模式)
        traditional_context = retriever.query_graph_enhanced(question, "none")
        
        # 3. PPR 图 RAG 检索 (ppr 模式)
        ppr_context = retriever.query_graph_enhanced(question, "ppr")
        
        # 4. 语义游走图 RAG 检索 (heuristic_walk 模式)
        walk_context = retriever.query_graph_enhanced(question, "heuristic_walk")
        
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "naive_context": naive_context,
            "traditional_context": traditional_context,
            "ppr_context": ppr_context,
            "walk_context": walk_context
        })

    # 保存结果到 tests/temp_data/retrieval_sanguo_results.json
    output_path = "E:/project/advanced-rag/tests/temp_data/retrieval_sanguo_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Successfully saved {len(results)} retrieval results to {output_path}")

if __name__ == "__main__":
    main()

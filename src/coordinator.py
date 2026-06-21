import os
from src.graph_search import run_personalized_pagerank, run_semantic_random_walk

class RAGCoordinator:
    def __init__(self, loader, splitter, embedding_service, db_adapter, reranker):
        """
        初始化 RAGCoordinator
        :param loader: 文档加载器实例 DocumentLoader
        :param splitter: 父子块切分器实例 SemanticParentChildSplitter
        :param embedding_service: 向量生成服务实例 LocalEmbeddingService
        :param db_adapter: 向量数据库适配器实例 ChromaAdapter
        :param reranker: 重排服务实例 RerankerService
        """
        self.loader = loader
        self.splitter = splitter
        self.embedding_service = embedding_service
        self.db_adapter = db_adapter
        self.reranker = reranker

    def add_file(self, file_path: str) -> None:
        """
        加载文件 -> 切分父子块 -> 补齐元数据 -> 生成 Dense 向量 -> 持久化入库
        :param file_path: 文件路径
        """
        # 1. 读文件内容
        text = self.loader.load(file_path)
        if not text:
            raise ValueError(f"无法从文件中提取任何文本，请检查文件是否为空或属于未OCR的扫描件: {file_path}")

        # 2. 根据扩展名判定是否为 Markdown 文件，并切出父子块
        ext = os.path.splitext(file_path)[1].lower()
        is_markdown = (ext == '.md')
        chunks = self.splitter.create_parent_child_chunks(text, is_markdown=is_markdown)
        if not chunks:
            raise ValueError(f"未能为该文件生成任何有效的语义切片块: {file_path}")

        # 获取源文件绝对路径和文件名
        source_path = os.path.abspath(file_path)
        filename = os.path.basename(file_path)

        # 3. 补齐元数据（source_path, filename, char_start, char_end）
        current_pos = 0
        for chunk in chunks:
            child_text = chunk["child_text"]
            # 顺序在 text 中查找 child_text 的起始和结束位置
            start_idx = text.find(child_text, current_pos)
            if start_idx == -1:
                # 若从当前位置往后找不到，则全局查找
                start_idx = text.find(child_text)
            
            if start_idx != -1:
                end_idx = start_idx + len(child_text)
                current_pos = end_idx
            else:
                start_idx = 0
                end_idx = 0

            chunk["source_path"] = source_path
            chunk["filename"] = filename
            chunk["char_start"] = start_idx
            chunk["char_end"] = end_idx

        # 4. 调用 embedding_service 生成每个子块的 Dense 向量 (使用批量并行接口)
        child_texts = [chunk["child_text"] for chunk in chunks]
        dense_embeddings = self.embedding_service.get_dense_embeddings_batch(child_texts)

        # 5. 调用 db_adapter 持久化入库
        self.db_adapter.add_chunks(chunks, dense_embeddings)

    def query(self, user_question: str, graph_search_mode: str = "heuristic_walk") -> str:
        """
        双通道混合检索 -> 一次重排锁定 Seed Node -> 图检索拓扑扩展 -> 二路融合 -> 二次重排与断崖截断 -> 格式化输出
        :param user_question: 用户提问
        :param graph_search_mode: 图检索模式，可选 "heuristic_walk"、"ppr" 或 "none"
        :return: 拼接格式化后的上下文字符串
        """
        # 1. 计算提问的 Dense 和 Sparse 向量
        dense_vec = self.embedding_service.get_dense_embedding(user_question)
        sparse_vec = self.embedding_service.get_sparse_embedding(user_question)

        # 2. 调用 db_adapter.hybrid_search 初筛去重召回 (Top 15)
        candidates = self.db_adapter.hybrid_search(dense_vec, sparse_vec, top_k=15)
        if not candidates:
            return ""

        # 3. 一次重排锁定 Seed Node
        # 对初筛的全部候选块进行重排，不进行截断限制，以便寻找最高得分的父块作为 Seed Node
        first_rerank = self.reranker.rerank(user_question, candidates, top_k=len(candidates))
        if not first_rerank:
            return ""

        seed_candidate = first_rerank[0]
        seed_node_id = seed_candidate["metadata"].get("parent_id")

        # 4. 图拓扑扩展 (1-2跳扩散)
        graph_pids = []
        if seed_node_id:
            if graph_search_mode == "heuristic_walk":
                graph_pids = run_semantic_random_walk(self.db_adapter.graph, seed_node_id, dense_vec, top_k=5)
            elif graph_search_mode == "ppr":
                graph_pids = run_personalized_pagerank(self.db_adapter.graph, seed_node_id, top_k=5)

        # 5. 双路合流
        # 以 parent_id 作为去重键，建立初筛映射
        candidates_dict = {c["metadata"]["parent_id"]: c for c in candidates}
        
        for pid in graph_pids:
            if pid not in candidates_dict:
                # 若图检索捞回的 parent_id 在初筛中不存在，则自动通过内存图中的节点属性补齐构建 candidate
                if pid in self.db_adapter.graph:
                    node_data = self.db_adapter.graph.nodes[pid]
                    candidates_dict[pid] = {
                        "content": node_data.get("parent_text", ""),
                        "metadata": {
                            "parent_id": pid,
                            "parent_text": node_data.get("parent_text", ""),
                            "source_path": node_data.get("source_path", ""),
                            "filename": node_data.get("filename", ""),
                            "char_start": 0,
                            "char_end": 0
                        }
                    }
        
        combined_candidates = list(candidates_dict.values())

        # 6. 二次 Rerank 与断崖截断
        # 将合流后的所有候选送入 reranker.rerank 深度重排，最多截取前 5 个
        selected = self.reranker.rerank(user_question, combined_candidates, top_k=5)
        if not selected:
            return ""

        # 在返回前，自动按精排得分降序检查相邻得分落差，若落差大于 1.5，则即时断开截断后续低相关文本
        if len(selected) > 1:
            cutoff_idx = len(selected)
            for i in range(len(selected) - 1):
                drop = selected[i]["rerank_score"] - selected[i+1]["rerank_score"]
                if drop > 1.5:
                    cutoff_idx = i + 1
                    break
            selected = selected[:cutoff_idx]

        # 7. 执行父块替换并拼接格式化后的上下文字符串
        formatted_parts = []
        for idx, candidate in enumerate(selected, 1):
            filename = candidate["metadata"].get("filename", "未知文件")
            parent_text = candidate["metadata"].get("parent_text", "")
            part_str = f"[片段{idx}] (来源: {filename})\n{parent_text}"
            formatted_parts.append(part_str)

        return "\n\n".join(formatted_parts)


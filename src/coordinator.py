import os

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

        # 获取源文件绝对路径 and 文件名
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

    def query(self, user_question: str) -> str:
        """
        双通道混合检索 -> 重排精选 -> 父块替换并追加来源文献 -> 格式化输出
        :param user_question: 用户提问
        :return: 拼接格式化后的上下文字符串
        """
        # 1. 计算提问的 Dense 和 Sparse 向量
        dense_vec = self.embedding_service.get_dense_embedding(user_question)
        sparse_vec = self.embedding_service.get_sparse_embedding(user_question)

        # 2. 调用 db_adapter.hybrid_search 初筛去重召回 (Top 15)
        candidates = self.db_adapter.hybrid_search(dense_vec, sparse_vec, top_k=15)
        if not candidates:
            return ""

        # 3. 调用 reranker.rerank 进行深度重排精选 (Top 5)
        selected = self.reranker.rerank(user_question, candidates, top_k=5)
        if not selected:
            return ""

        # 4. 执行父块替换并拼接格式化后的上下文字符串
        formatted_parts = []
        for idx, candidate in enumerate(selected, 1):
            filename = candidate["metadata"].get("filename", "未知文件")
            parent_text = candidate["metadata"].get("parent_text", "")
            # 按照格式拼接：[片段i] (来源: 文件名)\n父块文本...
            part_str = f"[片段{idx}] (来源: {filename})\n{parent_text}"
            formatted_parts.append(part_str)

        return "\n\n".join(formatted_parts)


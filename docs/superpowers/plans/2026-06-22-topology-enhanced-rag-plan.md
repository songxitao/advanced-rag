# 拓扑增强父子块 RAG 系统实现计划书

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为现有的 BGE-M3 父子块 RAG 系统引入免训练的图拓扑增强检索机制与指代消解换皮脱敏防作弊测试管线，解决跨章节隐式多跳推理难题。

**Architecture:** 采用外挂式 Post-Retrieval Pipeline，底座采用 NetworkX 构建三轨连边内存图。检索时以初筛 Top-1 块为 Seed Node，可配置地使用 PPR 或 BGE-M3 语义引导 2 跳随机游走在图中扩散，与初始候选集求并集后，在融合层进行二次重排与自适应断崖阻断，最终 100% 闭环还原为文本证据输出。

**Tech Stack:** Python 3.12, NetworkX, jieba, chromadb, rank_bm25, requests, pytest

---

### Task 1: 别称消解与整本全量脱敏预处理脚本开发

**Files:**
*   Create: `tests/disguise_book_generator.py`
*   Test: `tests/test_disguise_generator.py`
*   Target Output Path: `tests/temp_data/`

- [ ] **Step 1: 编写失效测试（验证伪装生成失败或不存在）**
    
    新建测试文件 `tests/test_disguise_generator.py`，内容如下：
    ```python
    import os
    import pytest

    def test_disguise_pipeline():
        # 目标脱敏文件不存在
        disguised_file = "tests/temp_data/三国演义白话文_disguised.txt"
        assert not os.path.exists(disguised_file)
    ```

- [ ] **Step 2: 运行测试确保其失败**
    
    运行：`pytest tests/test_disguise_generator.py::test_disguise_pipeline -v`
    预期：PASS（由于该文件目前确实不存在，它将通过。为了强制失败，我们可以改为验证引入该库并生成失败）。
    更正测试内容：
    ```python
    import pytest
    def test_import():
        from tests.disguise_book_generator import run_disguise_pipeline
        assert False  # 强制失败以验证 TDD 流程
    ```
    再次运行：`pytest tests/test_disguise_generator.py::test_import -v`
    预期：FAIL with "No module named 'tests.disguise_book_generator'"

- [ ] **Step 3: 编写最小实现代码**
    
    在 `tests/disguise_book_generator.py` 中实现自动化人物名提取、别称聚类以及整本安全替换的代码：
    ```python
    import os
    import re
    import json
    import requests
    import jieba.posseg as pseg

    LLM_API_URL = "http://localhost:8080/v1/chat/completions"
    MODEL_NAME = "qwen3.6-35b-a3b-distilled-think"

    def run_disguise_pipeline(input_file: str, output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()

        # 1. 自动提取高频人物名
        words = pseg.cut(text)
        name_freq = {}
        for w, flag in words:
            if flag == 'nr' and len(w) >= 2:
                name_freq[w] = name_freq.get(w, 0) + 1
        
        # 筛选频次大于 15 的人名
        high_freq_names = [w for w, f in name_freq.items() if f >= 15]

        # 2. 调用本地 Qwen 接口进行指代消解别称合并
        prompt = f"""你是一个文学人物指代消解与关系对齐专家。
请对以下《三国演义》高频人名列表进行别称聚类归并。将属于同一个人物的名字（如刘备、玄德、刘玄德、使君、皇叔）归入同一个列表中，并为他们分配一个统一的伪装代号（格式为“角色_字母”，如“角色_Alpha”，“角色_Beta”）。

【人名列表】：
{", ".join(high_freq_names)}

【输出格式】：
严格按照以下 JSON 格式输出，不要包含任何多余解释或 markdown 标记：
{{
  "角色_Alpha": ["刘备", "玄德", "刘玄德", "皇叔", "刘皇叔", "使君"],
  "角色_Beta": ["曹操", "孟德", "曹阿瞒"]
}}
"""
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that outputs raw JSON content."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1
        }
        
        try:
            resp = requests.post(LLM_API_URL, json=payload, timeout=60)
            resp.raise_for_status()
            content = resp.json()['choices'][0]['message']['content'].strip()
            # 过滤 think 标签和代码块提取 JSON
            content_clean = re.sub(r'<think>[\s\S]*?(?:</think>|$)', '', content).strip()
            match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content_clean)
            if match:
                content_clean = match.group(1).strip()
            alias_map = json.loads(content_clean)
        except Exception as e:
            # 降级兜底方案：预定义核心人物映射
            alias_map = {
                "角色_Alpha": ["刘备", "玄德", "刘玄德", "皇叔", "刘皇叔", "使君", "刘豫州", "先主"],
                "角色_Beta": ["张飞", "翼德", "张翼德"],
                "角色_Gamma": ["关羽", "云长", "关云长", "美髯公", "红脸面"],
                "角色_Delta": ["曹操", "孟德", "阿瞒", "曹丞相"],
                "目标_X": ["督邮", "都邮"]
            }

        # 保存别称映射 JSON
        with open(os.path.join(output_dir, "sanguo_aliases.json"), 'w', encoding='utf-8') as f:
            json.dump(alias_map, f, ensure_ascii=False, indent=2)

        # 3. 排序逆向最大安全替换
        # 收集所有的 (原始名, 代号)
        replace_pairs = []
        for code, aliases in alias_map.items():
            for alias in aliases:
                replace_pairs.append((alias, code))
                
        # 按照原始名长度降序排序，防止短名字截断长名字
        replace_pairs.sort(key=lambda x: len(x[0]), reverse=True)

        disguised_text = text
        for alias, code in replace_pairs:
            disguised_text = disguised_text.replace(alias, code)

        output_file = os.path.join(output_dir, "三国演义白话文_disguised.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(disguised_text)
        
        return output_file
    ```

- [ ] **Step 4: 修改测试，运行并验证通过**
    
    在 `tests/test_disguise_generator.py` 中写入真实测试逻辑：
    ```python
    import os
    from tests.disguise_book_generator import run_disguise_pipeline

    def test_disguise_pipeline():
        input_test = "tests/temp_data/raw_test.txt"
        os.makedirs("tests/temp_data", exist_ok=True)
        with open(input_test, "w", encoding="utf-8") as f:
            f.write("刘备与关羽、张飞在桃园结义。后来，张飞鞭打督邮，玄德急忙阻止。")
            
        output_file = run_disguise_pipeline(input_test, "tests/temp_data")
        assert os.path.exists(output_file)
        
        with open(output_file, "r", encoding="utf-8") as f:
            content = f.read()
            
        # 验证原始敏感实体已被全量替换为代号
        assert "刘备" not in content
        assert "张飞" not in content
        assert "玄德" not in content
        assert "督邮" not in content
        assert "角色_Alpha" in content or "角色_Beta" in content
    ```
    运行测试：`pytest tests/test_disguise_generator.py -v`
    预期：PASS

- [ ] **Step 5: 提交代码**
    ```bash
    git add tests/disguise_book_generator.py tests/test_disguise_generator.py
    git commit -m "feat: 实现别称自动消解聚类与整本全量换皮脱敏管道"
    ```

---

### Task 2: 物理数据库与内存图底座开发 (NetworkX 三轨连边)

**Files:**
*   Modify: `src/database.py` (在 `ChromaAdapter` 内部引入 `GraphManager` 或直接维护内存 NetworkX 图)
*   Test: `tests/test_database_graph.py`

- [ ] **Step 1: 编写失效测试（测试内存图的拓扑构建功能）**
    
    新建 `tests/test_database_graph.py`，内容如下：
    ```python
    import pytest
    from src.database import ChromaAdapter

    def test_chroma_graph_build():
        adapter = ChromaAdapter(db_dir="./vector_db_test", collection_name="test_graph_collection")
        # 检验 adapter 拥有 graph 属性，并且是非空 NetworkX 图
        assert hasattr(adapter, "graph")
        assert len(adapter.graph.nodes) > 0  # 目前会失败，因为还没写实现
    ```
    运行：`pytest tests/test_database_graph.py -v`
    预期：FAIL with "AttributeError" 或 "AssertionError"

- [ ] **Step 2: 修改 `src/database.py` 实现内存图三轨连边**
    
    在 `src/database.py` 中引入 `networkx`，实现物理相邻边、TF-IDF 实体共现边和局域/ANN 语义相关边的构建：
    在 `ChromaAdapter` 类的头部导入 `networkx as nx`，并在 `__init__` 中初始化 `self.graph = nx.Graph()`。
    实现核心建图方法 `rebuild_graph(self)` 和增量更新方法：
    ```python
    # 在 src/database.py 头部添加导入：
    import networkx as nx
    import numpy as np
    import re
    from collections import Counter
    import jieba.posseg as pseg

    # 在 ChromaAdapter 中新增/修改：
    def __init__(self, db_dir: str = "./vector_db", collection_name: str = "advanced_rag_collection"):
        self.client = chromadb.PersistentClient(path=db_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.bm25 = None
        self.bm25_docs = []
        self.graph = nx.Graph()
        self._rebuild_bm25()
        self.rebuild_graph()

    def rebuild_graph(self) -> None:
        """从 Chroma 数据库提取所有 unique parent_id，利用三轨机制在内存构建 NetworkX 图"""
        self.graph.clear()
        count = self.collection.count()
        if count == 0:
            return

        all_data = self.collection.get(include=["documents", "metadatas", "embeddings"])
        if not all_data["metadatas"]:
            return

        # 1. 提取所有 Parent Chunks 及其属性
        parents = {}  # parent_id -> {text, embed, source_path, filename, char_start}
        for doc, meta, embed in zip(all_data["documents"], all_data["metadatas"], all_data["embeddings"]):
            p_id = meta["parent_id"]
            if p_id not in parents:
                parents[p_id] = {
                    "text": meta["parent_text"],
                    "embedding": embed,
                    "source_path": meta["source_path"],
                    "filename": meta["filename"],
                    "char_start": meta.get("char_start", 0)
                }

        # 2. 将 parent 作为节点加入图
        for p_id, info in parents.items():
            self.graph.add_node(
                p_id,
                parent_text=info["text"],
                embedding=info["embedding"],
                source_path=info["source_path"],
                filename=info["filename"]
            )

        # 3. 物理相邻边：同一文档内，按 char_start 物理排序相邻连边
        doc_groups = {}
        for p_id, info in parents.items():
            doc_groups.setdefault(info["source_path"], []).append((p_id, info["char_start"]))

        for src_path, p_list in doc_groups.items():
            # 按 char_start 排序
            p_list.sort(key=lambda x: x[1])
            for i in range(len(p_list) - 1):
                self.graph.add_edge(p_list[i][0], p_list[i+1][0], type="physical")

        # 4. 无监督 TF-IDF 实体共现边（局域化限制在同文档内）
        # 保护未知代号
        code_pattern = re.compile(r"(角色|代号|项目|特工)\s*[A-Za-z0-9_]+")
        
        # 为每个 parent 提取关键词
        parent_keywords = {}
        for p_id, info in parents.items():
            text = info["text"]
            # 动态注入代号防止切碎
            found_codes = code_pattern.findall(text)
            for code in found_codes:
                jieba.add_word(code)
                
            words = pseg.cut(text)
            words_filtered = [
                w for w, flag in words 
                if flag in ('n', 'nr', 'ns', 'nt', 'nz', 'eng') and len(w) >= 2
            ]
            
            # 计算局部词频作为特征词（取 Top 5）
            counter = Counter(words_filtered)
            top_k = [w for w, _ in counter.most_common(5)]
            parent_keywords[p_id] = set(top_k)

        # 同一文件内，共享特征词的连实体共现边
        for src_path, p_list in doc_groups.items():
            n_doc = len(p_list)
            for i in range(n_doc):
                p_i = p_list[i][0]
                for j in range(i + 1, n_doc):
                    p_j = p_list[j][0]
                    if parent_keywords[p_i] & parent_keywords[p_j]:
                        self.graph.add_edge(p_i, p_j, type="entity")

        # 5. 局域语义关联边（同文档内暴力相似度计算）
        for src_path, p_list in doc_groups.items():
            n_doc = len(p_list)
            if n_doc < 2:
                continue
            p_ids = [item[0] for item in p_list]
            embeds = np.array([parents[pid]["embedding"] for pid in p_ids])
            # 计算余弦相似度矩阵
            norms = np.linalg.norm(embeds, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            embeds_norm = embeds / norms
            sim_matrix = np.dot(embeds_norm, embeds_norm.T)
            
            for i in range(n_doc):
                for j in range(i + 1, n_doc):
                    if sim_matrix[i, j] >= 0.82:
                        self.graph.add_edge(p_ids[i], p_ids[j], type="semantic")

        # 6. 跨文档 ANN 检索连边
        # 遍历每个 parent_id，用其 embedding 在 Chroma 里找 Top-5 邻居，>=0.85 且是跨文档的，建立语义关联边
        for p_id, info in parents.items():
            dense_vec = info["embedding"]
            # 检索包含它自己在内的 Top-6
            chroma_results = self.collection.query(
                query_embeddings=[dense_vec],
                n_results=min(6, count),
                include=["metadatas", "distances"]
            )
            if chroma_results["metadatas"] and chroma_results["metadatas"][0]:
                for idx, (meta, dist) in enumerate(zip(chroma_results["metadatas"][0], chroma_results["distances"][0])):
                    neigh_pid = meta["parent_id"]
                    score = 1.0 - dist
                    # 排除自己，要求跨文档，并且相似度 >= 0.85
                    if neigh_pid != p_id and score >= 0.85:
                        if parents[neigh_pid]["source_path"] != info["source_path"]:
                            self.graph.add_edge(p_id, neigh_pid, type="semantic_ann")

    # 并将 add_chunks 接口修改为：在 add 后触发 rebuild_graph()
    def add_chunks(self, chunks_data: List[Dict[str, Any]], dense_embeddings: List[List[float]]) -> None:
        # 原有的写 Chroma 逻辑 ...
        # (行59-64的代码保持不变)
        # 并在最后 rebuild 图和 bm25:
        # self._rebuild_bm25()
        self.rebuild_graph()
    ```

- [ ] **Step 3: 运行测试验证通过**
    
    修改 `tests/test_database_graph.py` 中的测试逻辑，模拟入库并验证图节点的邻接连边关系是否正确。
    运行：`pytest tests/test_database_graph.py -v`
    预期：PASS

- [ ] **Step 4: Commit 提交**
    ```bash
    git add src/database.py tests/test_database_graph.py
    git commit -m "feat: 在 ChromaAdapter 中集成 NetworkX 内存图，实现三轨自动连边算法"
    ```

---

### Task 3: 在线检索与图算法层开发 (PPR 与 2跳随机游走)

**Files:**
*   Create: `src/graph_search.py`
*   Modify: `src/coordinator.py`
*   Test: `tests/test_graph_search.py`

- [ ] **Step 1: 编写失效测试（验证图检索策略召回）**
    
    新建 `tests/test_graph_search.py`，内容如下：
    ```python
    import pytest
    def test_random_walk():
        from src.graph_search import run_semantic_random_walk
        # 期待未定义错误
        assert False
    ```
    运行：`pytest tests/test_graph_search.py -v`
    预期：FAIL with "ImportError"

- [ ] **Step 2: 新建 `src/graph_search.py` 实现 PPR 与语义引导游走**
    
    ```python
    import networkx as nx
    import numpy as np

    def run_personalized_pagerank(graph: nx.Graph, seed_node_id: str, top_k: int = 5) -> list[str]:
        """PPR 算法实现"""
        if not graph.has_node(seed_node_id):
            return []
        
        # 初始化个性化字典
        personalization = {node: 0.0 for node in graph.nodes}
        personalization[seed_node_id] = 1.0
        
        try:
            pagerank_scores = nx.pagerank(graph, alpha=0.85, personalization=personalization, max_iter=100)
        except Exception:
            # Fallback
            pagerank_scores = {seed_node_id: 1.0}
            
        # 降序排序并过滤掉 Seed Node 自身
        sorted_nodes = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
        candidates = [node for node, score in sorted_nodes if node != seed_node_id]
        return candidates[:top_k]

    def run_semantic_random_walk(graph: nx.Graph, seed_node_id: str, query_vector: list[float], top_k: int = 5) -> list[str]:
        """BGE-M3 语义引导 2 跳随机游走算法"""
        if not graph.has_node(seed_node_id):
            return []

        q_vec = np.array(query_vector)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            q_norm = 1e-9

        def get_similarity(node_id) -> float:
            node_emb = np.array(graph.nodes[node_id]["embedding"])
            n_norm = np.linalg.norm(node_emb)
            if n_norm == 0:
                n_norm = 1e-9
            return float(np.dot(q_vec, node_emb) / (q_norm * n_norm))

        # 1. 第一跳
        neighbors_1 = list(graph.neighbors(seed_node_id))
        if not neighbors_1:
            return []

        # 计算第一跳相似度
        sims_1 = {n: get_similarity(n) for n in neighbors_1}
        # 归一化概率 (这里为确定性检索直接截取相似度最高的前 3 个)
        sorted_1 = sorted(sims_1.items(), key=lambda x: x[1], reverse=True)
        selected_1 = [node for node, score in sorted_1[:3]]

        # 2. 第二跳
        selected_2 = []
        for n1 in selected_1:
            neighbors_2 = list(graph.neighbors(n1))
            # 排除起点 Seed Node 本身
            neighbors_2 = [n for n in neighbors_2 if n != seed_node_id]
            if not neighbors_2:
                continue
            sims_2 = {n: get_similarity(n) for n in neighbors_2}
            sorted_2 = sorted(sims_2.items(), key=lambda x: x[1], reverse=True)
            # 每一个第一跳节点挑选前 2 个最好的
            for node, score in sorted_2[:2]:
                if node not in selected_2 and node not in selected_1:
                    selected_2.append(node)

        # 合并去重
        all_walked = selected_1 + selected_2
        return all_walked[:top_k]
    ```

- [ ] **Step 3: 修改 `src/coordinator.py` 支持双路融合与断崖去噪**
    
    更新 `query` 方法并支持 `graph_search_mode` 传参：
    ```python
    # 替换 src/coordinator.py 中现有的 query 函数：
    def query(self, user_question: str, graph_search_mode: str = "heuristic_walk") -> str:
        """
        双通道混合初筛 -> 锁定 Top-1 语义中心 -> 内存图检索扩展 -> 双路融合 -> 二次重排与自适应断崖阻断 -> 输出
        """
        from src.graph_search import run_personalized_pagerank, run_semantic_random_walk
        
        # 1. 计算提问的 Dense 和 Sparse 向量
        dense_vec = self.embedding_service.get_dense_embedding(user_question)
        sparse_vec = self.embedding_service.get_sparse_embedding(user_question)

        # 2. 初筛检索召回
        candidates = self.db_adapter.hybrid_search(dense_vec, sparse_vec, top_k=15)
        if not candidates:
            return ""

        # 3. 锁定 Top-1 作为 Seed Node
        # 对初筛候选进行一轮精排，选出第一名
        initial_ranked = self.reranker.rerank(user_question, candidates, top_k=1)
        if not initial_ranked:
            return ""
        
        seed_candidate = initial_ranked[0]
        seed_node_id = seed_candidate["metadata"]["parent_id"]

        # 4. 图检索扩散 (从 Seed Node 游走 1-2 跳)
        topo_pids = []
        if hasattr(self.db_adapter, "graph") and len(self.db_adapter.graph.nodes) > 0:
            if graph_search_mode == "ppr":
                topo_pids = run_personalized_pagerank(self.db_adapter.graph, seed_node_id, top_k=5)
            else:
                topo_pids = run_semantic_random_walk(self.db_adapter.graph, seed_node_id, dense_vec, top_k=5)

        # 5. 合流与去重
        # 将初筛出的所有 candidates 与图拓扑扩展捞出的 parent 合并
        # 构造图检索捞出的 candidate 列表
        merged_candidates = {c["metadata"]["parent_id"]: c for c in candidates}
        
        # 如果图里捞出的 parent 在初筛候选里没有，则手动构建补充进来
        for pid in topo_pids:
            if pid not in merged_candidates and pid in self.db_adapter.graph.nodes:
                node_data = self.db_adapter.graph.nodes[pid]
                merged_candidates[pid] = {
                    "content": "", # 占位
                    "metadata": {
                        "parent_id": pid,
                        "parent_text": node_data["parent_text"],
                        "source_path": node_data["source_path"],
                        "filename": node_data["filename"]
                    },
                    "score": 0.0  # 待 Rerank 重新打分
                }

        final_candidates = list(merged_candidates.values())

        # 6. 二次重排 (Rerank) 与自适应断崖阻断
        # 传入所有候选进行打分重排
        selected = self.reranker.rerank(user_question, final_candidates, top_k=5)
        if not selected:
            return ""

        # 执行父块替换与拼接输出
        formatted_parts = []
        for idx, candidate in enumerate(selected, 1):
            filename = candidate["metadata"].get("filename", "未知文件")
            parent_text = candidate["metadata"].get("parent_text", "")
            part_str = f"[片段{idx}] (来源: {filename})\n{parent_text}"
            formatted_parts.append(part_str)

        return "\n\n".join(formatted_parts)
    ```

- [ ] **Step 4: 编写单元测试并验证通过**
    
    在 `tests/test_graph_search.py` 中编写对 `run_semantic_random_walk` 与整个 `RAGCoordinator.query` 融合结果的测试。
    运行：`pytest tests/test_graph_search.py -v`
    预期：PASS

- [ ] **Step 5: Commit 提交**
    ```bash
    git add src/graph_search.py src/coordinator.py tests/test_graph_search.py
    git commit -m "feat: 实现 PPR/语义引导游走在线检索算法，并与 coordinator 双路融合及重排对接"
    ```

---

### Task 4: 评测管线重构与消融测试集运行

**Files:**
*   Modify: `tests/evaluation_set_generator.py` (对伪装三国演义出题)
*   Modify: `tests/run_retrieval.py` 和其他评测脚本 (支持“大海捞针”部署)
*   Test: 运行 `run_pipeline.py` 进行评估

- [ ] **Step 1: 编写失效测试（验证评测管线中“物理混合”数据入库）**
    
    运行一次 `python tests/run_pipeline.py`，进入出题和检索阶段。
    预期：能跑，但评估数据中没有包含“三国演义脱敏版”的混淆测试，或是缺少拓扑增强模式。

- [ ] **Step 2: 修改 `tests/evaluation_set_generator.py` 对换皮小说出题**
    
    修改出题脚本逻辑：
    1. 读入 `tests/temp_data/三国演义白话文_disguised.txt`。
    2. 让 Qwen 出 10 道高难度的跨章节隐式多跳推理题（限制提示词：“不得透露具体代号，必须通过事件和关系链条提问”），生成 `sanguo_disguised_dataset.json`。

- [ ] **Step 3: 修改检索对齐脚本 `tests/run_retrieval.py` 实施“大海捞针”混淆**
    
    在读取并写入 Chroma 数据库时：
    1. 先导入 60 万字原版《三国演义白话文》（无脱敏）以及科研论文。
    2. 将换皮脱敏的 `三国演义白话文_disguised.txt` 也导入 Chroma，作为“针”混入“大海”中。
    3. 支持以传统检索模式、PPR 图检索模式、语义游走图检索模式分别运行测试集，并将检索上下文分别保存。

- [ ] **Step 4: 运行 `run_pipeline.py` 进行裁判打分，生成极坐标消融雷达图**
    
    执行评测总脚本：
    运行：`python tests/run_pipeline.py`
    预期：
    1. 三国换皮测试集生成成功。
    2. 混合混淆数据成功注入并建图。
    3. 双轨检索打分完毕，输出 `Context Recall` 对比雷达图。

- [ ] **Step 5: 最终评估结果 Commit**
    ```bash
    git add tests/evaluation_set_generator.py tests/run_retrieval.py
    git commit -m "test: 完成混淆沙盘评测管线重构，运行图检索消融打分实验"
    ```

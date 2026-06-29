# 带权剪枝局部子图 PPR 检索方案实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重构 `advanced-rag` 中的 PPR 图检索模式，通过构建带权异质图与检索时 2-Hop 边权重剪枝，彻底解决全局超级枢纽词引起的拓扑膨胀和无关噪声问题。

**Architecture:** 
1. 在 [database.py](file:///E:/project/advanced-rag/src/database.py) 中，在重建内存图谱时统计名词特征的全局词频，为物理边、语义边和基于 IDF 赋权的实体边赋予 `'weight'` 属性，并进行 $\exp(4 \cdot W_{base})$ 指数拉伸。
2. 在 [graph_search.py](file:///E:/project/advanced-rag/src/graph_search.py) 中，重构 [run_personalized_pagerank](file:///E:/project/advanced-rag/src/graph_search.py#L16)，使其能够依据剪枝阈值 `edge_threshold` 提取 Seed 的 2-Hop 带权子图，并在子图上执行阻尼系数 $\alpha=0.85$ 的 `nx.pagerank`。

**Tech Stack:** Python, NetworkX, NumPy, pytest

---

### Task 1: 内存图数据底座边权重化与 IDF 机制改造

**Files:**
- Modify: `src/database.py:60-216`
- Test: `tests/test_database_graph.py`

- [ ] **Step 1: 在 `tests/test_database_graph.py` 中编写验证边权重的测试**
  在测试类末尾添加新测试，构造带有常见词与罕见词的文本，测试图重建后是否正确生成了带权重的实体边，并且其权重经过了指数拉伸且常见词边权重远低于罕见词边。

  ```python
  def test_graph_edge_weights_and_idf():
      import shutil
      import tempfile
      from src.loader import DocumentLoader
      from src.splitter import SemanticParentChildSplitter
      from src.embedding import LocalEmbeddingService
      from src.database import ChromaAdapter
      from src.coordinator import RAGCoordinator

      tmpdir = tempfile.mkdtemp()
      try:
          loader = DocumentLoader()
          emb = LocalEmbeddingService(device="cpu")
          splitter = SemanticParentChildSplitter(embedding_service=emb, threshold=None, child_size=30)
          db = ChromaAdapter(db_dir=tmpdir)
          coordinator = RAGCoordinator(loader, splitter, emb, db, None)

          # 写入包含高频词"刘备"和罕见词"督邮"的测试文档
          # 块1：刘备 督邮 怒鞭
          # 块2：刘备 督邮 刁难
          # 块3：刘备 娶妻 孙尚香
          test_file = os.path.join(tmpdir, "test_weight_doc.txt")
          with open(test_file, "w", encoding="utf-8") as f:
              f.write("角色 刘备 和 角色 督邮 在这里，怒鞭督邮。\n\n"
                      "角色 刘备 被 角色 督邮 刁难了。\n\n"
                      "角色 刘备 娶了 角色 孙尚香。")
          
          coordinator.add_file(test_file)
          graph = db.graph
          
          # 验证实体边权重
          # 寻找仅通过常见词"刘备"相连的实体边（块3和块1之间仅有"刘备"）
          # 寻找通过罕见词"督邮"相连的实体边（块1和块2之间有"督邮"和"刘备"）
          node_1_id = None
          node_2_id = None
          node_3_id = None
          for node_id, data in graph.nodes(data=True):
              text = data.get("parent_text", "")
              if "怒鞭督邮" in text:
                  node_1_id = node_id
              elif "刁难" in text:
                  node_2_id = node_id
              elif "娶了" in text:
                  node_3_id = node_id

          assert node_1_id is not None
          assert node_2_id is not None
          assert node_3_id is not None

          # 块1与块2包含"督邮"（罕见），块1与块3仅包含"刘备"（高频）
          edge_1_2 = graph.get_edge_data(node_1_id, node_2_id)
          edge_1_3 = graph.get_edge_data(node_1_id, node_3_id)

          # 必须包含带有指数拉伸的 weight 属性
          assert "weight" in edge_1_2
          assert "weight" in edge_1_3
          assert edge_1_2["weight"] > edge_1_3["weight"]
      finally:
          shutil.rmtree(tmpdir)
  ```

- [ ] **Step 2: 运行测试以验证其失败**
  Run: `pytest tests/test_database_graph.py -k test_graph_edge_weights_and_idf -v`
  Expected: FAIL (AssertionError 或 KeyError: 'weight' 不存在)

- [ ] **Step 3: 修改 `src/database.py` 以实现特征词 IDF 统计与边权重拉伸**
  在 `ChromaAdapter.rebuild_graph` 中：
  1. 收集所有的特征提取名词（来自 `pseg.cut` 的结果），计算全局 `DF` 字典：
     ```python
     all_entity_words = []
     for pid in self.graph.nodes:
         text = self.graph.nodes[pid].get("parent_text", "")
         words = []
         if text:
             for word, flag in pseg.cut(text):
                 word_stripped = word.strip()
                 if word_stripped and (flag.startswith('n') or flag == 'eng'):
                     words.append(word_stripped)
         # 去重，每个父块对一个词只计 1 次频次
         all_entity_words.append(set(words))
     
     # 统计全局频次 DF
     df_counter = Counter()
     for word_set in all_entity_words:
         df_counter.update(word_set)
     
     # 计算 IDF 函数
     N_nodes = len(self.graph.nodes)
     def get_idf(word):
         df_val = df_counter.get(word, 0)
         if df_val == 0:
             return 0.0
         return float(np.log(1.0 + N_nodes / df_val))
     ```
  2. 修改实体共现边计算逻辑，将权重写入：
     ```python
     # 实体共现边基础权重
     shared_words = set(keywords[u]) & set(keywords[v])
     sum_idf = sum(get_idf(w) for w in shared_words)
     w_base = min(1.0, sum_idf * 0.2)
     weight = float(np.exp(4 * w_base))
     self.graph.add_edge(u, v, type="entity", weight=weight)
     ```
  3. 修改物理相邻边，写入权重：
     ```python
     w_base = 0.3
     weight = float(np.exp(4 * w_base))
     self.graph.add_edge(u, v, type="physical", weight=weight)
     ```
  4. 修改局域与跨文档语义边，写入权重：
     ```python
     w_base = float(sim_matrix[i, j])
     weight = float(np.exp(4 * w_base))
     self.graph.add_edge(u, v, type="semantic", weight=weight)
     ```
     跨文档：
     ```python
     w_base = float(sim)
     weight = float(np.exp(4 * w_base))
     self.graph.add_edge(u, v, type="semantic", weight=weight)
     ```

- [ ] **Step 4: 运行测试验证其通过**
  Run: `pytest tests/test_database_graph.py -k test_graph_edge_weights_and_idf -v`
  Expected: PASS

- [ ] **Step 5: 提交 Task 1 代码**
  ```bash
  git add src/database.py tests/test_database_graph.py
  git commit -m "feat: 增加异质边 IDF 赋权与指数拉伸计算"
  ```

---

### Task 2: 实现动态带权剪枝局部子图 PPR

**Files:**
- Modify: `src/graph_search.py:16-45`
- Test: `tests/test_graph_search.py`

- [ ] **Step 1: 在 `tests/test_graph_search.py` 中编写验证带权 2-Hop 子图剪枝的测试**
  在文件里添加新测试，构造一个大图并包含强弱不同的边。从 Seed 开始游走，验证当设置了较强门控时，弱边所连的邻居没有被包含在子图和 PPR 最终计算结果中。

  ```python
  def test_weighted_subgraph_pagerank():
      g = nx.Graph()
      # 节点
      g.add_node("seed", embedding=[1.0, 0.0])
      g.add_node("strong_1hop", embedding=[0.9, 0.1])
      g.add_node("weak_1hop", embedding=[0.1, 0.9])
      
      # 边，一个强，一个弱（强边通过通行门控，弱边不行）
      g.add_edge("seed", "strong_1hop", weight=10.0) # 大于门控阈值 7.3
      g.add_edge("seed", "weak_1hop", weight=2.0)    # 小于门控阈值 7.3
      
      # 从 seed 出发进行带权剪枝 PPR
      # 阈值为 7.3 (相当于 w_base >= 0.5)
      res = run_personalized_pagerank(g, "seed", top_k=5, edge_threshold=7.3)
      
      pids = [node for node, score in res]
      assert "strong_1hop" in pids
      assert "weak_1hop" not in pids # 弱边邻居在提取子图时被剪枝忽略
  ```

- [ ] **Step 2: 运行测试验证其失败**
  Run: `pytest tests/test_graph_search.py -k test_weighted_subgraph_pagerank -v`
  Expected: FAIL (AssertionError: "weak_1hop" 仍然出现在了结果中)

- [ ] **Step 3: 修改 `src/graph_search.py` 中 `run_personalized_pagerank` 的实现**
  1. 重构方法签名，支持传入 `edge_threshold: float = 0.0`：
     ```python
     def run_personalized_pagerank(graph: nx.Graph, seed_node_id: str, top_k: int = 5, edge_threshold: float = 0.0) -> list[tuple[str, float]]:
     ```
  2. 实现带权 2-Hop 提取逻辑：
     ```python
     if seed_node_id not in graph:
         return []
     if len(graph.nodes) <= 1:
         return []
         
     # 提取强 1 跳邻居
     neighbors_1st = set()
     for n in graph.neighbors(seed_node_id):
         edge_data = graph.get_edge_data(seed_node_id, n)
         # 防御性：若旧测试图无 weight，取无穷大确保其不被过滤
         weight = edge_data.get("weight", float('inf'))
         if weight >= edge_threshold:
             neighbors_1st.add(n)
             
     # 提取强 2 跳邻居
     neighbors_2nd = set()
     for n1 in neighbors_1st:
         for n in graph.neighbors(n1):
             edge_data = graph.get_edge_data(n1, n)
             weight = edge_data.get("weight", float('inf'))
             if weight >= edge_threshold and n != seed_node_id:
                 neighbors_2nd.add(n)
                 
     # 组装节点集合并提取子图
     target_nodes = {seed_node_id} | neighbors_1st | neighbors_2nd
     if len(target_nodes) < 3: # 节点太少，不足以构成合理的转移链条
         return []
         
     sub_graph = graph.subgraph(target_nodes).copy()
     ```
  3. 执行 `nx.pagerank`：
     ```python
     personalization = {node: 0.0 for node in sub_graph.nodes}
     personalization[seed_node_id] = 1.0
     
     try:
         scores = nx.pagerank(sub_graph, alpha=0.85, personalization=personalization, max_iter=100, weight='weight')
     except Exception:
         return []
         
     sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
     result = [(node, score) for node, score in sorted_nodes if node != seed_node_id]
     return result[:top_k]
     ```

- [ ] **Step 4: 运行测试验证其通过**
  Run: `pytest tests/test_graph_search.py -k test_weighted_subgraph_pagerank -v`
  Expected: PASS

- [ ] **Step 5: 提交 Task 2 代码**
  ```bash
  git add src/graph_search.py tests/test_graph_search.py
  git commit -m "feat: 重构 PPR 实现带权 2-Hop 局部子图求解"
  ```

---

### Task 3: 联调对接与全量单元测试校验

**Files:**
- Modify: `src/graph_search.py:171-173`
- Test: `tests/test_graph_search.py`

- [ ] **Step 1: 对接 [GraphPostRetriever](file:///E:/project/advanced-rag/src/graph_search.py#L111) 的门控参数**
  在 `src/graph_search.py` 的 `query_graph_enhanced` 中调用 `run_personalized_pagerank` 时，传入 `edge_threshold = np.exp(4 * 0.5)`（即基础权重阈值 $0.5$ 对应的拉伸阈值 $\approx 7.389$）：
  ```python
  elif graph_search_mode == "ppr":
      import numpy as np
      edge_thr = float(np.exp(4 * 0.5))
      graph_scores = run_personalized_pagerank(self.db_adapter.graph, seed_node_id, top_k=5, edge_threshold=edge_thr)
  ```

- [ ] **Step 2: 运行全量单元测试**
  Run: `pytest tests/test_graph_search.py -v`
  Expected: 所有测试（包括之前的熔断和重排旧测试）全部 PASS。

- [ ] **Step 3: 运行消融分析验证评估结果**
  运行消融分析或流程评估程序，确认是否能正确召回“怒鞭督邮”相关 Chunk 且无“三顾茅庐”噪声干扰。
  Run: `python tests/test_ppr_thresholds.py`

- [ ] **Step 4: 提交 Task 3 代码**
  ```bash
  git add src/graph_search.py
  git commit -m "feat: 联调对接基于 0.5 基础门控的 PPR 局部检索"
  ```

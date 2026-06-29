# 基于 IDF 边剪枝与 2 跳局部子图 PPR 降噪检索方案设计

本项目针对 `advanced-rag` 中的 PPR (Personalized PageRank) 检索模式进行重构。主要解决由于“常见词（超级枢纽词）”导致的拓扑膨胀、PPR 分值被节点总数稀释、以及引入大量非因果关联噪声（如“刘备相关的无关故事”）的问题。

---

## 🏗️ 架构与组件设计

本次设计主要涉及两个核心组件的改造：
1. **图数据底座 ([database.py](file:///E:/project/advanced-rag/src/database.py))**：引入全局特征词 IDF 统计，给异质边（物理、语义、实体）赋予带特异性的权重，并进行非线性指数拉伸。
2. **图检索组件 ([graph_search.py](file:///E:/project/advanced-rag/src/graph_search.py))**：从全局大图游走，升级为**“检索时动态带权剪枝 2-Hop 提取”**，仅在极小的局部子图上运行 PPR 游走，并显式传入 `alpha=0.85` 控制探索与传送概率。

---

## 🛠️ 具体技术实现方案

### 1. 图数据层改造

在 [ChromaAdapter.rebuild_graph](file:///E:/project/advanced-rag/src/database.py#L60) 中构建图谱时，所有的边都必须计算一个代表相关性强度的 `'weight'` 属性：

#### 1.1 全局特征词 IDF 计算
在提取父块分词特征后，首先计算所有特征名词在整个图数据库中的出现频次（Document Frequency，简称 DF）。定义特征词 $e$ 的逆文档频率 $IDF(e)$ 为：
$$IDF(e) = \log\left(1 + \frac{N}{DF(e)}\right)$$
其中 $N$ 为图中节点（父块 Chunk）的总数。

#### 1.2 边的基础权重（Base Weight）赋值
不同类型的边具有不同的基础关联度：
* **物理相邻边 (Physical Edges)**：代表相邻物理上下文的顺承关系，设定固定权重 $W_{base} = 0.3$。
* **局域与跨文档语义边 (Semantic Edges)**：直接使用两个节点嵌入之间的余弦相似度作为权重：$W_{base} = Sim(u, v)$。
* **实体共现边 (Entity Edges)**：如果两个 Chunk 共享特征词，其权重由共享词的 IDF 累加并缩放限制在 $[0.2, 1.0]$ 区间：
  $$W_{base} = \min\left(1.0, \sum_{e \in Shared} IDF(e) \times 0.2\right)$$
  对于“刘备”这类极高频词，因为 $DF(e)$ 很大，其 $IDF(e)$ 趋近于 0，使得仅通过“刘备”相连的边基础权重极低。

#### 1.3 边权重指数拉伸（拉大强弱差距）
在将边加入图谱前，对基础权重进行指数级放大，压制弱关联：
$$\text{weight} = \exp(4 \cdot W_{base})$$
在 NetworkX 中写入边属性：
```python
self.graph.add_edge(u, v, weight=weight, type=...)
```

---

### 2. 检索层改造

在 [graph_search.py](file:///E:/project/advanced-rag/src/graph_search.py) 中，重构 [run_personalized_pagerank](file:///E:/project/advanced-rag/src/graph_search.py#L16)：

#### 2.1 动态“带权剪枝”子图提取
在检索阶段，接收 Query 并通过 Rerank 确定 Seed 节点（一阶段排序首位）。
设定边权重通行阈值 $\text{edge\_threshold} = \exp(4 \cdot 0.5) \approx 7.3$（即基础权重 $\ge 0.5$）。
1. **寻找强 1 跳邻居**：遍历 Seed 的邻接边，仅保留连边 $\text{weight} \ge \text{edge\_threshold}$ 的邻居节点。
2. **寻找强 2 跳邻居**：从上述 1 跳邻居出发，再次遍历它们的邻接边，仅保留连边 $\text{weight} \ge \text{edge\_threshold}$ 的邻居节点（排除 Seed 自身）。
3. **提取子图**：将 Seed 节点、强 1 跳和强 2 跳邻居的节点集合取出，在内存中动态提取子图克隆：
   `sub_graph = graph.subgraph(target_nodes).copy()`。

#### 2.2 防御退化机制
若提取出的子图节点总数小于 3 个，说明 Seed 节点在局部处于拓扑孤立状态，不进行图游走，直接返回空，避免强行计算引入无关概率值。

#### 2.3 子图 PPR 求解
在微型子图 `sub_graph` 上运行 Personalized PageRank：
```python
scores = nx.pagerank(
    sub_graph, 
    alpha=0.85,                      # 85% 探索概率，15% 强制传送回 Seed
    personalization={seed_node_id: 1.0}, # 以 Seed 为能量源
    weight='weight'                  # 依据指数拉伸后的权重分配转移概率
)
```
过滤掉 Seed 节点自身后，对分数进行降序排序，截取 Top-K 节点返回。

---

## 🧪 验证与评估方案

### 1. 验证目标
* **消除超级词噪音**：在检索“张飞怒鞭督邮原因”时，能够成功召回「桃园结义」和「督邮刁难刘备」，并且**不引入**只和刘备共现但与事件无关的「三顾茅庐」等 Chunk。
* **低延迟响应**：检索阶段的子图提取 and PPR 计算延迟控制在 5 毫秒以内。

### 2. 验证方法
1. **运行现有评估集**：运行 [tests/evaluate_results.py](file:///E:/project/advanced-rag/tests/evaluate_results.py) 评估 PPR 模式在消融实验中的 Faithfulness 和 Accuracy 得分变化。
2. **对比基准**：期望本方案能使 PPR Graph RAG 的内容精确度（Accuracy）明显超越重构前的 5.2，并且逼近或超越 Heuristic Walk 的 6.7。

# src

> 实现工业级混合检索与分层切片 RAG 服务端 API，支持语义分片、重排精选和图搜索增强。

该模块是一个完整的 RAG（检索增强生成）服务后端，核心流程包括：文档加载与格式转换 → 基于语义的父子块切分 → Dense/Sparse 双通道向量生成 → ChromaDB 持久化存储 → 混合检索（BM25 + 向量相似度）→ CrossEncoder 精排重排 → 父块上下文拼接输出。此外提供图增强检索模式，通过 NetworkX 构建父子块关联图，支持 Personalized PageRank 和语义引导随机游走进行二跳邻居召回。模块还包含性能基准测试脚本、Obsidian 日志批量导入工具及日志清洗工具。

## Files

### `src/app.py`

FastAPI Web 服务入口，定义 API 路由与数据模型，启动时延迟初始化全局 RAGCoordinator。

- `QueryRequest` (class) - 查询请求体，包含 query 字符串和 top_k 参数
- `AddFileRequest` (class) - 文件导入请求体，包含 file_path
- `GraphQueryRequest` (class) - 图检索请求体，含 query 和 graph_search_mode（如 heuristics_walk）
- `startup_event` (function) - FastAPI 启动钩子，延迟加载并实例化 RAGCoordinator 及其所有依赖组件
- `retrieve` (function) - POST /retrieve 路由，执行混合检索与重排，支持 text/plain 响应头适配 Dify
- `add_file` (function) - POST /add_file 路由，将本地文件导入向量库
- `retrieve_graph` (function) - POST /retrieve_graph 路由，执行图增强混合检索

### `src/coordinator.py`

RAG 编排核心类，串联加载、切分、嵌入、入库和查询的完整管线。

- `RAGCoordinator` (class) - 全局 RAG 编排器，持有 loader/splitter/embedding_service/db_adapter/reranker
- `add_file` (function) - 完整入库流程：读文件→切父子块→补齐元数据(source_path/filename/char_start/end)→批量生成 Dense 向量→持久化入库
- `query` (function) - 查询管线：计算 query 的 dense/sparse 向量→混合检索 Top15→重排精选 Top5→父块替换并格式化输出

### `src/database.py`

向量存储适配器层，封装 ChromaDB 持久化、BM25 倒排索引重建和 NetworkX 图构建。

- `VectorStoreAdapter` (class) - 抽象基类，定义 add_chunks 和 hybrid_search 接口
- `ChromaAdapter` (class) - ChromaDB 适配器，维护内存 BM25 索引和 NetworkX 关联图
- `_rebuild_bm25` (function) - 从 Chroma collection 提取文档与元数据，用 jieba 分词重建 BM25Okapi 倒排索引
- `rebuild_graph` (function) - 从 Chroma 提取所有 chunks，构建 NetworkX 图并建立三轨连边（父子/兄弟/语义相似）
- `_add_weighted_edge` (function) - 添加带权边，遵循强关联优先策略（取最大值）

### `src/embedding.py`

本地嵌入服务，封装 BAAI/bge-m3 模型的稠密向量、批量向量和稀疏词频向量生成。

- `LocalEmbeddingService` (class) - 基于 SentenceTransformer 的嵌入服务，支持 CPU/CUDA
- `get_dense_embedding` (function) - 单条文本生成 1024 维稠密向量
- `get_dense_embeddings_batch` (function) - 批量生成稠密向量，分 batch_size=8 批次处理并打印进度防死锁怀疑
- `get_sparse_embedding` (function) - 利用 tokenizer 提取 tokens 并统计词频，返回稀疏向量（词频字典）

### `src/splitter.py`

语义父子块切分器，基于 mistune AST 解析 Markdown 结构进行层次化分片。

- `SemanticParentChildSplitter` (class) - 语义父子块切分器，支持 threshold 和 child_size 参数
- `create_parent_child_chunks` (function) - 核心切分方法，根据扩展名判断文件类型并生成父子块结构
- `render_inline` (function) - 递归渲染 mistune AST 子节点为 Markdown 文本（处理 text/emphasis/strong/link/image）
- `render_ast_node_to_md` (function) - 将完整 AST 节点渲染回 Markdown，支持 heading/paragraph/block_code/list/table

### `src/loader.py`

多格式文档加载器，支持 txt/md 直读、PDF 通过 MinerU API 转换及 fitz 回退。

- `DocumentLoader` (class) - 文档加载器，统一输出 Markdown 字符串
- `_convert_via_markitdown` (function) - 通过子进程调用 conda 环境中的 markitdown CLI 将文件转为 Markdown
- `load` (function) - 统一加载入口：txt/md 直读；pdf 走 MinerU API（pipeline 后端），失败则回退 fitz(PyMuPDF)

### `src/reranker.py`

精排重排服务，使用 CrossEncoder 对候选块进行语义打分与自适应断崖截断。

- `RerankerService` (class) - 基于 BAAI/bge-reranker-v2-m3 的精排服务
- `rerank` (function) - 精排核心：计算 query-candidate 对分数→降序排序→自适应语义断崖截断（相邻落差>cliff_threshold）

### `src/graph_search.py`

图检索算法实现，提供 Personalized PageRank 和语义引导随机游走两种二跳召回策略。

- `cosine_similarity` (function) - 计算两个稠密向量的余弦相似度
- `run_personalized_pagerank` (function) - 以 seed 为唯一能量源，在 2-hop 剪枝子图上计算 Personalized PageRank，返回 Top-K 父块
- `run_semantic_random_walk` (function) - 语义引导随机游走：第1跳取 Top-3 邻居（按 query 相似度），第2跳各取 Top-2，合并去重后返回 Top-K

### `src/benchmark.py`

性能基准测试脚本，对比重构前后（串行单条 vs 批处理并行）的嵌入生成耗时。

- `LegacyEmbeddingWrapper` (class) - 模拟重构前串行单条调用模式的包装类
- `run_benchmark` (function) - 运行对比测试：100 句测试文本，分别用串行和并行模式切分，输出耗时与提速倍数

### `src/import_obsidian.py`

批量导入工具脚本，遍历目录下的 .md 文件并通过 API 录入向量库。

- `import_markdown_logs` (function) - 递归扫描目录下所有 .md 文件，逐个 POST /add_file 接口导入并统计成功/失败数

### `src/log_cleaner.py`

日志清洗工具，仅保留 .log 文件中 [WARNING]/[ERROR] 行。

- `LogCleaner` (class) - 日志清洗器
- `clean_log` (function) - 读取 .log 文件，过滤保留以 [WARNING] 或 [ERROR] 开头的行

### `src/database.py.bak`

database.py 的备份副本，内容一致，用于版本回退参考。

### `src/graph_search.py.bak`

graph_search.py 的备份副本，内容一致，用于版本回退参考。

## Key Concepts

- **父子块切分 (Parent-Child Chunking)**: 先切分出大块（parent）保持上下文完整性，再对每个父块细粒度切分为子块（child）用于检索；检索命中子块后回退到其父块文本作为最终上下文，兼顾召回精度与语义连贯性
- **混合检索 (Hybrid Search)**: 同时使用 Dense 向量相似度（BGE-M3）和 Sparse BM25 词频索引进行初筛召回 Top-15，结合两者优势覆盖同义词泛化与精确匹配场景
- **Adaptive Cliff Cutoff**: 
- **图增强检索 (Graph Search)**: 以召回种子节点为中心构建二跳子图，通过 Personalized PageRank 或语义引导随机游走扩散召回关联父块，捕捉跨文档的隐式知识关联
- **离线优先架构**: 所有模型（BGE-M3、bge-reranker-v2-m3）和向量库均使用本地路径与 HF_HUB_OFFLINE=1 配置，确保无外网环境可独立运行

## Internal Relationships

- `app.py` → `coordinator.py`: app.py 在启动事件中将 RAGCoordinator 及其全部依赖实例化并挂载到 app.state.coordinator
- `coordinator.py` → `database.py`: coordinator.add_file() 通过 db_adapter.add_chunks() 持久化入库；query() 通过 db_adapter.hybrid_search() 初筛召回
- `coordinator.py` → `splitter.py`: coordinator.add_file() 调用 splitter.create_parent_child_chunks() 进行语义切分
- `coordinator.py` → `embedding.py`: coordinator 通过 embedding_service.get_dense_embeddings_batch() 批量生成向量，query() 中分别获取 dense/sparse 向量
- `coordinator.py` → `reranker.py`: coordinator.query() 在混合检索后调用 reranker.rerank() 进行精排重排
- `database.py` → `graph_search.py`: database.py 的 ChromaAdapter 维护 NetworkX 图（rebuild_graph），graph_search.py 的算法函数直接接收该图执行 PageRank/随机游走召回
- `app.py` → `graph_search.py`: retrieve_graph 路由在运行时 from src.graph_search 导入 GraphPostRetriever，配合 coordinator 执行图检索
- `loader.py` → `splitter.py`: loader.load() 输出 Markdown 文本后由 splitter 进行结构化切分
- `import_obsidian.py` → `app.py`: 通过 HTTP POST /add_file 远程调用 app.py 提供的 API 完成批量导入

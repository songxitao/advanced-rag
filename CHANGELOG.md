# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-07-19

### Removed
- 根目录内部文档与运行时缓存（HANDOFF.md、walkthrough.md、cache.json 等）
- .agents/、.claude/、.codegraph/ 等 AI 内部配置从 Git 跟踪中移除
- scratch/ 一次性实验脚本目录（保留 check_dependencies.py）

### Changed
- 根目录文档归位：DIFY_RAG_GUIDE.md → docs/deploy/，README_GNN.md → docs/gnn/
- 启动脚本移入 scripts/ 目录
- tests/ 拆分：评测管线 → scripts/evaluation/，数据集 → scripts/data/，工具 → scripts/dev/
- 更新 .gitignore，完善 20+ 条忽略规则
- 修复 pyproject.toml 构建后端为 setuptools.build_meta

### Added
- GitHub Issue/PR 模板（bug_report、feature_request、PULL_REQUEST_TEMPLATE）
- .handoffs/ 本地手稿存档体系（gitignored，按时间排序）

## [1.0.0] - 2025-01-27

### Added

#### 文档解析与加载
- 完成 DocumentLoader 开发，支持 txt、pdf、docx、srt 格式文档加载（[ed0f31c]）
- 集成 MarkItDown 命令行，统一转换 docx/xlsx/xls 为标准 Markdown 结构（[ac47ce7]）
- 接入 MinerU 微服务，实现 PDF-Markdown 语义切片重构，升级 GPU 向量计算加速（[c7463d6]）

#### 语义切片与去重
- 实现语义化切片与父子关系生成器（splitter.py），150 词 Child 块 + 800 词 Parent 块（[f0ba277]）
- 实现 Parent Size Guarantee 并跑通全部测试（[acb5d41]）
- 集成 mistune AST 解析器，提取 Markdown 块节点（[9ea4b6c]）
- 滑动窗口哈希去重机制，Embedding 推理阶段提速达 4.79 倍（[50bc170]）

#### 向量表征与存储
- 实现本地向量表征服务 BGE-M3 并添加测试用例（[fdfac96]）
- 配置 Embedding 与 Reranker 底层模型的 GPU 加速与批量预测推理（[da61a5d]）
- 实现抽象数据库适配器与 ChromaDB/BM25 双通道混合检索（[466f0e5]）

#### 检索与重排
- 实现重排服务（CrossEncoder）和总编排调度器，强制离线加载模型（[932b6cd]）
- 实现 FastAPI 检索与文档入库端点（[559f55e]）
- 支持通过环境变量动态指定推理设备，增加启动交互式选择（[a164108]）
- 支持 `text/plain` 响应格式的检索 API（[1f4f614]）

#### 图增强检索（PPR / GNN）
- 在 ChromaAdapter 中集成 NetworkX 内存图，实现三轨自动连边算法（[b988470]）
- 实现 PPR/语义引导游走在线检索算法，与 coordinator 双路融合及重排对接（[d4c5650]）
- 将图检索与融合逻辑解耦为独立外挂类（[1b5490b]）
- 重构检索合流层，实现非对称 3+2 通道配额、两路独立免检图谱及 0.5 熔断门控（[646c043]）
- 重构检索融合层，引入向量 1.5 断崖检测、0.5 熔断门控、解耦独立打分（[400813b]）
- 增加异质边 IDF 赋权与指数拉伸计算（[b63e0fc]）
- 重构 PPR 实现带权 2-Hop 局部子图求解（[c47238c]）
- 联调对接基于 0.5 基础门控的 PPR 局部检索（[1dc78fa]）
- 优化 PPR 门控参数至 0.3 以平衡去噪与多跳召回（[e9846ae]）

#### 评测管线
- 添加 Naive 与 Advanced RAG 双轨量化评测方案设计文档及执行计划（[3a82823], [99aa0f7]）
- 添加测试数据集问答对生成脚本（[3da5dd2]）
- 添加 RAG 检索与数据对齐脚本（[7025ab5]）
- 添加管道控制管理器菜单脚本（[85b59dc]）
- 完成混淆沙盘评测管线重构，获得全隐式出题下的图检索消融打分雷达图（[6b97c74]）
- 出题脚本引入关键词绝杀，检索评测脚本开启 CUDA 显存加速（[c65802f]）
- 完成图通路评测与 GNN 空间紧邻重构，获得最新消融打分雷达图（[dd2055b]）

#### 脱敏与辅助工具
- 实现别称自动消解聚类与整本全量换皮脱敏管道（[6da42d7]）

### Changed

#### 性能优化
- 优化向量检索双通道多线程性能，引入精排断崖自适应截断及向量化批量处理（[20f2371]）
- 动态化项目根目录与子进程执行路径，优化阶段遍历顺序（[9d1a861]）
- 在子进程运行环境中显式添加 PYTHONPATH 以确保模块导入路径（[483c83b]）

#### 部署与稳定性
- 强制 RAG 服务运行在 CPU 以避免显存溢出错误（[adb3b15]）
- 修复脱敏伪装管道的 JSON 健壮性、单字防误杀与测试 Mock 可移植性（[2eeff7e]）
- 为 `add_file` 接口添加空白及无切片物理校验抛出，防止扫描件静默导入失败（[c3c4820]）

### Added

#### 开发与部署工具
- 添加依赖检测脚本并完成库的安装（[b17d6fd]）
- 添加一键启动本地 RAG 服务的批处理 bat 脚本（[a205f84]）
- 启用 VitePress Mermaid 插件用于文档图表渲染（[01a309a]）

#### 文档
- 创建通俗易懂的项目 README 指南（[e86927e]）
- 添加 Naive / Advanced RAG 双轨量化评测方案设计文档与执行计划（[3a82823], [99aa0f7]）
- 增加带权子图 PPR 检索降噪方案设计文档与实现计划（[aade321], [cda5650]）
- 更新评测方案执行计划，增加双语英文论文与段落语义切分出题（[c1d98b1]）
- 更新 Dify 部署手册以配合交互式推理设备选择（[959abd6]）
- 将 GNN 图拓扑重构更新作为 README_GNN.md 独立输出（[713ad0d]）
- 更新 README 项目介绍与 Handoff 会话移交文档（[4f31275]）

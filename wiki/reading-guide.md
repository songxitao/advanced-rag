# Reading Guide

这是一个基于 Python 的 RAG（检索增强生成）服务端项目，核心能力包括语义分片、重排精选和图搜索增强。阅读顺序遵循「入口→配置→文档→核心实现→测试/集成」的路径，优先理解数据流与架构决策，再深入具体实现。

## Step 1: 项目概览与启动方式 (~10 min)

**Files:** `README.md`, `package.json`, `ONBOARDING_MANUAL.md`

先看 README 了解项目定位、依赖安装和启动命令；package.json 确认 Node 侧工具链（VitePress 文档站）；ONBOARDING_MANUAL.md 是官方入职手册，包含环境搭建步骤。这步建立全局认知：项目怎么跑起来、目录结构长什么样。

## Step 2: AI Agent 工作区规则与规划 (~10 min)

**Files:** `.agents/AGENTS.md`, `.agents/plans/markitdown-integration.md`

AGENTS.md 定义了 Agent 读码行为规范，了解后你才知道后续 handoff 文件的写作约定；markitdown-integration.md 是功能规划文档，理解 markitdown 集成的设计意图，这对理解 src/ 中相关代码的来龙去脉很关键。

## Step 3: 架构决策与演进记录 (~15 min)

**Files:** `DIFY_RAG_GUIDE.md`, `README_GNN.md`, `.claude/handoffs/2026-07-02-201301-gnn-rag-refactor.md`

DIFY_RAG_GUIDE.md 给出 RAG 整体架构设计；README_GNN.md 解释图神经网络增强的设计动机；最早的 GNN-RAG handoff 记录了一次关键重构，读它能理解为什么代码长这样（而非应该怎样）。这些文档是理解「业务意图」的钥匙。

## Step 4: 应用入口与主流程 (~20 min)

**Files:** `src/app.py`

这是服务端 API 的唯一入口（134 行），从路由注册、中间件到请求处理链路完整呈现。重点看：路由如何映射到业务逻辑、RAG 管线（切片→检索→重排→生成）在代码中的组织方式、配置来源。这是理解数据流的枢纽文件。

## Step 5: 核心业务模块深入 (~30 min)

**Files:** `src/`

以 app.py 中 import 的模块为线索，逐层阅读 src/ 下的核心代码。重点关注：语义分片逻辑、重排精选算法、图搜索检索增强组件的实现。对照 DIFY_RAG_GUIDE.md 中的架构图，验证代码是否忠实反映了设计文档。

## Step 6: 手记与重构记录（按时间线） (~15 min)

**Files:** `.claude/handoffs/2026-07-02-201606-gnn-pr-refactoring.md`, `.claude/handoffs/2026-07-10-215815-markitdown-integration.md`

这些 handoff 文件是开发过程的「黑匣子」，记录了每次重构的决策、踩坑和遗留问题。按时间顺序读，能还原代码演化的因果链——为什么某个函数被拆了、某个参数加了默认值。

## Step 7: 测试与集成验证 (~15 min)

**Files:** `tests/`, `scratch/`

33 个测试文件是理解 API 契约最精确的文档——每个 test 就是一次「输入→期望输出」的规格说明。scratch/ 目录包含端到端集成演示，可实际运行以验证本地环境配置是否正确。

## Step 8: 文档站与缓存 (~5 min)

**Files:** `docs/`, `docs/.vitepress/cache/deps/`

docs/ 是 VitePress 文档源码，可 `npm run docs:dev` 本地预览。deps 目录下的 JS 文件（如 sanitize-url、主题索引）是构建产物缓存，通常无需阅读，除非排查文档站构建问题。

## Tips

- 先跑通再深读：先用 ONBOARDING_MANUAL.md 的指引把项目启动起来，看到服务正常运行后再逐层阅读代码，效率远高于纯静态阅读。
- 以 src/app.py 为锚点做双向对照：入口文件向上对照架构文档、向下顺藤摸瓜到各模块，比漫无目的地浏览快得多。
- handoff 文件按时间倒序读可能更省力——最新的 handoff 往往总结了之前所有重构的结论，先读它再回溯旧文件事半功倍。
- tests/ 目录是 API 规格说明书，遇到不理解的业务逻辑时，先看对应测试用例中的输入输出示例。

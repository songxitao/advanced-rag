# RAG Evaluation Pipeline Refactoring and Ablation Test Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重构 RAG 评测管线并运行图检索消融打分实验（包含传统 RAG、PPR 图 RAG、语义游走图 RAG、以及 Naive RAG 作为基线四者的量化对比）。

**Architecture:** 
1. 在微服务 `src/app.py` 中新增 `/retrieve_graph` 接口，整合 `GraphPostRetriever` 支持图增强多模式检索。
2. 重构 `tests/evaluation_set_generator.py` 读取混淆三国演义文本，通过本地 Qwen 接口生成 10 道针对伪装代号角色的跨章节隐式多跳问题，将 QA 保存至 `tests/temp_data/test_sanguo_dataset.json`。
3. 重构检索与评测流水线，包括“大海背景噪声”（原著各章节文本 + 学术论文）及“伪装情节文本”的混合入库。
4. 运行完整非交互式评测，使用指定 Python 虚拟环境并输出对比雷达图。

**Tech Stack:** FastAPI, ChromaDB, NetworkX, Pytest, Matplotlib, Qwen3.6-35b

---

### Task 1: Web Service API & Tests
**Files:**
- Modify: `src/app.py`
- Modify: `tests/test_app.py`

- [ ] **Step 1.1: Modify `src/app.py` to add `/retrieve_graph`**
  导入 `from src.graph_search import GraphPostRetriever`，定义 `GraphQueryRequest` 模型，新增 POST `/retrieve_graph` 接口，支持根据参数 `graph_search_mode` (如 `heuristic_walk`、`ppr` 等) 进行图增强检索，返回 `{ "context": context }`。

- [ ] **Step 1.2: Add integration tests for `/retrieve_graph`**
  在 `tests/test_app.py` 中新增 `test_api_retrieve_graph` 测试函数，使用 `unittest.mock.patch` 模拟 `GraphPostRetriever.query_graph_enhanced` 返回值，验证接口的输入校验与返回响应格式。

- [ ] **Step 1.3: Run app tests using specified python path**
  运行：`E:\conda\envs\deepseek-ocr\python.exe -m pytest tests/test_app.py -v`
  预期：全部 PASSED。

---

### Task 2: Obfuscation Evaluation Set Generation
**Files:**
- Modify: `tests/evaluation_set_generator.py`

- [ ] **Step 2.1: Implement text segment chunks parsing & template configuration**
  修改 `tests/evaluation_set_generator.py`，使之读取 `tests/temp_data/三国演义白话文_disguised.txt`。支持提取段落 chunks 并采样。
  设计强关联的跨章节隐式多跳问题生成提示词，使 Qwen 接口能够生成不需要出现 `[角色_X]` 却能隐式指代该角色的多跳问题。

- [ ] **Step 2.2: Save generated evaluation set to test_sanguo_dataset.json**
  请求本地 Qwen 生成 10 道题目，将结果保存至 `tests/temp_data/test_sanguo_dataset.json`。若请求网络或 JSON 解析失败增加健壮重试逻辑。

---

### Task 3: Multi-RAG Retrieval Ablation Running
**Files:**
- Modify: `tests/run_retrieval.py`

- [ ] **Step 3.1: Implement "Needle in a Haystack" DB Indexing**
  在 `tests/run_retrieval.py` 中，读取原著 `E:/project/pyltp-books-master/pyltp-books-master/mybooks/Book/三国演义白话文` 目录下的所有 txt 章节，并将其逐个注入库；读取换皮脱敏的 `tests/temp_data/三国演义白话文_disguised.txt` 混入写入库；同时保留原有学术论文。增加防重复写入校验。

- [ ] **Step 3.2: Perform Retrieval on the 4 ablation pipelines**
  读取 `tests/temp_data/test_sanguo_dataset.json`。对每个问题，分别在 Naive RAG、传统 RAG (none 模式)、PPR 图 RAG (ppr 模式)、以及语义游走图 RAG (heuristic_walk 模式) 下执行检索，获取各自的 context，并将结果保存至 `tests/temp_data/retrieval_sanguo_results.json`。

---

### Task 4: Answer Generation & Local Qwen-Judge Evaluation
**Files:**
- Modify: `tests/generate_answers.py`
- Modify: `tests/evaluate_results.py`
- Modify: `tests/run_pipeline.py`

- [ ] **Step 4.1: Modify `tests/generate_answers.py` for 4 pipelines**
  使其支持读取 `retrieval_sanguo_results.json`，分别为 `naive_context`、`traditional_context`、`ppr_context`、以及 `walk_context` 生成四种模型回答，保存到 `tests/temp_data/answer_sanguo_results.json`。

- [ ] **Step 4.2: Modify `tests/evaluate_results.py` for 4-track scoring & Radar Chart**
  使其支持计算并量化打分 4 个变体；并在雷达图上绘制 4 条不同颜色的曲线；保存图表到 `tests/outputs/evaluation_radar.png` 和 `tests/evaluation_radar.png`。保存分数至 `tests/temp_data/evaluation_sanguo_scores.json`。

- [ ] **Step 4.3: Add non-interactive option in `tests/run_pipeline.py`**
  修改 `tests/run_pipeline.py` 支持 `--all` 参数以进入非交互模式。同时将内部执行的脚本路径更新为针对三国的各子脚本路径（或复用但适配三国路径参数）。

---

### Task 5: Execute and Commit
- [ ] **Step 5.1: Run full pipeline**
  执行命令：`E:\conda\envs\deepseek-ocr\python.exe tests/run_pipeline.py --all`
  验证所有评测结果落盘以及雷达图生成。

- [ ] **Step 5.2: Git commit**
  使用 `git add` 和 `git commit` 提交工作。

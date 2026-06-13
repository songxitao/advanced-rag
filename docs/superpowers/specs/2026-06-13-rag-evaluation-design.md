# 📊 Naive vs Advanced RAG 双轨量化评测方案设计文档 (Design Spec)

本文档旨在详述 Naive RAG (对照组) 与 Advanced RAG (实验组) 双盲无偏量化评测体系的架构与实现流程。本体系采用分段离线流水线（Segmented Offline Pipeline）架构设计，并引入了中英双语多目标文档评测以及中立段落级语义切片，保障评测无偏性与专业度。

---

## ⚖️ 1. 评测实验组与对照组设计

| 评估维度 | 对照组：Naive RAG (e:/project/rag) | 实验组：Advanced RAG (e:/project/advanced-rag) |
| :--- | :--- | :--- |
| **切片策略** | Naive `RecursiveCharacterTextSplitter` (800字符) | 语义边界自适应切片 + 父子块替换 (Parent-Child) |
| **检索机制** | 单通道稠密向量检索 (BGE-M3) | Dense (BGE-M3) + Sparse (BM25) 双通道并发检索 |
| **重排与截断**| 无重排，直接返回初筛结果 | BGE-Reranker-v2-m3 精排 + 语义断崖自适应截断 |
| **物理数据库** | `E:/project/rag/vector_db` | `E:/project/advanced-rag/vector_db` |

### 📚 双语对齐数据底座 (Target Documents)
为了综合评估中英文环境下的表现，我们引入两篇不同语言的专业学术论文：
1. **中文论文**：`E:/desktop/code/New folder/paper song.docx`
2. **英文论文**：`E:/project/DeepSeek-OCR/ocr_results/44221625_LI LEI/44221625_LI LEI_merged.docx`

> ⚠️ **数据底座对齐前提**：测试运行前，Naive RAG 和 Advanced RAG 数据库需同步导入这两篇论文。Advanced RAG 库中保留已有的 40 多篇日常开发/调试日志（作为干扰背景噪音），模拟“大海捞针”的多目标检索场景。

---

## 🧠 2. 评测模型角色分工

评测在本地 Llama-Server 路由端点 (`http://localhost:8080/v1`) 环境下运行：
* **出题官 / 打分裁判 (Judge LLM)**：`qwen3.6-35b-a3b-distilled-think` (Qwen 35B 思考大模型，温度 `0.0` 确保出题和评分标准的高度客观一致)。
* **做题考生 (Solver LLM)**：`gemma4-mtp-nothink` (Gemma 4 12B 无思考模型，响应速度快，代表中轻量应用模型的表现，温度 `0.7` 模拟生成多样性)。

---

## 🛠️ 3. 分段流水线整体架构

整个评测流程由四个隔离的子脚本和一个主控中心组成，均保存在 `tests/` 目录下：

```
tests/
├── run_pipeline.py          # 统一评测主控中心（CLI 菜单界面）
├── evaluation_set_generator.py # 阶段 1：中立段落级语义出题
├── run_retrieval.py         # 阶段 2：双语数据对齐与双轨检索提取
├── generate_answers.py      # 阶段 3：Solver 做题解答生成
└── evaluate_results.py      # 阶段 4：Ragas 裁判评分与雷达图产出
```

### 3.1 阶段一：中立段落级语义出题
* **脚本**：`tests/evaluation_set_generator.py`
* **无偏切片原理**：直接读取中文和英文 docx 文件，采用**“按段落自然边界进行语义合并”**的切片策略。连续拼接自然段落直到总字符数达到约 1000 字符合并为一个语义块（保证不斩断任何技术句子、图表引用或公式）。
* **出题分布**：从中文论文中随机抽取 15 个语义块，从英文论文中随机抽取 15 个语义块，共计 30 组 `(Question, Ground Truth)` 存入 `tests/test_dataset.json`。
* **Prompt 模板**：
  ```text
  你是一个严谨的学术评测出题官。请阅读以下从论文中提取的文本片段（可能是中文或英文），为其设计一个具体的技术性问题，并给出该问题在原文中能够直接印证的标准答案（Ground Truth）。如果原文是英文，请用英文出题和给出答案；如果是中文，请用中文出题和给出答案。
  
  【限制要求】：
  1. 问题必须针对文本中的核心技术细节、公式或实验结论，切忌泛泛而谈。
  2. 标准答案必须完全忠实于原文，不得夹杂任何外部知识。
  3. 请严格按照以下 JSON 格式输出，不要包含任何多余解释或 markdown 标记（如 ```json ... ```）：
  {
    "question": "问题内容",
    "ground_truth": "标准答案内容"
  }
  
  【原文片段】：
  {{source_context}}
  ```

### 3.2 阶段二：双语数据对齐与双轨检索
* **脚本**：`tests/run_retrieval.py`
* **原理**：
  1. 动态引入 `E:/project/rag/rag_engine.py` 与 `src.coordinator`。
  2. 检查两套 RAG 系统的 Chroma 库中是否索引了 `paper song.docx` 和 `44221625_LI LEI_merged.docx` 两篇文档，若缺失则自动触发索引。
  3. 载入 `test_dataset.json`，遍历问题，分别调用 Naive RAG 的 `search_with_context` 和 Advanced RAG 的 `RAGCoordinator.query`，提取上下文。
  4. 写入 `tests/retrieval_results.json`，包含 `naive_context` 与 `advanced_context`
  
### 3.3 阶段三：考生答题
* **脚本**：`tests/generate_answers.py`
* **原理**：调用 `gemma4-mtp-nothink`，针对每个问题，分别用 naive 和 advanced 的 context 生成两组答案 `naive_answer` 与 `advanced_answer`，存入 `tests/answer_results.json`。
* **Prompt 模板**：
  ```text
  请你扮演一个专业的答题助手。请结合我提供的【参考资料】，准确回答【问题】。
  
  【参考资料】：
  {{context}}
  
  【问题】：
  {{question}}
  
  【答题要求】：
  1. 必须优先使用【参考资料】中的事实进行回答，答案需要准确、精炼。
  2. 若资料中未提及相关信息，请直接回答：“参考资料中未提及相关信息，无法回答。”
  ```

### 3.4 阶段四：大模型裁判评分与可视化
* **脚本**：`tests/evaluate_results.py`
* **原理**：
  1. 将数据转换成 Ragas 接收的 Dataset。
  2. 将 Ragas 的 Evaluator LLM 桥接指向本地 Qwen-Think 端口。
  3. 测算 4 项黄金指标：`Faithfulness`（忠实度）、`Answer Relevance`（答案相关性）、`Context Recall`（检索召回率）、`Context Precision`（检索精准度）。
  4. 使用 `matplotlib` 绘制两组得分的对比雷达图，输出至 `tests/outputs/evaluation_radar.png`，保存详细打分表为 `tests/evaluation_scores.json`。

---

## 📊 4. 评测控制中心主菜单交互设计

主脚本 `tests/run_pipeline.py` 将提供极简 of CLI 菜单交互：
1. 自动执行依赖检测（`ragas`、`matplotlib`、`python-docx`、`jieba`）及 LLM 路由连通性。
2. 支持执行 1-4 任意单一阶段（使用 JSON 缓存），或一键自动顺序运行全部阶段。

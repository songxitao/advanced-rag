# Handoff: 端侧多模态 RAG 集成成果与简历级 Agent 设计计划

## 📅 Session Metadata
- **Last Modified**: 2026-07-05T22:25:00+08:00
- **Workspace Root**: [E:/project/advanced-rag](file:///E:/project/advanced-rag)
- **Target User**: 尖子 (称呼其为 “尖子”)

---

## 🎯 1. 核心集成成果 (Current State)

本 Session 成功打通并验证了 **“有限显存硬件资源下，PDF 高精度解析 ➡️ Markdown 分级语义切片 ➡️ GPU 加速向量化入库”** 的端到端链路：

1. **常驻微服务化 (FastAPI)**：
   - 部署并挂载了常驻 API 实例：
     - **`8010` 端口**：MinerU PDF 结构化解析微服务 (CPU 模式)。
     - **`8000` 端口**：Advanced RAG 后端微服务 (GPU/CUDA 模式)。
   - **消灭 9 秒冷启动**：消灭了 Python 命令行每次重新加载 7 个算法模型带来的 9 秒前置延迟，通过网络内存直通，实现实时秒级解析响应。
2. **高精度 Markdown 语义切片链路**：
   - 修正了 `DocumentLoader` 对 MinerU JSON 键名（不带后缀的文件名干，如 `demo1`）的提取对齐。
   - 修正了 `coordinator.py` 对 Markdown 切片的设备和后缀类型判定，使得 PDF 转换出的 Markdown 能够完整保留其 HTML 表格（`<table>`）与 LaTeX 公式（`$` / `$$`）层级，进入 `SemanticParentChildSplitter` 进行层级大纲切片。
3. **安全防崩溃 (Fail-safe)**：
   - 完美定位并修复了特殊英文字符连字（如 `ﬂ`）在 Uvicorn GBK 代码页控制台打印预览时引发的 `UnicodeEncodeError` 崩溃。

---

## 💎 2. 简历级亮点与 Agent 设计计划 (Resume Highlights)

如果将本项目写进简历，这是一套极具技术深度的 **“有限资源约束下的高性能端侧智能体检索系统（Edge-Agent RAG）”** 案例。其核心架构与工程亮点如下：

### 💡 亮点一：异构设备动态路由负载均衡 (CPU-GPU Heterogeneous Load Balancing)
- **设计痛点**：端侧消费级显卡（如 12GB RTX 40系）在运行本地大语言模型（如 Qwen-7B 占 6-8GB 显存）时，如果再并跑多模态 Layout 分析和 OCR，极易发生显存溢出（OOM）。
- **工程解法**：
  - 将 **PDF OCR 与版面分析（MinerU，高显存占用、延迟低敏感）** 异步路由至 **CPU** 上执行。因为知识库构建通常为前置异步动作，用 CPU 运行可以为在线生成留出 4GB+ 宝贵显存。
  - 将 **分片向量计算（Dense Embedding，高延迟敏感、显存占用小）** 路由至 **GPU (CUDA)** 执行。在 GPU 矩阵乘法加速下，将 **392 个 Chunks 的 Embedding 向量计算时间从数十秒缩短至 1.5 秒内**，完美避免了 CPU 吃满 100% 导致的系统卡死，保障了在线检索的敏捷响应。

### 💡 亮点二：去盘化内存直通与微服务解耦 (Zero-Disk-IO & Microservice Pipeline)
- **工程解法**：放弃了“调用命令行生成物理文件再读取”的传统耦合方式，通过 HTTP 协议将 PDF 原始二进制数据流 POST 传递给 MinerU 解析端，MinerU 在内存中完成版面重组后，直接以 JSON String 将 Markdown 文本返回给 RAG。**全程无磁盘 IO，避免了 Windows 下多进程读写冲突、文件锁死与高并发磁盘瓶颈**。

### 💡 亮点三：三轨融合的端侧多模态协同方案 (Three-Track Edge-Multimodal Pipeline)
- **ASR 轨**：使用 **SenseVoice** (阿里端侧高性能语音模型) 进行音频实时转写，具备极低时延与强悍的中英混合、方言识别表现，作为端侧 Agent 的轻量敏捷语音入口。
- **OCR 轨**：使用 **MinerU** 进行复杂 PDF（含跨栏表格、LaTeX 公式）的排版大纲提取，解决传统 OCR 提取乱码及结构丢失的问题。
- **RAG & 终端推理轨**：利用 **BGE-M3** 生成稠密向量 + **ChromaDB** 进行父子块混合检索 + 本地 **llama-server** 驱动本地大模型完成最终的 Agent 决策生成。

---

## 📋 3. 下一步行动计划 (Next Steps)

1. **一键启动自动化脚本配置**：
   - 鉴于 Windows 终端下对带 `$` 符号环境变量（如 `$env:`）的转义 Bug，编写基于 Python 或是 GBK 编码的 Windows `.bat` 批处理，封装为 `一键启动MinerU微服务.bat` 和 `一键启动RAG后端.bat`，简化开发部署流程。
2. **Dify API 接口对接**：
   - 配置 Dify 的自定义知识库 Loader 或者是自定义 Tool，将我们已经部署在 `http://127.0.0.1:8000/add_file` 的高精度 RAG API 注入 Dify，全面替换其原有的脏文本 PDF 加载器。
3. **加入 SenseVoice 语音交互模块**：
   - 在 Streamlit 交互前端中添加录音按钮，调用 SenseVoice 将语音命令实时转换为 Text，传给 Agent 执行检索问答。

---

## 📜 4. 机器读取规则 (Workspace rules. E:\project\advanced-rag\.agents\AGENTS.md)
我们已在项目下创建了 `AGENTS.md`，规定后续新加入的子智能体在开发时：
* 🚨 **必须优先使用 `codegraph` 的 `codegraph_explore` 动作** 进行项目符号与依赖寻址分析，禁止疯狂使用 `view_file` 大段读取代码，以节约 Token 并保持思路的绝对精确性。

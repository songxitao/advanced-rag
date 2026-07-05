# Workspace Rules for AI Agents

## 📋 开发与调试纪律
1. **禁止疯狂读取代码文件**：
   - 禁止在未定位问题前大量使用 `view_file` 或命令行 `cat` 动作反复阅读源文件。
   - **必须优先调用 `codegraph` 服务的工具（如 `codegraph_explore`）** 进行代码库的类、方法依赖和符号索引解算，精准定位有嫌疑的逻辑线后，仅针对特定文件的几行关键段做有限读取。

## ⚙️ 架构与部署规范
1. **PDF 高精度解析服务 (MinerU)**：
   - **必须使用 CPU 推理设备**。
   - 启动环境变量必须为 `RAG_DEVICE=cpu` 且 `MINERU_MODEL_SOURCE=local`。
   - 目的：规避 PDF 版面分析、OCR 与表格提取在 GPU 上带来的超高瞬时显存开销，保护 GPU 显存健康。
2. **RAG 后端检索与向量服务 (Advanced RAG)**：
   - **必须使用 GPU/CUDA 推理设备**。
   - 启动环境变量必须为 `RAG_DEVICE=cuda`。
   - 目的：实现对数百个父子块 Dense Embedding 批量向量计算的秒级计算，避免 CPU 核心满载 100% 造成系统卡死与发热。

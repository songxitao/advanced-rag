# Handoff: 将 markitdown 集成至 advanced-rag 文档加载与切分管线

## Session Metadata
- Created: 2026-07-10 21:29:01
- Project: E:\project\advanced-rag
- Branch: main
- Session duration: 1 hour

### Recent Commits (for context)
  - c7463d6 feat: 将 MinerU 微调 PDF-Markdown 切片重构为 GPU 加速
  - e9846ae tune: 优化 PPR 门槛值 0.3 推进去重召回
  - 1dc78fa feat: 对接门槛值 0.5 的 PPR 分流
  - c47238c feat: 重构 PPR 实现带权 2-Hop 路由图
  - 50bc170 feat: 优化 Task 1 的图检索，强化全局注意去重

## Handoff Chain
- **Continues from**: [2026-07-02-201606-gnn-pr-refactoring.md](./2026-07-02-201606-gnn-pr-refactoring.md)
  - Previous title: 2026-07-02-201606-gnn-pr-refactoring
- **Supersedes**: None

## Current State Summary
尖子与前任 Agent 讨论并完成了关于 RAG 进货价项目（针对型号复杂、价格敏感的表格和异构文档）的设计方案。当前任务是为 advanced-rag 项目引入 markitdown 作为 RAG 管线最前端的统一格式转换层。当前代码架构支持 is_markdown=False（正则切句）和 is_markdown=True（基于 AST 强保护表格 and 代码块）。我们需要将非 md 文件统一通过 markitdown 转成 Markdown，然后走后者的强保护路径。

## Architecture Overview
RAG 工作流为：loader.py 提取 -> splitter.py 切片 -> database.py 向量入库 -> app.py 混合检索调用。由于 splitter.py 具有基于 Markdown 语法树保护代码块和表格的能力，如果异构文档被前置转为 Markdown，则切片器能保持极佳的块划分。

## Critical Files
- src/loader.py: 文档加载器，负责将各种格式的文件读取并提取为文本流。
- src/splitter.py: 切句切块逻辑，其中 create_parent_child_chunks 方法根据 is_markdown 执行分支切分。
- src/coordinator.py: 检索协调器，串联加载、切片与入库流程。

## Key Patterns Discovered
- is_markdown=True 下有特殊的占位符转换和还原逻辑，专门用于保护 table 和 block_code 节点，免受语义切块的破坏。
- RAG 检索使用了 Chroma 向量检索与 BM25 关键词并发双通道检索。

## Work Completed

### Tasks Finished
- 在本地 Conda 成功部署了全新的虚拟环境 markitdown-env，完成了 markitdown[all] 全套可选依赖的本地安装与测试验证。
- 梳理了 splitter.py 的内部切分机制。

## Files Modified
None (仅进行前序调研与虚拟环境安装配置，未修改核心文件)

## Decisions Made
- 决定在 Loader 阶段将异构文件通过 markitdown 统一转为 Markdown。
- 决定对于转换后的 Markdown 文本在切块时强制走 is_markdown=True 分支，以利用语法树保护表格结构。

## Immediate Next Steps
1. 修改 src/loader.py，集成 markitdown，使其具备将 .pdf, .docx, .xlsx, .xls 统一转换为 Markdown 纯文本的能力。
2. 在 src/loader.py 中，对转换完的文本执行简单的清洗（如去除无意义硬换行、修复连字符粘连）。
3. 修改 src/coordinator.py 或集成层，将 is_markdown 强制传入为 True，保证统一走 mistune 语法树解析分支。

## Blockers/Open Questions
None.

## Deferred Items
None.

## Important Context
绝对不能直接将异构文件提取纯文本后走 is_markdown=False 语义相似度切分。因为 Excel 表格和 Word 表格会被切碎，丢失行列交叉语义，导致 LLM 对价格等数字信息的读取发生严重幻觉。必须先走 markitdown 生成 Markdown 管道表格形式，然后再进语法树保护。

## Assumptions Made
假设运行环境能直接通过 Conda 命令 conda run -n markitdown-env markitdown 调用或通过微服务接口通信。

## Potential Gotchas
PDF 文件提取出 Markdown 后，英文单词有可能发生空格丢失型粘连（如 WASEDAUNIVERSITY），需要在 loader.py 做初级正则容错清洗。

## Environment State

### Tools/Services Used
- Conda (markitdown-env 虚拟环境，包含 Python 3.12 及 markitdown 依赖)。

### Active Processes
None.

### Environment Variables
None.

## Related Resources
- README.md

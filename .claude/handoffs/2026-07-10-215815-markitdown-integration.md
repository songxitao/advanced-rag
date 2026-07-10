# Handoff: 将 markitdown 集成至 advanced-rag 文档加载与切分管线

## Session Metadata
- Created: 2026-07-10 21:58:15
- Project: E:\project\advanced-rag
- Branch: main
- Session duration: 1.5 hours

### Recent Commits (for context)
  - ac47ce7 feat: 集成 markitdown 统一 docx/xlsx/xls 格式转换为 Markdown
  - c7463d6 feat: 接入 MinerU 微服务与 PDF-Markdown 语义切片重构，升级为 GPU 向量计算加速
  - e9846ae tune: 优化 PPR 门控参数至 0.3 以平衡去噪与多跳召回
  - 1dc78fa feat: 联调对接基于 0.5 基础门控的 PPR 局部检索

## Handoff Chain
- **Continues from**: [2026-07-10-212901-integrate-markitdown.md](./2026-07-10-212901-integrate-markitdown.md)
  - Previous title: 将 markitdown 集成至 advanced-rag 文档加载与切分管线
- **Supersedes**: None

## Current State Summary
本会话已完全实现了将 Microsoft MarkItDown 文档转换工具链集成至 RAG 管线的 Loader 层。`.docx`、`.xlsx` 和 `.xls` 格式的文件均能统一通过子进程调用转换输出为 Markdown，从而允许下层 Splitter 高质量地应用 Markdown AST 进行语义切片。代码已经过单元测试及手动端到端验证，并完整提交入库。

## Codebase Understanding

### Architecture Overview
- `src/loader.py` 中的 `DocumentLoader` 增加私有方法 `_convert_via_markitdown`，通过子进程以 `conda run` 的形式调用独立虚拟环境中的 markitdown CLI。在 Windows 环境下增加了对于 Anaconda/Miniconda 默认目录的自适应 `conda.bat` 路径探测以保证健壮性。
- `src/coordinator.py` 的 `RAGCoordinator.add_file` 扩展了 `is_markdown` 后缀名数组，引导转换后的 Markdown 字符串流向高质量切片分支。
- `tests/test_loader.py` 优化了单体测试，将 `import docx` 放入测试用例内部并在首行添加 `pytest.importorskip("docx")`，允许在缺少 `python-docx` 库的环境里优雅跳过测试，防范了测试崩溃。

### Critical Files
| File | Purpose | Relevance |
|------|---------|-----------|
| [src/loader.py](file:///E:/project/advanced-rag/src/loader.py) | 文档加载提取 | 调用 markitdown CLI 实现多格式转换，彻底清除了顶层 `import docx` 依赖 |
| [src/coordinator.py](file:///E:/project/advanced-rag/src/coordinator.py) | 流程与检索协调 | 扩展 `is_markdown` 判断，使转换后的 Markdown 能进入 AST 语义分片流程 |
| [tests/test_loader.py](file:///E:/project/advanced-rag/tests/test_loader.py) | 提取模块单元测试 | 优化对 docx 测试的惰性导入，保证精简环境运行的兼容性 |

### Key Patterns Discovered
- **进程级解耦隔离依赖**：通过 `subprocess.run` 异步调用外部独立 conda 环境中的 CLI 工具。避免了 RAG 核心运行环境与文档解析器的底层臃肿库（如 Mammoth、lxml 等）发生严重版本冲突（如 Pydantic、numpy 等科学计算库版本冲突），保证了服务的安全和纯净。

## Work Completed

### Tasks Finished
- [x] 在 `src/loader.py` 中新增 `_convert_via_markitdown` 私有方法，支持 Windows 下 conda 路径自动探测寻址
- [x] 替换原 `.docx` 分支为 markitdown 转换，彻底删除顶层 `import docx` 模块，解除主环境的库硬依赖
- [x] 新增 `.xlsx` 和 `.xls` 处理分支，同样调用 markitdown 提取为 Markdown 表格
- [x] 扩展 `src/coordinator.py` 中的 `is_markdown` 后缀名列表
- [x] 优化 `tests/test_loader.py` 单体测试用例，应用惰性导入及 pytest.importorskip 防止精简环境整体测试崩溃
- [x] 端到端功能测试验证与 Pytest 自动化测试通过，变更已暂存并成功 Commit。

### Files Modified
| File | Changes | Rationale |
|------|---------|-----------|
| [src/loader.py](file:///E:/project/advanced-rag/src/loader.py) | 增加 `_convert_via_markitdown`；替换 docx 分支；增加 xlsx/xls 分支；删除 `import docx`。 | 集成 markitdown 多格式提取并解耦主环境硬依赖。 |
| [src/coordinator.py](file:///E:/project/advanced-rag/src/coordinator.py) | 扩展 `is_markdown` 列表。 | 允许新格式进入高精度 Markdown 语义切片逻辑。 |
| [tests/test_loader.py](file:///E:/project/advanced-rag/tests/test_loader.py) | 惰性加载 `import docx`，并在测试用例里加 `importorskip("docx")`。 | 防止主环境无包时测试集整体加载崩溃。 |

### Decisions Made
| Decision | Options Considered | Rationale |
|----------|-------------------|-----------|
| **采用进程级 conda 环境隔离** | 1. 在 RAG 主虚拟环境中安装 markitdown 依赖； 2. 通过微服务进行网络交互。 | RAG 主环境依赖（PyTorch, Chroma）与 markitdown 依赖有大量底层库版本重叠（如 pydantic, numpy），极易产生冲突；且文档量较少时，几百毫秒的子进程开销完全可以接受。 |
| **测试用例采用 pytest 惰性跳过** | 1. 强制在主环境里安装 docx； 2. 直接删除 docx 单元测试。 | 必须保证单体单元测试的回归完整性，通过惰性导入在主环境缺失包时自动 skip，既保证了主环境的运行安全性，又保留了完整用例。 |

## Pending Work

### Immediate Next Steps
1. 在生产/测试环境对大型复杂表格（合并单元格、多 Sheet 表）进行 markitdown 转换效果与语义切片召回精度的消融评测。
2. 观测多并发摄入文件时，大量 `subprocess` 触发 conda 执行的时延及系统负载开销。

### Blockers/Open Questions
- 无。

### Deferred Items
- 无。

## Context for Resuming Agent

### Important Context
- 外部虚拟环境名为 `markitdown-env`，其在本地 Windows 系统的 conda.bat 路径可能未被注册进 Path。已在 `_convert_via_markitdown` 内部进行了常见 Windows conda 路径的自适应 fallback 拦截（包含 `D:\program files\Miniconda\condabin\conda.bat` 等），调试时如提示找不到 conda，请先检查此 Fallback 数组。

### Assumptions Made
- 假设宿主机已安装 Anaconda/Miniconda 且内部构建好了 `markitdown-env` 虚拟环境。

### Potential Gotchas
- 在 Windows 下如果既无全局 Path，又不在 Fallback 安装路径内，执行解析会抛出 `RuntimeError("conda 命令未找到")`，请在 `src/loader.py` 的 common_paths 数组中添加对应的绝对路径。

## Environment State

### Tools/Services Used
- Conda (`markitdown-env` 虚拟环境，包含 `markitdown[all]`)
- Pytest (自动化回归测试)

### Active Processes
- 无。

### Environment Variables
- 无。

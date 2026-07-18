# root

> 项目根目录：存放文档、环境配置与启动脚本，不含业务代码。

根目录是项目的门面与运维层，不包含任何 Python 源码。它由三类文件组成：① 多份 README/手册（README.md, README_GNN.md, ONBOARDING_MANUAL.md, DIFY_RAG_GUIDE.md, HANDOFF.md）从不同受众视角（用户、Dify 集成者、接手工程师）描述同一套 RAG 引擎；② 前端文档站配置（package.json / package-lock.json），用 VitePress + Mermaid 渲染架构图；③ 运维脚本（.bat 批处理）负责 Conda 环境激活与 uvicorn 启动。cache.json 仅记录 CI 状态，无业务逻辑。

## Files

### `README.md`

项目主 README，面向通用读者，概述核心特性矩阵、架构图（Mermaid）、性能基准与快速上手步骤。

### `package.json`

文档站 npm 配置：定义 vitepress / mermaid / vitepress-plugin-mermaid 为 devDependencies，提供 docs:dev / docs:build / docs:preview 脚本。

- `scripts.docs:dev` (script) - 启动 VitePress 本地文档预览
- `scripts.docs:build` (script) - 构建静态文档站

### `package-lock.json`

VitePress 依赖的精确锁定文件，确保文档站构建可复现。

### `.gitignore`

排除 Python 缓存、node_modules、测试临时目录与敏感上手手册（ONBOARDING_MANUAL.md）。

### `DIFY_RAG_GUIDE.md`

面向 Dify 用户的集成教程：铁三角架构说明、host.docker.internal 避坑、HTTP+LLM 节点编排与最终闭环。

### `HANDOFF.md`

工程交接文档，记录 markitdown 集成变更、文件影响面、决策依据与后续待办，供接手工程师阅读。

### `ONBOARDING_MANUAL.md`

保姆级上手指南：环境预配置、一键启动（含 CPU/GPU 交互选择）、API 测试示例与 Dify 跨容器网络避坑。

### `README_GNN.md`

GNN/图拓扑增强版 RAG 的独立 README，侧重图谱游走、二次重排与学术检索案例，与主 README 互补。

### `cache.json`

CI/CD 缓存状态记录（last_run / jobs），无业务逻辑。

## Key Concepts

- **多受众文档策略**: 同一套 RAG 引擎通过 README.md（通用）、README_GNN.md（图谱增强版）、ONBOARDING_MANUAL.md（上手）、DIFY_RAG_GUIDE.md（Dify 集成）四份文档覆盖不同读者，避免单一大文档臃肿
- **VitePress + Mermaid 文档站**: 
- **Conda 环境隔离启动**: .bat 脚本统一封装 conda activate deepseek-ocr + uvicorn 启动流程，屏蔽开发机器差异
- **host.docker.internal 跨容器网络**: 

## Internal Relationships

- `ONBOARDING_MANUAL.md` → `README.md`: 上手手册引用主 README 的环境与 API 说明，作为其补充而非重复
- `DIFY_RAG_GUIDE.md` → `启动RAG服务.bat`: Dify 集成指南中提到的端口 :8000 由该批处理脚本负责拉起 uvicorn 服务
- `package.json` → `.gitignore`: package-lock.json 被 .gitignore 排除，确保文档站依赖锁定文件可复现但不污染仓库

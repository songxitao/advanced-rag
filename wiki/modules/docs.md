# docs

> VitePress 文档站源码与缓存，以及项目架构/评测的规划与设计规格说明书。

该目录包含两个独立但并行的内容体系：一是 docs/ 下的 VitePress 站点源码（config.js、index.md、intro.md、poiclaw/ 系列教程），用于将 Advanced RAG / PoiClaw Agent 的技术细节以 Markdown 文档形式对外发布；二是 docs/superpowers/plans/ 与 specs/ 下的 Markdown 规划文件，记录了拓扑增强 RAG、PPR 图检索、评测管线重构等功能的详细实现计划与设计规格。docs/.vitepress/cache/deps/ 是 VitePress 构建时自动生成的预构建依赖缓存（Vite Deps Cache），由 vite 在首次 dev/build 时自动生成，不应手动编辑。

## Files

### `docs/vitepress/config.js`

VitePress 站点全局配置：标题、导航栏、侧边栏层级结构、社交链接与页脚版权信息。

- `defineConfig` (function) - VitePress 配置入口，被 withMermaid 包装后导出。
- `withMermaid` (function) - vitepress-plugin-mermaid 提供的插件函数，为站点注入 Mermaid 图表渲染能力。

### `docs/index.md`

文档站首页（layout: home），展示 Hero 区、项目特性卡片与快速开始入口。

### `docs/intro.md`

项目总览页，介绍语义锚点父子替换、自适应语义断崖阻断、双通道并发三大核心算法原理。

### `docs/poiclaw/*.md`

PoiClaw Agent 重构实战系列教程（前言与 uv 上手、状态机与工作流剖析、JSON 容错解析与 Pytest、命令沙箱与敏感词过滤、Goal Loop 长时间闭环），以 Markdown + Mermaid 流程图/时序图形式编写。

- `repair_and_parse_json` (function) - 手写容错 JSON 解析器：尝试 json.loads → 正则提取 {…} → 尾部逗号修复，三级降级。
- `execute_cmd_safe` (function) - 安全命令执行器：黑名单拦截 + Windows chcp 65001 编码自愈 + subprocess.run 超时保护。
- `GoalLoop` (class) - 长时间目标闭环引擎，while 循环驱动 LLM 调用→工具执行→记忆追加的数据流。

### `docs/vitepress/cache/deps/package.json`

Vite Deps Cache 的模块声明文件，标记整个 deps 目录为 ESM 类型。

### `docs/vitepress/cache/deps/@braintree_sanitize-url.js`

@braintree/sanitize-url 库的预构建产物（Vite Deps Cache），提供 sanitizeUrl 函数用于 URL 清洗与协议白名单校验。

- `sanitizeUrl` (function) - 对输入 URL 做 HTML 实体解码、控制字符过滤、协议白名单校验，非法协议返回 about:blank。

### `docs/vitepress/cache/deps/@theme_index.js`

VitePress 默认主题组件的预构建产物，包含 Layout、VPFeatures、VPHomeHero 等所有默认 UI 组件的导入与导出。

- `Layout` (import) - 站点默认页面布局组件。
- `useMediaQuery` (function) - 响应式媒体查询 Hook，被 @theme_index 和 chunk-PUPTUIKR 共享使用。

### `docs/vitepress/cache/deps/_metadata.json`

Vite Deps Cache 的元数据索引，记录每个依赖包的源码路径、缓存文件路径、hash 值与是否需要 CommonJS 兼容（needsInterop）。

- `optimized` (object) - 键为包名，值为已预构建缓存的元信息；hash 用于判断缓存是否过期。
- `chunks` (object) - 被提取到公共 chunk（chunk-BUSYA2B4 等）中的共享模块列表。

### `docs/vitepress/cache/deps/chunk-BUSYA2B4.js`

Vite Deps Cache 的公共 chunk，导出 __commonJS helper，用于在 ESM 环境下兼容 CommonJS 模块。

- `__commonJS` (function) - 将 CommonJS 风格的模块包装为 ESM 可导入形式，内部使用单例缓存避免重复执行。

### `docs/vitepress/cache/deps/dayjs.js`

dayjs 库的预构建产物，提供轻量级日期解析与格式化能力。

- `dayjs` (function) - 日期解析/格式化核心函数，支持 locale、UTC、相对时间等。

### `docs/vitepress/cache/deps/vitepress___@vue_devtools-api.js`

@vue/devtools-api 的预构建产物，提供 Vue DevTools 通信桥接能力（含 rfdc 深拷贝等工具函数）。

- `__copyProps` (function) - 对象属性浅拷贝工具，用于 ESM 导出时的命名兼容处理。
- `rfdc2` (function) - 快速深拷贝函数（rfdc），支持 Map/Set/Buffer 等构造函数处理。

### `docs/vitepress/cache/deps/vitepress___@vueuse_core.js`

@vueuse/core 的预构建产物，导出 Composition API 工具集（useMediaQuery、useFetch、useLocalStorage 等数百个 composables）。

- `useMediaQuery` (function) - 响应式媒体查询 Hook。
- `useFetch` (function) - 基于 fetch 的异步请求 composable，内置 loading/error/数据状态管理。

### `docs/vitepress/cache/deps/vue.js`

Vue 3 运行时（esm-bundler）的预构建产物，导出 reactive、computed、ref、h、defineComponent 等全部核心 API。

- `reactive` (function) - 创建响应式对象。
- `computed` (function) - 派生只读计算属性。
- `ref` (function) - 创建响应式引用。
- `h` (function) - VNode 渲染函数。

### `docs/superpowers/plans/2026-06-22-evaluation-pipeline-refactor.md`

RAG 评测管线重构与消融实验计划：四路检索（Naive / 传统 / PPR / 语义游走）的量化对比方案，含 API 接口、混淆数据集生成、索引构建、评测打分与雷达图输出。

- `/retrieve_graph` (endpoint) - 新增 POST 接口，支持 graph_search_mode 参数进行图增强检索。
- `GraphPostRetriever` (class) - 图增强后检索器，整合 PPR/语义游走等检索策略。

### `docs/superpowers/plans/2026-06-22-topology-enhanced-rag-plan.md`

拓扑增强父子块 RAG 系统实现计划书：外挂式 Post-Retrieval Pipeline、NetworkX 三轨连边内存图、PPR 与语义游走双检索模式、指代消解脱敏预处理管线。

- `run_disguise_pipeline` (function) - 离线伪装脱敏管道：NER 实体提取 → LLM 别称聚类 → 逆向最大替换生成伪装文本。
- `ChromaAdapter.rebuild_graph` (method) - 基于 Chroma 元数据重建 NetworkX 内存图，实现三轨连边（物理/TF-IDF/语义）。

### `docs/superpowers/plans/2026-06-29-ppr-weighted-subgraph.md`

带权剪枝局部子图 PPR 检索方案实现计划：全局 IDF 边权重化、edge_threshold 阈值提取 2-Hop 带权子图、在子图上执行 PPR。

- `run_personalized_pagerank` (function) - 重构后的 PPR 函数：先按权重阈值剪枝提取局部子图，再在子图上执行 nx.pagerank。
- `edge_threshold` (variable) - 边通行阈值（≈7.3），用于过滤弱关联边，控制游走范围。

### `docs/superpowers/specs/2026-06-22-topology-enhanced-rag-design.md`

拓扑增强父子块 RAG 系统完整设计规格书：架构流向图、离线伪装脱敏管道、NetworkX 内存图节点/边设计、在线双路融合检索（PPR + 语义游走）、二次重排与断崖截断、防作弊评测体系。

- `Seed Node` (concept) - 初筛 Top-1 父块，作为拓扑游走的能量起点。
- `语义断崖截断` (mechanism) - 相邻重排分落差超阈值（1.5）时即时切断后续低相关文本。

### `docs/superpowers/specs/2026-06-29-ppr-weighted-subgraph-design.md`

IDF 边剪枝与 2 跳局部子图 PPR 降噪检索方案详细设计：IDF 公式、边基础权重赋值规则、指数拉伸公式、动态子图提取算法与退化防御机制。

- `IDF(e)` (formula) - 逆文档频率公式，log(1 + N/DF(e))，用于量化特征词稀有度。
- `weight = exp(4·W_base)` (formula) - 边权重指数拉伸公式，拉大强弱关联差距。

### `docs/superpowers/specs/2026-07-02-project-manual-generator-design.md`

CodeGraph 驱动的三驾马车说明书生成器设计：基于静态符号解算自动生成 README / ONBOARDING_MANUAL / AGENTS.md。

- `codegraph` (tool) - 本地代码静态分析工具，提供 init/status/explore 命令构建源码图谱索引。

## Key Concepts

- **Vite Deps Cache**: Vite 在首次 dev/build 时将 node_modules 中的 ESM/CJS 依赖预构建为浏览器可直接加载的 chunk，缓存在 docs/.vitepress/cache/deps/ 下；_metadata.json 管理缓存过期策略。
- **语义锚点父子替换**: 
- **自适应语义断崖阻断**: 
- **PPR 图检索**: 
- **IDF 边权重化**: 
- **2-Hop 带权剪枝子图**: 
- **指代消解脱敏管道**: 
- **Goal Loop 长时间闭环**: 
- **三驾马车文档生成器**: 

## Internal Relationships

- `docs/vitepress/config.js` → `docs/vitepress/cache/deps/@theme_index.js`: config.js 通过 VitePress 内部机制引用默认主题组件（经 @theme_index），侧边栏/导航配置驱动站点结构。
- `docs/vitepress/cache/deps/@theme_index.js` → `docs/vitepress/cache/deps/vue.js`: @theme_index 导入 Vue 3 核心 API（reactive/computed/ref/h），所有 VitePress 页面组件均基于 Vue 构建。
- `docs/vitepress/cache/deps/@theme_index.js` → `docs/vitepress/cache/deps/vitepress___@vueuse_core.js`: 主题组件使用 @vueuse/core 的 composables（如 useMediaQuery）实现响应式交互。
- `docs/vitepress/cache/deps/@theme_index.js` → `docs/vitepress/cache/deps/@braintree_sanitize-url.js`: 主题中处理 URL/链接时依赖 sanitizeUrl 做安全清洗。
- `docs/vitepress/cache/deps/_metadata.json` → `docs/vitepress/cache/deps/*.js`: _metadata.json 索引所有缓存 JS 文件的 hash，Vite 据此判断是否需要重新预构建。
- `docs/superpowers/specs/2026-06-22-topology-enhanced-rag-design.md` → `docs/superpowers/plans/2026-06-22-topology-enhanced-rag-plan.md`: 规格书（Spec）定义系统架构与算法，计划文件将其拆解为可执行的 checkbox 任务步骤。
- `docs/superpowers/plans/2026-06-29-ppr-weighted-subgraph.md` → `docs/superpowers/specs/2026-06-29-ppr-weighted-subgraph-design.md`: PPR 带权子图计划与对应设计规格书一一对应，计划中的 Step 1/2/3 直接实现 Spec 中定义的 IDF 边权重化与剪枝算法。
- `docs/poiclaw/goal.md` → `docs/poiclaw/sandbox.md`: Goal Loop 的 while 循环通过调用 execute_cmd_safe（sandbox）执行沙箱命令，两者共同构成 Agent 的决策-执行闭环。

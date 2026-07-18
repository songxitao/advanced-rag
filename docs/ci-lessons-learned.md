# CI 打地鼠 debug 全纪录 — 教训与经验

> 项目: advanced-rag (高精度语义切片与自适应断崖重排引擎)
> 时间: 2026-07-18
> 场景: 给已有本地 GPU 开发环境的 ML 项目，从零搭建 GitHub Actions CI

---

## 一、问题的本质

这不是一个"测试写得不好"的问题，而是一个**代码设计时没考虑 CI 环境差异**的问题。

本地开发环境 vs CI 环境的关键差异：

| 维度 | 本地 | CI (GitHub Actions) |
|------|------|-------------------|
| GPU | ✅ 有 (RTX 12GB) | ❌ 无 |
| 模型缓存 | ✅ 已下载 BGE-M3 等 | ❌ 无 |
| PyTorch | ✅ 已安装 | ❌ 需安装（4min+） |
| sentence-transformers | ✅ 已安装 | ❌ 需安装（2min+） |
| 操作系统 | Windows | Ubuntu |
| 文件路径 | `D:\xxx` | `/home/runner/xxx` |
| 执行时间预算 | 不限 | 越快越好 |

**教训 1：写 src 模块代码时就要想"这行 import 在 CI 能不能跑？"** 这是所有问题的总根源。

---

## 二、5 轮 CI 报错全记录

### 🔴 Round 1: conftest find_spec ValueError

**报错：** `ValueError: torch.__spec__ is None`

**触发时机：** pytest 收集阶段，加载 conftest.py

**问题代码：**
```python
# ❌ 有问题的写法
if _util.find_spec("torch") is None or ...
```

**根因：** `importlib.util.find_spec()` 对通过 `sys.modules` 注入的桩模块会返回 `__spec__ = None`，导致 `ValueError`。

**修复：**
```python
# ✅ 正确的写法
_torch_mod = sys.modules.get("torch", None)
if _torch_mod is None or getattr(_torch_mod, "_IS_STUB", False):
    collect_ignore.append("test_ppr_thresholds.py")
```

**教训：** 桩模块要用 `sys.modules` 判断，不要用 `find_spec`。

---

### 🔴 Round 2: ModuleNotFoundError: No module named 'src'

**报错：** `from src.app import app → ModuleNotFoundError: No module named 'src'`

**触发时机：** pytest 收集阶段，加载 test_app.py

**根因：** CI 环境下 Python 找不到项目根目录，`src` 不在 `sys.path` 中。

**修复：** 在 `tests/conftest.py` 顶部自动注册项目根路径：
```python
import sys, os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
```

**教训：** 写 conftest 的第一行就应该是项目根路径注册。这不只是 CI 问题——换目录跑 pytest 也会遇到。

---

### 🔴 Round 3: ModuleNotFoundError: No module named 'sentence_transformers'

**报错：** `from sentence_transformers import SentenceTransformer → ModuleNotFoundError`

**触发时机：** pytest 收集阶段，import `src.embedding` 时

**根因：** `src/embedding.py` 在**模块顶层**写了 `from sentence_transformers import SentenceTransformer`。pytest 只要收集到引用该模块的测试文件（即使标记了 `@slow`），就会触发这个 import。

```python
# ❌ 模块顶层 import — 收集阶段就崩
from sentence_transformers import SentenceTransformer

class LocalEmbeddingService:
    def __init__(self, ...):
        self.model = SentenceTransformer(model_name, device=device)
```

**修复：** 移入 `__init__` 方法内部（惰性加载）：
```python
# ✅ 函数内 import — 只有实际实例化时才触发
class LocalEmbeddingService:
    def __init__(self, ...):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
```

**教训 — 最重要的教训之一：**
- `@pytest.mark.slow` **只跳过执行阶段，不跳过收集阶段**
- 如果测试文件 import 了一个模块，而该模块的顶层有重型 import，收集阶段就崩了——`@slow` 救不了
- 解决方案永远是**源模块做惰性 import**，而非在测试文件堆 `@slow`

同时要修的是 `src/reranker.py`（同样的 `from sentence_transformers import CrossEncoder`）。

**波及范围：** 3 个测试文件（test_embedding / test_coordinator / test_reranker）被间接触发。

---

### 🔴 Round 4: ModuleNotFoundError: No module named 'rank_bm25'

**报错：** `from rank_bm25 import BM25Okapi → ModuleNotFoundError`

**触发时机：** pytest 收集阶段，import `src.database` 时

**根因：** `rank_bm25` 是轻量依赖（纯 Python 实现），在 `src/database.py` 模块顶层 import，但漏列入了 `requirements.lock`。

**修复：** 加入 `rank-bm25>=0.2` 到 `requirements.lock`。

**教训：**
- 不要边修边补——**一次性扫全量**所有 `src/` 文件的模块顶层 import，对照 `requirements.lock` 逐项核实
- `rank_bm25` 虽是轻量包，但对 pytest 收集阶段也是致命的

---

### 🔴 Round 5: 漏标 @pytest.mark.slow

**报错：**
- `test_database_graph.py::test_graph_edge_weights_and_idf` — 创建 `LocalEmbeddingService(device="cpu")`
- `test_graph_search.py::test_coordinator_graph_retrieval_and_cliff` — 同上

**触发时机：** pytest **执行**阶段（收集阶段通过了，但运行到该函数时崩了）

**根因：** 这两个测试函数内部调用了需要 `sentence_transformers` 的真实模型，但没标 `@pytest.mark.slow`。

**修复：** 补标 `@pytest.mark.slow`。

**教训：**
- 任何创建 `LocalEmbeddingService`、`RerankerService` 或类似真实模型实例的测试函数，**必须**标 `@slow`
- 这条纪律应该在写测试的第一个函数时就建立，而不是等 CI 炸了再补

---

## 三、最终 CI 分层架构

经过 5 轮修修补补，最终形成三层防线：

```
pytest tests/ -m 'not slow'  (CI 执行)
    │
    ├─ 第 1 层: conftest 桩系统
    │   └─ torch 不存在时自动注入假 torch，让 pytest 收集阶段不崩
    │
    ├─ 第 2 层: collect_ignore
    │   └─ 跳过 test_ppr_thresholds.py（非测试文件 + 模块级 import torch）
    │
    ├─ 第 3 层: @pytest.mark.slow 标记
    │   └─ 7 个测试函数被跳过（需要真实模型）
    │
    └─ requirements.lock（13 个轻量包）
        └─ 不含 torch / sentence-transformers，CI 30 秒装完
```

**CI 能跑的测试（轻量）：**

| 文件 | 测试函数数 | 依赖 |
|------|:---------:|------|
| test_splitter.py | 7 | MockEmbeddingService + mistune |
| test_database.py | 1 | ChromaAdapter |
| test_database_graph.py | 2 | ChromaAdapter + numpy |
| test_graph_search.py | 6 | MockRerankerService + networkx |
| test_loader.py | 6 | fitz + importorskip 守卫 |
| test_app.py | 4 | fastapi（惰性 import 设计优秀） |
| test_disguise_generator.py | 2 | requests + jieba |
| **合计** | **28** | |

**本地跑的测试（重型）：**

| 文件 | 测试函数数 | 需要 |
|------|:---------:|------|
| test_embedding.py | 3 | BGE-M3 模型 |
| test_coordinator.py | 1 | 全链路加载 |
| test_reranker.py | 2 | CrossEncoder 模型 |
| test_database_graph.py | 1 | LocalEmbeddingService |
| test_graph_search.py | 1 | LocalEmbeddingService |
| **合计** | **8** | |

---

## 四、回答尖子的灵魂拷问

> "这是我们测试脚本的耦合问题吗？"

**不是。** 3/5 的轮次是 src 源模块的问题（顶层重型 import、漏依赖、conftest 设计缺陷），只有 2/5 是测试脚本的疏忽（漏标 @slow）。

> "写 TDD 的时候能避免吗？"

**能避免一部分，但不是你想象的那种"先写测试"的 TDD。**

TDD 的真正价值在于**倒逼你设计出好测试的接口**。如果你写 `LocalEmbeddingService` 时先想"我怎么测试它？"，你会自然地把 `sentence_transformers` 放在 `__init__` 内部加载，而不是模块顶层。

但 TDD **防不住** `rank_bm25` 漏列 `requirements.lock` 这种 CI 基础设施问题。那是另一层面的纪律。

> "怎么根治？"

**把"CI 能不能跑"变成编码时的本能反应**，靠两条纪律：

```
纪律 1: src/ 模块顶层只放标准库 import
        任何第三方库 import，要么在 __init__ 内部，要么在函数内部

纪律 2: 写测试函数的第一件事就是判断"这个测试要不要真实模型？"
        要 → 立刻加 @pytest.mark.slow
        不要 → 用 Mock 或桩
```

---

## 五、经验清单（可复用检查表）

### 新项目 CI 搭建 checklist

```
□ 第 0 步：把 CI 当作一等公民，项目初始化时就规划好
□ conftest.py 第一行：项目根路径注册到 sys.path
□ src/ 模块顶层只 import 标准库
   ● 重型依赖（torch, transformers）放 __init__ 内部
   ● 轻量依赖（rank_bm25 等）放 requirements.lock
□ requirements.lock 只锁轻量依赖，重型留本地
□ 每个调真实模型/后端的测试函数，写的第一行是 @pytest.mark.slow
□ 在 pyproject.toml 注册 [tool.pytest.ini_options] markers
□ pytest -m 'not slow' 跑通后，再逐个跑 @slow 验证
```

### 这次踩坑的 "三不要"

| 不要 | 要 |
|------|-----|
| 不要 `find_spec()` 判断桩 | 要 `sys.modules.get()` + `_IS_STUB` |
| 不要指望 `@slow` 救收集阶段崩 | 要从源模块做惰性 import |
| 不要炸一个修一个（打地鼠） | 要一次性扫全量 src import + 对照 requirements.lock |

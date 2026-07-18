# CI 故障排查经验教训

> 项目：advanced-rag
> 日期：2026-07-18
> 触发：`test_run_personalized_pagerank` 在 GitHub Actions CI 上失败，本地无法复现

---

## 1. 根因：NetworkX 3.6.1 的 `nx.pagerank` 强制依赖 scipy

### 现象
CI 上 `test_run_personalized_pagerank` 返回空列表，断言 `len(res) > 0` 失败。

### 排查过程
| 步骤 | 操作 | 结果 |
|------|------|------|
| 1 | 本地运行测试（nx 3.6.1 + numpy 2.5.1） | ✅ 通过 |
| 2 | 推测 weight='weight' 在无权重边上异常 | 在 nx 3.0/3.2/3.6 均正常工作，排除 |
| 3 | 在纯净 venv 安装 exact 相同包列表 | ❌ **复现成功** |
| 4 | 逐一排查 | `ModuleNotFoundError: No module named 'scipy'` |

### 根因
NetworkX 从 3.6 开始，`nx.pagerank` 内部改用 scipy 稀疏矩阵实现。即使是 3 节点小图，**没有 scipy 也会直接抛异常**。

### 旧代码的坑
```python
# 旧代码 —— 静默吞掉了所有异常
try:
    scores = nx.pagerank(sub_graph, ...)
except Exception:
    return []           # ← 你看不到任何错误
```

### 修复
```python
# 在 requirements.lock 中补充：
scipy>=1.10.0   # networkx≥3.6 的 pagerank 强制依赖
```

---

## 2. 教训：bare `except Exception` 是魔鬼

### 错误写法
```python
try:
    scores = nx.pagerank(...)
except Exception:
    return []
```
- 吞掉了 `ModuleNotFoundError`（缺少 scipy）
- 吞掉了 `ValueError`（参数错误）
- 吞掉了 `TypeError`（API 变化）
- 完全没有日志，查问题全靠猜

### 正确做法
```python
try:
    scores = nx.pagerank(...)
except ImportError:
    # 具体处理缺少依赖的情况
    raise
except (ValueError, TypeError) as e:
    print(f"[WARN] pagerank 计算失败: {e}")
    return []
```

**原则**：`except Exception` 只应在以下场景使用：
- 你有明确的 fallback 逻辑
- 异常是不影响正确性的边缘情况
- 配合日志记录（至少 print 出来）

---

## 3. 教训：CI 环境与本地环境的依赖差异

### 看似相同，实则不同
- 本地环境安装了大量重型包（torch, ultralytics, scipy 等），无意中提供了 CI 缺失的依赖
- CI 只安装 `requirements.lock`，是一个 **最小依赖集**
- 此项目的 `requirements.lock` 之前没有 scipy，但 `nx.pagerank` 需要它

### 排查工具：纯净 venv 复现法
```bash
# 创建空 venv → 只装 CI 会装的包 → 跑测试
python -m venv /tmp/ci-repro
source /tmp/ci-repro/bin/activate
pip install -r requirements.lock pytest
pytest tests/ -x -m 'not slow'
```
> 这是排查 CI-only 失败最有效的手段——**消除本地额外依赖的干扰**。

---

## 4. 教训：CI 按字母序跑测试，-x 会隐藏后续失败

### 现象
CI 使用 `pytest -x`（失败即停止）。`test_database_graph_linking` 按字母序排在 `test_run_personalized_pagerank` 前面。

### 时间线

```
Commit c47238c: 加入 test_run_personalized_pagerank 测试
Commit b34d8f9: 加入 CI 配置（-x -m 'not slow'）
  → test_database_graph_linking 先失败，-x 停下
  → test_run_personalized_pagerank 从未被 CI 执行过！
```

直到我们把 `database.py` / `graph_search.py` 的 pending 改动提交，`test_database_graph_linking` 通过了，`test_run_personalized_pagerank` 才暴露出来。

### 教训
- `pytest -x` 在调试时方便，但会掩盖后面的问题
- 本地跑测试时建议先不用 `-x`，或者分批跑
- **CI 绿了 ≠ 所有测试都跑过**

---

## 5. 教训：改 splitter 默认参数会级联影响测试

### 改动
```python
# 旧：兜底合并只在 is_markdown 时启用，min_parent_size=150
if is_markdown:
    min_parent_size = 150
    ...

# 新：全局启用，min_parent_size=300（默认参数）
def __init__(self, ..., min_parent_size=300):
    ...
```

### 影响
测试 `test_dynamic_threshold` 用短文本（4个"第一部分。"），两段都小于 300 字符，被合并为 1 个 parent chunk。`assert len(parents) == 2` 失败。

### 教训
- 修改默认参数时，搜索所有调用处和测试用例
- 暴露参数的改法本身没错，但要同步更新测试
- 或者：为新旧行为写独立的测试用例

---

## 6. 总结：快速诊断 checklist

| 步骤 | 做什么 |
|------|--------|
| 1 | 确认本地能否复现（同样的 Python 版本、同样的包） |
| 2 | 如果无法复现 → 创建纯净 venv，只装 CI 会装的包 |
| 3 | 如果还不行 → 检查 CI 日志的 `pip install` 输出 |
| 4 | 找到差异依赖后，加锁文件 + 本地验证 |
| 5 | 永远不要用 `except Exception: pass` 掩盖异常 |
| 6 | 修改默认参数时检查所有测试用例 |

---

*本文由 WorkBuddy 在 advanced-rag CI 故障排查过程中自动生成。*

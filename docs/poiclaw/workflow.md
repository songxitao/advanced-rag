# 2. 状态机与 Workflow 剖析

在上一步中，我们把大模型（大脑）配置好了。现在我们要来解决 Agent 的“躯干”——**工作流（Workflow）与状态机**。

很多重型框架（如 LangChain）为了实现节点流转，写了成百上千行复杂的黑盒代码。而在 `PoiClaw` 中，我们使用 **同步状态机**，仅用不到 60 行代码就让节点之间可以“像画图一样优雅地连线”。

在看连线之前，我们先科普几个对 Python 不熟悉的人最容易懵的语法常识。

---

## 💡 新手科普：Python 面向对象三剑客

```python
class Node:
    def __init__(self) -> None:
        self.successors = {}
```
1. **`class Node`（类）**：它就像一张“设计图纸”。设计图纸本身不能直接拿来当房子住。
2. **`self`（自己）**：在图纸里，`self` 指代的就是“未来用这个图纸盖出来的那个具体的房子（实例）”。`self.successors` 就是在这个房子的墙上挖个壁橱。
3. **`__init__`（初始化函数）**：当你去盖房子时（`node = Node()`），这个函数会自动执行，帮你把壁橱（`successors`）建好。

---

## 魔法重载：连线运算符 `>>` 和 `-`

我们希望在定义工作流时，代码写起来非常像画图：
`节点 A 执行完 -> 如果返回 action 是 search 则进入节点 B`。

Python 允许我们对系统自带的符号（如 `>>` 和 `-`）进行**自定义功能改写（运算符重载）**。我们在 `core/node.py` 里重载了它们：

```python
# 1. 重载了减号 "-" 运算符
def __sub__(self, action: str) -> "Node":
    # 只要在 Node 后面写了一个 - "动作名字"，就会把这个动作名字暂时存进 _action 变量里
    self._action = action or "default"
    return self

# 2. 重载了右移 ">>" 运算符
def __rshift__(self, other: "Node") -> "Node":
    # 当你在 Node 后面写了 >> 目标节点，就会在当前的“壁橱”里，把动作和目标节点绑定存起来
    self.successors[self._action] = other
    self._action = "default" # 复位动作名字
    return other
```

### 🧠 傻瓜式连线翻译：
当你写：
```python
node_a - "search" >> node_b
```
* 第一步：`node_a - "search"` 执行。`node_a` 的 `_action` 变量被改成了 `"search"`。
* 第二步：`>> node_b` 执行。`node_a` 的 `successors` 字典里，塞进了一条规则：`{"search": node_b}`。
* 这行代码的大白话意思就是：**如果 `node_a` 执行完返回了 `"search"`，下一步就去跑 `node_b`**！

---

## 🔄 搜索与总结工作流的数据流向剖析

我们来剖析 `examples/workflow/main.py` 里的实战数据流。

### 1. 节点定义与连线：
```python
query = QueryNode()
search = SearchNode()
summarize = SummarizeNode()

query - "search" >> search         # 规则1: query 返回 search 去执行 search 节点
search - "summarize" >> summarize  # 规则2: search 返回 summarize 去执行 summarize 节点

flow = Flow(query) # 指定 query 节点为起点
flow.run("asyncio python best practices") # 启动！
```

### 2. 核心数据流转过程：
* **第一步：`query` 节点执行**
  * 接收输入 `"asyncio..."`，返回动作 `("search", "asyncio...")`。
  * 编排器 Flow 看到返回的动作是 `"search"`，去规则库里一查，发现对应的是 `search` 节点。于是把 `"asyncio..."` 传给 `search` 节点。
* **第二步：`search` 节点执行**
  * 接收到 `"asyncio..."`，调用 DuckDuckGo 搜索，抓取了 3 条网页标题（比如 "Best practice 1 | Best practice 2"）。
  * 返回动作 `("summarize", "Best practice 1 | Best practice 2")`。
  * 编排器 Flow 查规则，发现动作 `"summarize"` 对应 `summarize` 节点。
* **第三步：`summarize` 节点执行**
  * 接收到长网页标题，拼成 Prompt 喂给本地大模型：`"基于以下要点写一句话摘要：..."`。
  * 大模型总结出答案，返回 `("default", "这是总结的答案")`。
  * 编排器发现规则库里没有 `"default"` 对应的下一个节点了，**运行结束，退出并输出答案**！

---

## 🗣️ 面试官怎么问，你该怎么答？

> **面试官问**：*“业界常用的 Agent 编排框架很多（如 LangGraph, CrewAI），你为什么选择自己写底层的 Node 与 Flow 状态机？”*
>
> **你大方地答**：
> *“大厂的重型框架存在过度抽象、难以定制等问题。在很多实际工程落地场景中，我们需要极低的时延和 100% 可控的数据流。*
> *我重构的 `PoiClaw` 状态机仅用不到 60 行代码就实现了状态流转。利用 Python 的 `__rshift__` 和 `__sub__` 运算符重载，让我们可以用极其直观的代码定义拓扑连线。由于是手写，我们能对每一次状态跳转、数据传递和异常捕获的底层数据流讲得一清二楚，相比之下更有技术深度。”*

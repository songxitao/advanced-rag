# 5. 长时间任务（Goal Loop）大闭环

在许多人的观念里，能聊天、能跑个搜索就叫 Agent。但在真实的软件工程中，这种“一问一答、跑一轮就歇”的机器人，最多算是一个高级的“Chatbot”。

真正的智能体应当具备 **长时间自主死磕（Long-running Goal）** 的能力。

比如你给它下达目标：`/goal 请把 tests 文件夹下的所有测试文件通过`。它应该自己在后台修改代码、运行测试、查看报错、再次修改，直到测试 100% 通过为止，中间不需要你不停去“催促”它。

---

## 🔄 核心机制：Goal Loop 闭环是怎么跑的？

我们先用一张**时序图**来看清大模型、记忆、沙箱和循环引擎在每一次 `while` 迭代中是怎样在后台传递数据的：

```mermaid
sequenceDiagram
    autonumber
    participant LLM as PoiClaw 大脑 (Gemma/Qwen)
    participant Loop as GoalLoop 循环引擎
    participant Sandbox as 安全执行沙箱 (Python)
    participant Memory as 对话上下文记忆 (Messages)

    Loop->>Memory: 1. 获取当前所有的历史消息 Messages
    Memory->>LLM: 2. 发起 API 请求 (带工具列表与 System Prompt)
    LLM->>Loop: 3. 返回思考内容与 JSON 意图
    alt 模型吐出 tool_complete
        Loop->>Loop: 4. 关闭 goal_active，跳出循环，任务圆满结案！
    else 模型吐出 bash / write_file
        Loop->>Sandbox: 5. 调度安全沙箱执行操作
        Sandbox->>Loop: 6. 返回执行结果 stdout 或报错堆栈
        Loop->>Memory: 7. 将模型思考和沙箱输出 append 追加写入记忆
        Note over Loop, Memory: 状态更新，进入下一个 while 周期
    end
```

我们将大脑（大模型）、感官（Memory）和手脚（沙箱工具）通过一个 `while` 循环串联在一起，形成了 `PoiClaw` 智能体的终极控制中枢。

请双击打开 [poiclaw/core/goal.py](file:///e:/project/Learn-OpenClaw/poiclaw/core/goal.py) 文件：

```python
from typing import Any, Callable

class GoalLoop:
    def __init__(self, goal: str, max_steps: int = 10):
        self.goal = goal
        self.max_steps = max_steps
        self.goal_active = True # 目标活跃开关：True代表还要继续跑，False代表可以停了
        
        # 初始的记忆队列：我们把用户下达的 Goal 目标塞进 messages 数组里
        self.messages: list[dict[str, Any]] = [
            {"role": "user", "content": f"目标：{goal}"}
        ]

    def run(self, llm_caller: Callable) -> str:
        step = 0
        last_msg = ""
        
        # 只要开关是 True（目标未达成），并且没有超出我们限定的最大尝试步数
        while self.goal_active and step < self.max_steps:
            step += 1
            
            # 1. 呼叫大模型：把之前累积的完整记忆 messages（包括历史工具执行结果）全喂给它
            response = llm_caller(self.messages, tools=[])
            content = response.get("content", "")
            
            # 2. 自愈 JSON 解析：调用我们之前写的 repair_and_parse_json 函数
            try:
                from poiclaw.core.llm import repair_and_parse_json
                parsed = repair_and_parse_json(content)
                tool_name = parsed.get("tool")
                
                # 3. 目标达成判断：如果大模型输出的 tool 名是 "goal_complete"
                if tool_name == "goal_complete":
                    self.goal_active = False # 关闭开关，退出 while 循环
                    last_msg = parsed.get("args", {}).get("msg", "Goal Complete")
                    break
                
                # 4. 执行手脚命令：如果大模型想跑 bash 命令
                elif tool_name == "bash":
                    cmd = parsed.get("args", {}).get("cmd", "")
                    
                    # 调用我们第四章写好的“安全沙箱”执行工具
                    from poiclaw.tools.sandbox import execute_cmd_safe
                    res = execute_cmd_safe(cmd)
                    
                    # 5. 【数据流闭环关键点】：
                    # 我们把大模型的思考 (content) 和安全沙箱返回的执行结果 (res['stdout'])
                    # 依次追加（append）到 self.messages 数组里！
                    # 这样在下一次循环调用大模型时，它就能“看到”上一步命令运行的结果是成功还是失败！
                    self.messages.append({"role": "assistant", "content": content})
                    self.messages.append({"role": "user", "content": f"工具执行输出: {res['stdout']}"})
            
            except Exception as e:
                # 6. 【自动容错自愈机制】：
                # 如果模型输出的 JSON 连我们的自愈解析器都无法修复，或者命令执行报错了
                # 我们千万不能让程序崩溃退出！
                # 我们要把报错信息包装成一条 user 反馈，强行喂回给大模型：
                # “大模型你刚才给的格式不对/执行报错了：{e}。请根据这个错误重新生成！”
                self.messages.append({"role": "assistant", "content": content})
                self.messages.append({"role": "user", "content": f"解析/执行失败: {str(e)}，请重新生成并输出规范的 JSON。"})
        
        return last_msg
```

---

## 🚀 企业级工程落地：PM2 与飞书联动

1. **什么是 PM2？（后台守护进程）**：
   在真实的业务中，我们把 Agent 部署在 Linux 服务器上。如果关掉终端窗口，Agent 进程就会死掉。
   我们使用 **PM2** 工具来维持它。它就像是一个保镖：只要检测到 `PoiClaw` 因为系统原因崩溃或退出了，PM2 会在毫秒级自动帮它重启，维持 Agent 的 24 小时在线。
2. **如何与飞书/钉钉联动协作？**：
   我们使用 Flask/FastAPI 库在 Python 里开一个 Webhook 接口，暴露一个 URL 网址跟飞书机器人对接。
   你在手机飞书群里艾特机器人发送一句：`/goal 帮我把 main.py 里的测试补全`。
   这个指令就会通过网络传给 PM2 里的 `PoiClaw`，激活 `GoalLoop` 在沙箱里默默帮你改代码，跑测试，改好后在飞书群里通知你，形成完美的 Agentic Workflow！

---

## 🗣 ...面试官怎么问，你该怎么答？

> **面试官问**：*“业界有很多人在用 LangChain 或者是 AutoGPT 这种多轮自主执行引擎，你的 Goal Loop 相比之下有什么工程优势？”*
>
> **你大方地答**：
> *“LangChain 的长任务引擎高度黑盒化，报错调试极其困难。而且本地小模型无法支撑起复杂的链条调用。*
> *在 `PoiClaw` 中，我自主设计实现了**轻量级 Goal Loop 状态机**。我将大模型、自愈解析器、安全命令行沙箱和 Memory 上下文闭环在一个 `while` 控制循环中。*
> *它的两大工程优势是：第一，**异常自愈机制**——一旦 JSON 解析或沙箱命令执行失败，错误栈会自动作为 Prompt 反馈给 LLM 触发自我纠偏，形成闭环；第二，**流式 KV Cache 友好**——我们只进行线性追加写入 `session.jsonl`，这极大提高了本地 llama.cpp 推理时的缓存命中率，速度成倍提升，同时配合 Token 溢出压缩保障了长时间运行的可靠性。”*

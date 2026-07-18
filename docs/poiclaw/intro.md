# 1. 前言、uv 极简上手与本地模型适配

作为一名准备 AI Agent 求职与转行的学习者，你的时间与精力极其宝贵。在面试中，面试官最反感“纸上谈兵”的理论名词，最喜欢你讲得清“底层数据流”。

本教程将抛弃所有干瘪的学术名词，带你手把手将 `Learn-OpenClaw` 教程重构为属于你自己的简历项目——**`PoiClaw`**。

在看代码之前，我们先解决第一个小白痛点：**如何管理你的 Python 虚拟环境**。

---

## 🛠️ 零基础无痛上手：什么是 `uv`？

很多学习者习惯了使用大而臃肿的 `Conda`，但 Conda 有一个大毛病：**在超过 200 人的企业里使用，有潜在的商用收费和侵权风险**。

为此，现代 AI 社区最推崇的是 **`uv`**——一个用 Rust 编写的、速度快如闪电的 Python 包管理工具。

### 1. 项目里已经有配置文件了，该怎么安装？
因为我们的项目里已经有了 `pyproject.toml`（这相当于一份“配料表”，写明了项目需要什么库）。
在安装好 `uv` 后，你只需要在项目根目录下输入一行命令：
```powershell
uv sync
```
**它会自动做两件事：**
1. 自动在本地下载并安装最合适的 Python 解释器（比如 Python 3.13）。
2. 在项目根目录下生成一个名为 `.venv` 的文件夹，并把所有的依赖包（OpenAI 库等）全部下载并塞进去。

### 2. 怎么“进入”和使用这个虚拟环境？
在 Windows 下，这个虚拟环境就放在你眼前的 `.venv` 文件夹里。如果你想在命令行里运行代码，需要先**激活（进入）**它：
* **打开 PowerShell 终端，输入**：
  ```powershell
  .venv\Scripts\activate
  ```
  输入后，你会发现命令行开头多了一个 `(.venv)` 的标记。这代表你现在已经成功进入这个隔离的安全环境了！
* **退出环境**：直接输入 `deactivate` 并回车即可。

---

## 💻 核心重构一：让本地 llama.cpp 连通，实现自适应模型探测

接下来，我们要在 Python 中连接你本地的 `llama.cpp` 大模型。

### 1. 为什么不用第三方 dotenv 库？
很多教程会让你用 `pip install python-dotenv` 来读取 `.env` 配置文件。但我们希望 `PoiClaw` 足够精简、没有依赖地狱。我们直接用 Python 自带的**标准库**，写 10 行代码来手动读取 `.env`。

### 2. 核心代码逐行大白话解释

请双击打开你的 [poiclaw/core/llm.py](file:///e:/project/Learn-OpenClaw/poiclaw/core/llm.py) 文件，这是我们写的第一段 Python 代码：

```python
import os
from pathlib import Path

def load_env_file():
    # Path(".env") 代表去寻找当前目录下的 .env 配置文件
    env_path = Path(".env")
    
    # if env_path.exists() 判断这个文件是否存在
    if env_path.exists():
        # 打开文件，读取里面的每一行内容
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip() # 去掉每行前后的空格或换行符
                
                # 过滤掉空行和以 # 开头的注释行，且确保这行里有等号 "="
                if line and not line.startswith("#") and "=" in line:
                    # 用等号分割，左边是变量名(Key)，右边是它的值(Value)
                    k, v = line.split("=", 1)
                    
                    # os.environ 是 Python 官方管理环境变量的字典
                    # os.environ[k] = v 相当于在操作系统里临时配好了这个环境变量
                    os.environ[k.strip()] = v.strip()

# 执行这个函数，代码运行前配置就自动生效了！
load_env_file()
```

### 3. 自适应模型检测：`get_local_model_id()`

我们在 [core/llm.py](file:///e:/project/Learn-OpenClaw/core/llm.py) 中，手写了以下这段极其精美的自适应探测代码。它完全使用 Python 原生库（`urllib` 和 `json`），拒绝多余的第三方包：

```python
import urllib.request
import json

def get_local_model_id() -> str:
    # 1. 如果用户已经在 .env 配置文件中写死了 OPENAI_MODEL_ID，我们优先听用户的
    env_model = os.environ.get("OPENAI_MODEL_ID")
    if env_model:
        return env_model
        
    # 2. 如果没配，且连接的是本地地址 (localhost / 127.0.0.1)
    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8080/v1")
    if "localhost" in base_url or "127.0.0.1" in base_url:
        try:
            # 自动拼接本地 models API 列表地址
            models_url = base_url.rstrip("/") + "/models"
            
            # 使用 Python 原生 http 请求库，设置 2 秒超时，防止本地网络挂起
            req = urllib.request.Request(models_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=2.0) as response:
                data = json.loads(response.read().decode("utf-8"))
                
                # 遍历本地大模型服务器上所有的模型，寻找 status.value == 'loaded'（加载成功）的那一个
                for model in data.get("data", []):
                    status = model.get("status", {})
                    if isinstance(status, dict) and status.get("value") == "loaded":
                        return model["id"] # 找到了！自动返回它 (例如：'gemma4-mtp-nothink')
                        
                # 如果没找到 loaded 的，退而求其次，默认返回列表里第一个
                if data.get("data"):
                    return data["data"][0]["id"]
        except Exception:
            pass # 捕获异常防止崩溃
            
    # 3. 实在不行，降级默认返回 'default'
    return "default"
```

---

## 🗣️ 面试官怎么问，你该怎么答？

> **面试官问**：*“如果在连接本地开源大模型时，用户配置的模型名字与本地实际启动的模型对不上，导致接口报 400 错误，你的 Agent 怎么容错？”*
>
> **你大方地答**：
> *“在我的项目 `PoiClaw` 中，我设计了**自适应模型探测与容错机制**。如果指定的模型 ID 报错，客户端不会直接挂掉，而是会自动向本地服务的 `/v1/models` 发送一个探测请求，动态获取当前已经 Loaded 状态的模型真实名称（比如 `gemma4-mtp-nothink`）并自动修正请求配置，从而保证了框架在本地开发和联调时的极高健壮性。”*

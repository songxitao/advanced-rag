import os
import sys
import types
import importlib.util as _util

# 把项目根目录加到 sys.path，让 from src.xxx import xxx 在 CI 中也能找到模块
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

class _StubModule(types.ModuleType):
    _IS_STUB = True

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        stub = _StubModule(f"{self.__name__}.{name}")
        setattr(self, name, stub)
        return stub

    def __call__(self, *args, **kwargs):
        from unittest.mock import MagicMock
        return MagicMock()

# 注册 torch 桩（防止模块级 import torch 崩掉）
_STUB_MODULES = [
    "torch",
]

for mod_name in _STUB_MODULES:
    try:
        __import__(mod_name)
    except ImportError:
        stub = _StubModule(mod_name)
        if mod_name == "torch":
            stub.Tensor = type("Tensor", (), {})
            stub.cuda = _StubModule("torch.cuda")
            stub.cuda.is_available = lambda: False
            stub.set_num_threads = lambda x: None
        sys.modules[mod_name] = stub

# collect_ignore — 跳过不是测试文件的脚本
collect_ignore = []

# test_ppr_thresholds.py 没有 def test_* 函数，且模块级 import torch
_torch_mod = sys.modules.get("torch", None)
if _torch_mod is None or getattr(_torch_mod, "_IS_STUB", False):
    collect_ignore.append("test_ppr_thresholds.py")

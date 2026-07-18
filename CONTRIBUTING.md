# Contributing to advanced-rag

## 开发环境

```bash
conda create -n advanced-rag python=3.12 -y
conda activate advanced-rag
pip install -r requirements.txt
```

## 代码风格

本项目使用 ruff 进行代码格式化和 linting：

```bash
pip install ruff
ruff check src/ tests/
ruff format src/ tests/
```

## 运行测试

```bash
pytest tests/ -v
```

## Pull Request 流程

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交变更 (`git commit -m 'feat: add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 报告 Issue

请使用 GitHub Issues 提交 bug 报告或功能请求。

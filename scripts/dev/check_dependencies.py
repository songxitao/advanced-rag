import sys

modules = ['ragas', 'pandas', 'matplotlib', 'pydantic', 'openai', 'requests', 'jieba', 'chromadb', 'langchain', 'docx']
for module in modules:
    try:
        __import__(module)
        print(f"{module}: INSTALLED")
    except ImportError:
        print(f"{module}: NOT INSTALLED")

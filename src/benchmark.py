import time
import os
import sys

# 确保能找到 src 目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedding import LocalEmbeddingService
from src.splitter import SemanticParentChildSplitter

class LegacyEmbeddingWrapper:
    """
    一个包装类，故意隐藏 get_dense_embeddings_batch 接口，
    用来模拟重构前的『串行单条调用』模式。
    """
    def __init__(self, real_service):
        self.real_service = real_service

    def get_dense_embedding(self, text: str):
        return self.real_service.get_dense_embedding(text)

def run_benchmark():
    print("🔄 正在初始化 BGE-M3 模型（运行在 CPU 上以便于耗时统计）...")
    real_service = LocalEmbeddingService(device="cpu")
    
    # 准备测试文本：模拟一篇包含大约 100 句话的 Obsidian 日志
    sentences = [f"这是我的项目日志第 {i} 句话，用来测试语义分片。东南大学与早稻田大学展开学术合作。" for i in range(100)]
    test_text = "。".join(sentences)
    
    print(f"📄 测试文本已准备好：共 {len(sentences)} 句话，总字符数 {len(test_text)}")
    print("--------------------------------------------------")

    # 1. 测试重构前（串行单条模式）
    legacy_service = LegacyEmbeddingWrapper(real_service)
    splitter_legacy = SemanticParentChildSplitter(embedding_service=legacy_service, threshold=0.5, child_size=30)
    
    print("⚡ 正在测试：[重构前] 串行单条模式...")
    start_time = time.time()
    chunks_legacy = splitter_legacy.create_parent_child_chunks(test_text)
    legacy_duration = time.time() - start_time
    print(f"   ➔ 生成子块数: {len(chunks_legacy)}")
    print(f"   ➔ 耗时: {legacy_duration:.4f} 秒")
    print("--------------------------------------------------")

    # 2. 测试重构后（去重批处理并行模式）
    splitter_parallel = SemanticParentChildSplitter(embedding_service=real_service, threshold=0.5, child_size=30)
    
    print("🚀 正在测试：[重构后] 批处理+去重并行模式...")
    start_time = time.time()
    chunks_parallel = splitter_parallel.create_parent_child_chunks(test_text)
    parallel_duration = time.time() - start_time
    print(f"   ➔ 生成子块数: {len(chunks_parallel)}")
    print(f"   ➔ 耗时: {parallel_duration:.4f} 秒")
    print("--------------------------------------------------")

    # 3. 计算结果
    speedup = (legacy_duration - parallel_duration) / legacy_duration * 100
    times_faster = legacy_duration / parallel_duration
    print("📊 跑分统计结论：")
    print(f"   - 串行耗时: {legacy_duration:.2f} 秒")
    print(f"   - 并行耗时: {parallel_duration:.2f} 秒")
    print(f"   - 🚀 写入提速比例: {speedup:.2f}%")
    print(f"   - 🚀 性能提升倍数: {times_faster:.2f} 倍")
    print("==================================================")
    print("💡 您可以直接把这个『性能提升倍数』数据写进您的简历中！")

if __name__ == "__main__":
    run_benchmark()

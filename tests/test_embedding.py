import pytest
from src.embedding import LocalEmbeddingService

def test_embedding_service():
    # BGE-M3 默认跑在 cuda 上。如果显卡被占用，我们确保设备在构造函数里支持自定义（测试时可传 cpu，生产用 cuda）
    service = LocalEmbeddingService(device="cpu") # 测试用 cpu 确保任何电脑均可顺利通过，无需占用显存
    dense = service.get_dense_embedding("测试早稻田和东大")
    sparse = service.get_sparse_embedding("测试早稻田和东大")
    
    assert len(dense) == 1024  # BGE-M3 的标准稠密向量是 1024 维
    assert isinstance(sparse, dict)
    assert len(sparse) > 0

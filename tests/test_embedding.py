import pytest
from src.embedding import LocalEmbeddingService

@pytest.mark.slow
def test_embedding_service():
    # BGE-M3 默认跑在 cuda 上。如果显卡被占用，我们确保设备在构造函数里支持自定义（测试时可传 cpu，生产用 cuda）
    service = LocalEmbeddingService(device="cpu") # 测试用 cpu 确保任何电脑均可顺利通过，无需占用显存
    dense = service.get_dense_embedding("测试早稻田和东大")
    sparse = service.get_sparse_embedding("测试早稻田和东大")
    
    assert len(dense) == 1024  # BGE-M3 的标准稠密向量是 1024 维
    assert isinstance(sparse, dict)
    assert len(sparse) > 0

@pytest.mark.slow
def test_embedding_service_batch():
    service = LocalEmbeddingService(device="cpu")
    texts = ["测试早稻田", "测试东大"]
    dense_list = service.get_dense_embeddings_batch(texts)
    assert len(dense_list) == 2
    assert len(dense_list[0]) == 1024
    assert len(dense_list[1]) == 1024

@pytest.mark.slow
def test_embedding_device():
    import torch
    service = LocalEmbeddingService()
    if torch.cuda.is_available():
        assert service.model.device.type == "cuda"
    else:
        assert service.model.device.type == "cpu"

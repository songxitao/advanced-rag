import pytest
from fastapi.testclient import TestClient
from src.app import app

class MockCoordinator:
    def __init__(self):
        self.files_added = []

    def add_file(self, file_path: str):
        self.files_added.append(file_path)

    def query(self, query_str: str) -> str:
        if "早稻田" in query_str:
            return "[片段1] (来源: mock_doc.txt)\n早稻田大学计算机系招收研究生。"
        return ""

@pytest.fixture
def client():
    # 模拟 coordinator 并注入 app.state
    mock_coord = MockCoordinator()
    app.state.coordinator = mock_coord
    with TestClient(app) as c:
        yield c

def test_api_retrieve(client):
    response = client.post("/retrieve", json={"query": "关于早稻田大学的研究生课程", "top_k": 2})
    assert response.status_code == 200
    data = response.json()
    assert "context" in data
    assert "早稻田大学" in data["context"]

def test_api_add_file(client, tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("测试文本内容", encoding="utf-8")
    
    response = client.post("/add_file", json={"file_path": str(test_file)})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    # 验证是否成功调用了 MockCoordinator
    assert app.state.coordinator.files_added == [str(test_file)]

def test_api_validation_error(client):
    # 测试缺少字段的校验情况
    response = client.post("/retrieve", json={})
    assert response.status_code == 422

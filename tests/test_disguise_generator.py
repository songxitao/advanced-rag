import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'dev'))

import json
import pytest
import requests
from unittest.mock import patch, MagicMock
import jieba
from disguise_book_generator import run_disguise_pipeline

@pytest.fixture(autouse=True)
def setup_jieba():
    # 动态添加词典词，确保分词打标为 nr
    jieba.add_word("刘备", tag="nr")
    jieba.add_word("玄德", tag="nr")
    jieba.add_word("曹操", tag="nr")
    jieba.add_word("孟德", tag="nr")
    jieba.add_word("关羽", tag="nr")
    jieba.add_word("云长", tag="nr")

def test_run_disguise_pipeline_success(tmp_path):
    # 1. 准备 mock 小说文本
    # 包含“刘备” 20 次，“玄德” 20 次，“曹操” 20 次，“孟德” 20 次
    # 另外在文本中穿插“备”和“操”字，测试单字防误杀过滤
    mock_text = ("刘备玄德曹操孟德。操心操作亮起。" * 20) + " 玄德有备无患，备备备。"
    input_file = tmp_path / "mock_novel.txt"
    input_file.write_text(mock_text, encoding="utf-8")
    
    output_dir = tmp_path / "output"
    
    # 2. Mock 成功的接口返回，聚类字典中故意添加单字别名 "备" 和 "操"
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "刘备": ["刘备", "玄德", "备"],
                        "曹操": ["曹操", "孟德", "操"]
                    }, ensure_ascii=False)
                }
            }
        ]
    }
    
    with patch("requests.post", return_value=mock_response) as mock_post:
        ret_alias, ret_disguised = run_disguise_pipeline(str(input_file), str(output_dir))
        
        # 验证 API 被调用
        assert mock_post.called
        
    # 3. 验证文件被正确创建
    assert os.path.exists(ret_alias)
    assert os.path.exists(ret_disguised)
    
    # 4. 验证 JSON 聚类文件内容
    with open(ret_alias, 'r', encoding='utf-8') as f:
        aliases = json.load(f)
    assert "刘备" in aliases
    assert "曹操" in aliases
    assert aliases["刘备"] == ["刘备", "玄德", "备"]
    
    # 5. 验证脱敏后的文本
    with open(ret_disguised, 'r', encoding='utf-8') as f:
        disguised_text = f.read()
        
    # 验证“刘备”、“玄德”、“曹操”、“孟德”被完全替换为相应的代号
    assert "刘备" not in disguised_text
    assert "玄德" not in disguised_text
    assert "曹操" not in disguised_text
    assert "孟德" not in disguised_text
    assert "[角色_" in disguised_text
    
    # 验证单字防误杀：文本中的单个“备”和“操心操作”中的“操”没有被替换为 "[角色_"
    assert "操心" in disguised_text
    assert "操作" in disguised_text
    assert "有备无患" in disguised_text
    assert "备备备" in disguised_text

def test_run_disguise_pipeline_fallback(tmp_path):
    # 1. 准备 mock 小说文本
    mock_text = "刘备和关羽在琢郡结义，玄德与云长感情深厚。" * 20
    input_file = tmp_path / "mock_novel_fallback.txt"
    input_file.write_text(mock_text, encoding="utf-8")
    
    output_dir = tmp_path / "output_fallback"
    
    # 2. Mock 接口抛出异常
    with patch("requests.post", side_effect=requests.exceptions.RequestException("Network error")) as mock_post:
        ret_alias, ret_disguised = run_disguise_pipeline(str(input_file), str(output_dir))
        assert mock_post.called
        
    # 3. 验证是否回退到了 STATIC_ALIASES 字典
    with open(ret_alias, 'r', encoding='utf-8') as f:
        aliases = json.load(f)
    # STATIC_ALIASES 中应该有 12 个核心角色
    assert len(aliases) == 12
    assert "刘备" in aliases
    assert "关羽" in aliases
    
    # 4. 验证脱敏后的文本
    with open(ret_disguised, 'r', encoding='utf-8') as f:
        disguised_text = f.read()
        
    assert "刘备" not in disguised_text
    assert "玄德" not in disguised_text
    assert "关羽" not in disguised_text
    assert "云长" not in disguised_text
    assert "[角色_" in disguised_text

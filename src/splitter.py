import re
import math
import hashlib

class SemanticParentChildSplitter:
    def __init__(self, embedding_service, threshold=None, child_size=150, window_size=3):
        self.embedding_service = embedding_service
        self.threshold = threshold
        self.child_size = max(1, child_size)
        self.window_size = max(1, window_size)

    def _cosine_similarity(self, v1: list[float], v2: list[float]) -> float:
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm_v1 = math.sqrt(sum(a * a for a in v1))
        norm_v2 = math.sqrt(sum(b * b for b in v2))
        if norm_v1 == 0.0 or norm_v2 == 0.0:
            return 0.0
        return dot_product / (norm_v1 * norm_v2)

    def _split_parent_to_chunks(self, parent_text: str, child_size: int) -> list[str]:
        chunks = []
        remaining = parent_text
        # 英文与中文的常见标点符号和空格换行
        punctuation = "。？！，、；\n!?.,; "
        
        while len(remaining) > child_size:
            sub_text = remaining[:child_size]
            split_idx = -1
            # 从右往左找最后一个标点符号
            for i in range(len(sub_text) - 1, -1, -1):
                if sub_text[i] in punctuation:
                    split_idx = i + 1
                    break
            
            if split_idx == -1:
                # 如果没找到标点符号，暴力截断
                split_idx = child_size
                
            chunks.append(remaining[:split_idx])
            remaining = remaining[split_idx:]
            
        if remaining:
            chunks.append(remaining)
        return chunks

    def create_parent_child_chunks(self, text: str) -> list[dict]:
        if not text:
            return []
            
        # 1. 分句
        # 使用正向后瞻切分，保留标点
        raw_sentences = re.split(r'(?<=[。？！\n!?])', text)
        sentences = [s for s in raw_sentences if s]
        
        if not sentences:
            return []
            
        n = len(sentences)
        # 如果只有一句话，无法计算相似度，直接作为 parent chunk
        if n <= 1:
            parent_chunks = [text]
        else:
            # 2. 计算切分点 i (1 <= i < n) 的左右窗口 Embedding 相似度
            similarities = []
            for i in range(1, n):
                # 左侧窗口
                left_start = max(0, i - self.window_size)
                left_text = "".join(sentences[left_start:i])
                
                # 右侧窗口
                right_end = min(n, i + self.window_size)
                right_text = "".join(sentences[i:right_end])
                
                v_left = self.embedding_service.get_dense_embedding(left_text)
                v_right = self.embedding_service.get_dense_embedding(right_text)
                
                sim = self._cosine_similarity(v_left, v_right)
                similarities.append(sim)
                
            # 3. 确定切分阈值
            if self.threshold is not None:
                threshold = self.threshold
            else:
                if len(similarities) >= 2:
                    mean_sim = sum(similarities) / len(similarities)
                    variance = sum((x - mean_sim) ** 2 for x in similarities) / len(similarities)
                    std_sim = math.sqrt(variance)
                    threshold = mean_sim - 0.8 * std_sim
                else:
                    threshold = 0.5 # 默认 fallback 阈值
                    
            # 4. 根据阈值切分 Parent Chunks
            parent_chunks = []
            current_chunk = [sentences[0]]
            for i in range(1, n):
                sim = similarities[i - 1]
                if sim < threshold:
                    # 在此处切分
                    parent_chunks.append("".join(current_chunk))
                    current_chunk = [sentences[i]]
                else:
                    current_chunk.append(sentences[i])
            if current_chunk:
                parent_chunks.append("".join(current_chunk))
                
        # 5. 生成 Parent-Child 映射
        result = []
        for parent_text in parent_chunks:
            # 生成唯一的 parent_id
            parent_id = hashlib.md5(parent_text.encode('utf-8')).hexdigest()
            child_texts = self._split_parent_to_chunks(parent_text, self.child_size)
            for child_text in child_texts:
                result.append({
                    "child_text": child_text,
                    "parent_text": parent_text,
                    "parent_id": parent_id
                })
                
        return result

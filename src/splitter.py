import re
import math
import hashlib
import mistune

def render_inline(children) -> str:
    if not children:
        return ""
    result = []
    for child in children:
        ctype = child.get("type")
        if ctype == "text":
            result.append(child.get("raw", ""))
        elif ctype == "softbreak":
            result.append("\n")
        elif ctype == "linebreak":
            result.append("\n")
        elif ctype == "emphasis":
            result.append(f"*{render_inline(child.get('children'))}*")
        elif ctype == "strong":
            result.append(f"**{render_inline(child.get('children'))}**")
        elif ctype == "codespan":
            result.append(f"`{child.get('raw', '')}`")
        elif ctype == "link":
            url = child.get("attrs", {}).get("url", "")
            title = child.get("attrs", {}).get("title")
            title_part = f' "{title}"' if title else ''
            result.append(f"[{render_inline(child.get('children'))}]({url}{title_part})")
        elif ctype == "image":
            url = child.get("attrs", {}).get("url", "")
            result.append(f"![{render_inline(child.get('children'))}]({url})")
        elif ctype == "raw_html" or ctype == "inline_html":
            result.append(child.get("raw", ""))
        else:
            if "raw" in child:
                result.append(child["raw"])
            elif "children" in child:
                result.append(render_inline(child["children"]))
    return "".join(result)

def render_ast_node_to_md(node) -> str:
    ntype = node.get("type")
    
    if ntype == "heading":
        level = node.get("attrs", {}).get("level", 1)
        content = render_inline(node.get("children"))
        return f"{'#' * level} {content}"
        
    elif ntype == "paragraph":
        return render_inline(node.get("children"))
        
    elif ntype == "block_code":
        info = node.get("attrs", {}).get("info", "")
        raw = node.get("raw", "")
        if not raw.endswith("\n"):
            raw += "\n"
        return f"```{info}\n{raw}```"
        
    elif ntype == "thematic_break":
        return "---"
        
    elif ntype == "block_text":
        return render_inline(node.get("children"))
        
    elif ntype == "list_item":
        parts = []
        for child in node.get("children", []):
            parts.append(render_ast_node_to_md(child))
        return "\n".join(parts)
        
    elif ntype == "list":
        ordered = node.get("attrs", {}).get("ordered", False)
        bullet = node.get("bullet", "-")
        items = []
        for idx, item_node in enumerate(node.get("children", [])):
            item_content = render_ast_node_to_md(item_node)
            lines = item_content.split("\n")
            prefix = f"{idx + 1}. " if ordered else f"{bullet} "
            indent = " " * len(prefix)
            
            first_line = prefix + lines[0]
            rest_lines = [indent + l for l in lines[1:]]
            items.append("\n".join([first_line] + rest_lines))
        return "\n".join(items)
        
    elif ntype == "table":
        headers = []
        alignments = []
        rows = []
        
        for child in node.get("children", []):
            if child.get("type") == "table_head":
                for cell in child.get("children", []):
                    headers.append(render_inline(cell.get("children")))
                    alignments.append(cell.get("attrs", {}).get("align"))
            elif child.get("type") == "table_body":
                for row_child in child.get("children", []):
                    if row_child.get("type") == "table_row":
                        row_cells = []
                        for cell in row_child.get("children", []):
                            row_cells.append(render_inline(cell.get("children")))
                        rows.append(row_cells)
                        
        lines = []
        lines.append("| " + " | ".join(headers) + " |")
        sep_parts = []
        for align in alignments:
            if align == "left":
                sep_parts.append(":---")
            elif align == "right":
                sep_parts.append("---:")
            elif align == "center":
                sep_parts.append(":---:")
            else:
                sep_parts.append("---")
        lines.append("| " + " | ".join(sep_parts) + " |")
        for r in rows:
            lines.append("| " + " | ".join(r) + " |")
        return "\n".join(lines)
        
    else:
        if "raw" in node:
            return node["raw"]
        elif "children" in node:
            parts = []
            for child in node["children"]:
                parts.append(render_ast_node_to_md(child))
            return "".join(parts)
        return ""

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

    def create_parent_child_chunks(self, text: str, is_markdown: bool = False) -> list[dict]:
        if not text:
            return []
            
        placeholders = {}
        
        # 1. 分句
        if is_markdown:
            # 兼容处理：去除 Windows 下常见的 UTF-8 BOM 头，以防正则匹配失败
            text_clean = text.lstrip('\ufeff')
            # 保护 YAML 头部
            yaml_header = ""
            yaml_match = re.match(r'^---\s*\n([\s\S]*?)\n---', text_clean)
            if yaml_match:
                yaml_header = yaml_match.group(0)
                text_to_parse = text_clean[yaml_match.end():]
            else:
                text_to_parse = text_clean
                
            parser = mistune.create_markdown(renderer=None, plugins=['table'])
            ast = parser(text_to_parse)
            
            sentences = []
            if yaml_header:
                sentences.append(yaml_header)
                
            for node in ast:
                if node.get("type") == "blank_line":
                    continue
                
                rendered = render_ast_node_to_md(node)
                if not rendered.strip():
                    continue
                    
                ntype = node.get("type")
                # 保护代码块与表格
                if ntype in ("block_code", "table"):
                    ph = f"__BLOCK_PH_{len(placeholders)}__"
                    placeholders[ph] = rendered
                    sentences.append(ph)
                else:
                    sentences.append(rendered)
        else:
            # 传统文本文档的断句逻辑：在标点符号 (。？！!?) 或双换行 (\n\n) 处切分
            raw_sentences = re.split(r'(?<=[。？！!?])|(?<=\n\n)', text)
            sentences = [s for s in raw_sentences if s and s.strip()]
        
        if not sentences:
            return []
            
        n = len(sentences)
        # 如果只有一句话，无法计算相似度，直接作为 parent chunk
        if n <= 1:
            parent_chunks = [text]
        else:
            # 2. 计算切分点 i (1 <= i < n) 的左右窗口 Embedding 相似度
            boundary_texts = []
            all_texts_set = set()
            
            def restore_placeholders(t: str) -> str:
                for ph, orig in placeholders.items():
                    t = t.replace(ph, orig)
                return t

            for i in range(1, n):
                # 左侧窗口
                left_start = max(0, i - self.window_size)
                left_text = "".join(sentences[left_start:i])
                
                # 右侧窗口
                right_end = min(n, i + self.window_size)
                right_text = "".join(sentences[i:right_end])
                
                # 计算向量相似度时，还原占位符以计算真实文本语义
                left_text_real = restore_placeholders(left_text) if is_markdown else left_text
                right_text_real = restore_placeholders(right_text) if is_markdown else right_text
                
                boundary_texts.append((left_text_real, right_text_real))
                all_texts_set.add(left_text_real)
                all_texts_set.add(right_text_real)
                
            unique_texts = list(all_texts_set)
            
            # 优先使用批量生成向量接口，否则 fallback 到单条生成接口
            if hasattr(self.embedding_service, "get_dense_embeddings_batch"):
                vectors_list = self.embedding_service.get_dense_embeddings_batch(unique_texts)
                text_to_vector = dict(zip(unique_texts, vectors_list))
            else:
                text_to_vector = {}
                for t in unique_texts:
                    text_to_vector[t] = self.embedding_service.get_dense_embedding(t)
                    
            similarities = []
            for left_text, right_text in boundary_texts:
                v_left = text_to_vector[left_text]
                v_right = text_to_vector[right_text]
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
                
        # 4.5 兜底合并超短 parent_chunks (保证召回丰富文本，最小 150 字符，仅在 markdown 模式下启用以避免破坏传统文本基准测试)
        if is_markdown:
            min_parent_size = 150
            merged_parents = []
            temp_parent = ""
            for p_chunk in parent_chunks:
                if not temp_parent:
                    temp_parent = p_chunk
                else:
                    if len(temp_parent) < min_parent_size:
                        temp_parent = temp_parent.rstrip('\n') + '\n' + p_chunk.lstrip('\n')
                    else:
                        merged_parents.append(temp_parent)
                        temp_parent = p_chunk
            if temp_parent:
                if merged_parents and len(temp_parent) < min_parent_size:
                    merged_parents[-1] = merged_parents[-1].rstrip('\n') + '\n' + temp_parent.lstrip('\n')
                else:
                    merged_parents.append(temp_parent)
            parent_chunks = merged_parents

        # 5. 生成 Parent-Child 映射并还原占位符
        result = []
        
        def restore_placeholders(t: str) -> str:
            for ph, orig in placeholders.items():
                t = t.replace(ph, orig)
            return t

        for parent_raw in parent_chunks:
            parent_text = restore_placeholders(parent_raw) if is_markdown else parent_raw
            # 生成唯一的 parent_id
            parent_id = hashlib.md5(parent_text.encode('utf-8')).hexdigest()
            child_raw_texts = self._split_parent_to_chunks(parent_raw, self.child_size)
            for child_raw in child_raw_texts:
                child_text = restore_placeholders(child_raw) if is_markdown else child_raw
                result.append({
                    "child_text": child_text,
                    "parent_text": parent_text,
                    "parent_id": parent_id
                })
                
        return result

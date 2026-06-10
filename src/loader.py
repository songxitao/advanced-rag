import os
import re
import fitz
import docx

class DocumentLoader:
    def load(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
                
        elif ext == '.pdf':
            text = []
            doc = fitz.open(file_path)
            for page in doc:
                text.append(page.get_text())
            doc.close()
            return "".join(text).strip()
            
        elif ext == '.docx':
            doc = docx.Document(file_path)
            text = [p.text for p in doc.paragraphs]
            return "\n".join(text).strip()
            
        elif ext == '.srt':
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            cleaned_lines = []
            # 匹配时间轴格式，如: 00:00:01,000 --> 00:00:04,000 或者 00:00:01
            # 比较通用的检测是否是时间轴的方法：检查行中是否包含 '-->' 符号
            # 或者用正则匹配时间
            time_pattern = re.compile(r'\d{2}:\d{2}:\d{2}')
            for line in lines:
                line_str = line.strip()
                if not line_str:
                    continue
                # 过滤序号
                if line_str.isdigit():
                    continue
                # 过滤包含时间轴的行
                if '-->' in line_str or time_pattern.search(line_str):
                    continue
                cleaned_lines.append(line_str)
            return "\n".join(cleaned_lines).strip()
            
        else:
            raise ValueError(f"Unsupported file format: {ext}")

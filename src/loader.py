import os
import re
import fitz
import requests
import logging

class DocumentLoader:
    def _convert_via_markitdown(self, file_path: str) -> str:
        """
        通过子进程调用 markitdown CLI 将文件转换为 Markdown 格式字符串。
        markitdown 安装在独立的 conda 环境 markitdown-env 中。
        :param file_path: 待转换文件的绝对或相对路径
        :return: 转换后的 Markdown 字符串
        :raises RuntimeError: markitdown 调用失败时抛出
        """
        import subprocess
        import sys
        conda_cmd = "conda"
        if sys.platform == "win32":
            conda_cmd = "conda.bat"
            common_paths = [
                r"D:\program files\Miniconda\condabin\conda.bat",
                r"C:\Users\song\miniconda3\condabin\conda.bat",
                r"C:\ProgramData\anaconda3\condabin\conda.bat",
            ]
            for cp in common_paths:
                if os.path.exists(cp):
                    conda_cmd = cp
                    break
        abs_path = os.path.abspath(file_path)
        try:
            result = subprocess.run(
                [conda_cmd, "run", "-n", "markitdown-env", "markitdown", abs_path],
                capture_output=True,
                text=True,
                timeout=120,
                encoding="utf-8"
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"markitdown 转换失败 (exit code {result.returncode}): {result.stderr.strip()}"
                )
            output = result.stdout.strip()
            if not output:
                raise RuntimeError(f"markitdown 转换结果为空: {abs_path}")
            return output
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"markitdown 运行超时(120s): {abs_path}")
        except FileNotFoundError:
            raise RuntimeError("conda 命令未找到，请确保 conda 已安装并放入 PATH 中")

    def load(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
                
        elif ext == '.pdf':
            try:
                url = "http://127.0.0.1:8010/file_parse"
                data = {
                    "backend": "pipeline",
                    "parse_method": "auto",
                    "formula_enable": "true",
                    "table_enable": "true",
                    "return_md": "true",
                    "response_format_zip": "false"
                }
                with open(file_path, 'rb') as f:
                    files = [("files", (os.path.basename(file_path), f, "application/pdf"))]
                    response = requests.post(url, data=data, files=files, timeout=60)
                    response.raise_for_status()
                    res_json = response.json()
                    
                    md_content = None
                    if isinstance(res_json, dict):
                        filename_stem = os.path.splitext(os.path.basename(file_path))[0]
                        # 优先从 results -> filename_stem -> md_content 路径提取
                        md_content = res_json.get("results", {}).get(filename_stem, {}).get("md_content")
                        # 兼容其它可能的格式
                        if md_content is None:
                            md_content = res_json.get("md_content")
                        if md_content is None and isinstance(res_json.get("data"), dict):
                            md_content = res_json["data"].get("md_content")
                            
                    if md_content is None:
                        raise KeyError("md_content not found in response JSON")
                    return md_content.strip()
            except Exception as e:
                logging.warning(f"MinerU parsing failed ({e}). Falling back to fitz (PyMuPDF).")
                text = []
                doc = fitz.open(file_path)
                for page in doc:
                    text.append(page.get_text())
                doc.close()
                return "".join(text).strip()
            
        elif ext == '.docx':
            return self._convert_via_markitdown(file_path)
            
        elif ext in ['.xlsx', '.xls']:
            return self._convert_via_markitdown(file_path)
            
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

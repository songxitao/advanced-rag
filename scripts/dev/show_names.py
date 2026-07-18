import os
import sys
import json
import re
from collections import Counter
import jieba.posseg as pseg

# 强制输出为 UTF-8
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    input_path = r"E:/project/pyltp-books-master/pyltp-books-master/mybooks/Book/三国演义白话文"
    
    # 1. 加载文本
    files = [f for f in os.listdir(input_path) if f.endswith('.txt')]
    def extract_num(filename):
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else 0
    files.sort(key=extract_num)
    
    content = []
    for file in files:
        p = os.path.join(input_path, file)
        with open(p, 'r', encoding='utf-8') as f:
            content.append(f.read())
    text = "\n".join(content)
    
    # 2. 提取 nr (包括 nrfg 等以 nr 开头的所有人名词性) 和 nz (其他专名)
    counts = Counter()
    words = pseg.cut(text)
    
    for word, flag in words:
        # 支持捕获 nrfg 等所有人名词性细分
        if (flag.startswith('nr') or flag == 'nz') and len(word) >= 2:
            counts[word] += 1
        elif word in ("玄德", "翼德", "云长", "孔明", "孟德", "阿瞒", "卧龙", "皇叔", "都游", "督邮") and len(word) >= 2:
            counts[word] += 1
            
    # 按词频从高到低排序，过滤掉只出现过 1 次的词
    sorted_names = [{"name": name, "count": count} for name, count in counts.most_common() if count >= 2]
    
    output_file = r"e:/project/advanced-rag/tests/temp_data/extracted_names.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sorted_names, f, ensure_ascii=False, indent=2)
        
    print(f"成功保存高精分词人名词表到 {output_file}，共计 {len(sorted_names)} 个。")

if __name__ == "__main__":
    main()

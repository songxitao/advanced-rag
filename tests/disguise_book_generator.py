import os
import re
import json
import requests
import jieba.posseg as pseg
from collections import Counter

# Local Qwen API configuration
LLM_API_URL = "http://localhost:8080/v1/chat/completions"
MODEL_NAME = "qwen3.6-35b-a3b-distilled-think"

# Static pre-defined fallback mapping table
STATIC_ALIASES = {
    "刘备": ["刘备", "玄德", "刘玄德", "玄德公", "皇叔"],
    "曹操": ["曹操", "孟德", "曹孟德", "阿瞒"],
    "关羽": ["关羽", "云长", "关云长", "关公"],
    "张飞": ["张飞", "翼德", "张翼德"],
    "诸葛亮": ["诸葛亮", "孔明", "卧龙"],
    "孙权": ["孙权", "仲谋", "吴侯"],
    "周瑜": ["周瑜", "公瑾"],
    "吕布": ["吕布", "奉先"],
    "赵云": ["赵云", "子龙"],
    "司马懿": ["司马懿", "仲达"],
    "袁绍": ["袁绍", "本初"],
    "董卓": ["董卓", "仲颖"]
}

def load_book_text(input_path):
    """
    Robust loader that can load from a single text file or a directory containing sorted chapter text files.
    """
    if os.path.isdir(input_path):
        files = [f for f in os.listdir(input_path) if f.endswith('.txt')]
        # Sort files numerically if their name is digits
        def extract_num(filename):
            match = re.search(r'\d+', filename)
            return int(match.group()) if match else 0
        files.sort(key=extract_num)
        
        content = []
        for file in files:
            p = os.path.join(input_path, file)
            with open(p, 'r', encoding='utf-8') as f:
                content.append(f.read())
        return "\n".join(content)
    else:
        with open(input_path, 'r', encoding='utf-8') as f:
            return f.read()

def extract_json(text: str):
    """
    Robust JSON parser that filters out <think> tags and markdown code blocks.
    """
    # Clean think tags (including unclosed think blocks due to truncation)
    text = re.sub(r'<think>[\s\S]*?(?:</think>|$)', '', text).strip()
    
    # Try matching markdown code block content
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if match:
        text = match.group(1).strip()
        
    # Extract the outermost JSON braces
    match_braces = re.search(r'(\{[\s\S]*\})', text)
    if match_braces:
        text = match_braces.group(1).strip()
        
    # Clean trailing commas in objects or lists to prevent JSON decode errors
    text = re.sub(r',\s*([\]}])', r'\1', text)
    return json.loads(text)

def cluster_names_with_llm(names):
    """
    Send the frequent names list to local Qwen to cluster them.
    If it fails, automatically falls back to static pre-defined alias mapping.
    """
    if not names:
        print("[Generator] Names list is empty, falling back to static mapping.")
        return STATIC_ALIASES

    prompt = f"""你是一个精通《三国演义》的古典文学专家。
以下是我们在文本中提取出的高频人名/称呼词汇（频次 >= 15且词长 >= 2）：
{json.dumps(names, ensure_ascii=False)}

请对这些人名与别称（包括字、号、简称、全称等，例如刘备与玄德、刘玄德、玄德公；关羽与云长、关云长；曹操与孟德、曹孟德等）进行合并聚类。
要求：
1. 聚类时，仅对上面给出的列表中的词汇进行归并，必须是列表里存在的词。不要捏造或引入列表中不存在的词。
2. 聚类输出必须是标准的 JSON 对象，它的键是每个角色的“标准姓名”（如“刘备”、“曹操”、“关羽”、“张飞”、“诸葛亮”、“孙权”等），值是该角色所有关联别称的列表。
3. 请只输出 JSON 内容本身，不要有任何多余的解释、Markdown 格式标记（如 ```json ... ```）或思考过程。

示例：
{{
  "刘备": ["刘备", "玄德", "刘玄德", "玄德公"],
  "曹操": ["曹操", "孟德", "曹孟德"]
}}
"""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that outputs raw JSON content."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 2048
    }

    try:
        print(f"[Generator] Requesting LLM to cluster {len(names)} names...")
        resp = requests.post(LLM_API_URL, json=payload, timeout=45)
        resp.raise_for_status()
        content = resp.json()['choices'][0]['message']['content'].strip()
        
        # Clean and extract JSON
        clusters = extract_json(content)
        if isinstance(clusters, dict) and len(clusters) > 0:
            print(f"[Generator] Successfully clustered {len(clusters)} characters from LLM.")
            return clusters
        else:
            print("[Generator] LLM returned empty or invalid dict. Falling back to static mapping.")
            return STATIC_ALIASES
    except Exception as e:
        print(f"[Generator] LLM API call error: {e}. Falling back to static mapping.")
        return STATIC_ALIASES

def run_disguise_pipeline(input_file, output_dir):
    """
    Main pipeline function:
    1. Load book text.
    2. Segment using jieba.posseg, filter for 'nr' tag, frequency >= 15, length >= 2.
    3. Send names list to LLM API for alias clustering (fallback to static on failure).
    4. Save the alias clusters JSON as sanguo_aliases.json in output_dir.
    5. Build disguise mappings, sort by original name length in descending order, and replace.
    6. Save the disguised text as 三国演义白话文_disguised.txt in output_dir.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load book text
    print(f"[Generator] Loading text from {input_file}...")
    text = load_book_text(input_file)
    print(f"[Generator] Loaded {len(text)} characters of text.")
    
    # 2. Extract frequent person names using jieba.posseg
    print("[Generator] Extracting frequent names using jieba.posseg...")
    counts = Counter()
    words = pseg.cut(text)
    for word, flag in words:
        if flag == 'nr' and len(word) >= 2:
            counts[word] += 1
            
    frequent_names = [name for name, count in counts.items() if count >= 15]
    print(f"[Generator] Found {len(frequent_names)} names with frequency >= 15 and length >= 2.")
    
    # 3. Request LLM clustering or fallback
    clusters = cluster_names_with_llm(frequent_names)
    
    # 4. Save clusters to sanguo_aliases.json
    alias_json_path = os.path.join(output_dir, "sanguo_aliases.json")
    with open(alias_json_path, 'w', encoding='utf-8') as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)
    print(f"[Generator] Saved alias clusters to {alias_json_path}")
    
    # 5. Build replacement pairs and sort by original name length in descending order
    replace_pairs = []
    for idx, (std_name, aliases) in enumerate(clusters.items(), 1):
        if not isinstance(aliases, list):
            aliases = [aliases]
        # Collect unique aliases + standard name
        unique_aliases = set(aliases)
        unique_aliases.add(std_name)
        
        # Obfuscation code
        code = f"[角色_{idx}]"
        for alias in unique_aliases:
            alias = alias.strip()
            if alias:
                replace_pairs.append((alias, code))
                
    # Sort pairs by original name length in descending order (longest name first)
    replace_pairs.sort(key=lambda x: len(x[0]), reverse=True)
    
    # 6. Global replace
    print("[Generator] Obfuscating names in text...")
    disguised_text = text
    for alias, code in replace_pairs:
        disguised_text = disguised_text.replace(alias, code)
        
    # Save disguised text
    disguised_txt_path = os.path.join(output_dir, "三国演义白话文_disguised.txt")
    with open(disguised_txt_path, 'w', encoding='utf-8') as f:
        f.write(disguised_text)
    print(f"[Generator] Saved disguised text to {disguised_txt_path}")
    
    return alias_json_path, disguised_txt_path

import os
import re
import json
import requests
import jieba
import jieba.posseg as pseg
from collections import Counter

# Local Qwen API configuration
LLM_API_URL = "http://localhost:8080/v1/chat/completions"
MODEL_NAME = "qwen3.6-35b-a3b-opus-nothink"

# 核心词表，用于强行注入 jieba 分词器，防止粘连（如“许褚入”切分成一个词）
CORE_CHARACTERS = [
    "刘备", "关羽", "张飞", "诸葛亮", "曹操", "孙权", "周瑜", "吕布", "赵云", "司马懿",
    "袁绍", "董卓", "鲁肃", "魏延", "黄忠", "马超", "庞统", "陆逊", "邓艾", "姜维",
    "司马昭", "司马师", "曹丕", "袁术", "刘表", "刘璋", "王允", "蒋干", "孟获", "张松",
    "郭汜", "诸葛瑾", "孙坚", "孙策", "曹植", "曹真", "曹爽", "张鲁", "许褚", "夏侯惇",
    "夏侯渊", "张辽", "徐晃", "张郃", "太史慈", "甘宁", "吕蒙", "魏国", "蜀国", "吴国"
]

CORE_LOCATIONS = [
    "建业", "洛阳", "长安", "许昌", "荆州", "益州", "徐州", "兖州", "扬州", "冀州",
    "幽州", "并州", "凉州", "交州", "豫州", "青州", "汉中", "江夏", "新野", "樊城",
    "宛城", "寿春", "柴桑", "赤壁", "官渡", "临泾", "街亭", "五丈原", "西凉", "祁山",
    "合肥", "西川", "东川", "夏口", "南郡", "襄阳", "江陵", "许都", "琢郡"
]

# 静态Fallback对照表
STATIC_FALLBACK = {
    "characters": {
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
        "董卓": ["董卓", "仲颖"],
        "许褚": ["许褚"]
    },
    "locations": {
        "荆州": ["荆州"],
        "洛阳": ["洛阳", "东都"],
        "长安": ["长安"],
        "许昌": ["许昌", "许都"]
    }
}

# 注册词典
print("[Generator] Injecting core vocabularies into jieba dictionary...")
for char in CORE_CHARACTERS:
    jieba.add_word(char, tag="nr")
for loc in CORE_LOCATIONS:
    jieba.add_word(loc, tag="ns")

def load_book_text(input_path):
    """
    Robust loader that can load from a single text file or a directory containing sorted chapter text files.
    """
    if os.path.isdir(input_path):
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
        return "\n".join(content)
    else:
        with open(input_path, 'r', encoding='utf-8') as f:
            return f.read()

def extract_json(text: str):
    """
    Robust JSON parser that filters out <think> tags and markdown code blocks.
    """
    text = re.sub(r'<think>[\s\S]*?(?:</think>|$)', '', text).strip()
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if match:
        text = match.group(1).strip()
    match_braces = re.search(r'(\{[\s\S]*\})', text)
    if match_braces:
        text = match_braces.group(1).strip()
    text = re.sub(r',\s*([\]}])', r'\1', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"[Generator] JSON decode error: {e}. Returning empty dictionary.")
        return {}

def cluster_entities_with_llm(names, locations):
    """
    Send frequent names and locations to local Qwen to cluster them.
    """
    if not names and not locations:
        print("[Generator] Names and locations are empty, using static fallback.")
        return STATIC_FALLBACK

    prompt = f"""你是一个精通《三国演义》的古典文学专家。
以下是我们提取出的候选实体：
1. 候选人名/称呼列表：
{json.dumps(names, ensure_ascii=False)}

2. 候选地名/场所列表：
{json.dumps(locations, ensure_ascii=False)}

请以古典文学专家身份，分析上述列表中是否存在指向相同人或相同地点的别称（例如刘备与玄德、玄德公；曹操与曹孟德、曹军；洛阳与洛一陽等）。
要求：
1. 仅对上述列表中指向相同实体的同义别称进行合并聚类，输出到 "characters" 和 "locations" 模块。
2. 如果列表中某个实体没有任何同义别称，请直接忽略，不要输出到 JSON 中。
3. 聚类结果以标准 JSON 对象输出。它的键是标准名，值是该实体关联的所有别称（别称必须来自于上面的输入列表，不要捏造）。
4. 请只输出 JSON 内容本身，不要有任何多余的解释、Markdown 格式标记（如 ```json ... ```）或思考过程。

示例：
{{
  "characters": {{
    "刘备": ["刘备", "玄德", "刘玄德", "玄德公"],
    "曹操": ["曹操", "曹孟德", "曹军"]
  }},
  "locations": {{
    "洛阳": ["洛阳", "洛一陽"],
    "许昌": ["许昌", "许都"]
  }}
}}
"""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that outputs raw JSON content."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 4096
    }

    try:
        print(f"[Generator] Requesting LLM to cluster entities...")
        resp = requests.post(LLM_API_URL, json=payload, timeout=45)
        resp.raise_for_status()
        content = resp.json()['choices'][0]['message']['content'].strip()
        
        clusters = extract_json(content)
        if isinstance(clusters, dict) and ("characters" in clusters or "locations" in clusters):
            print(f"[Generator] Successfully clustered characters and locations from LLM.")
            return clusters
        else:
            print("[Generator] LLM returned invalid structure. Using fallback.")
            return STATIC_FALLBACK
    except Exception as e:
        print(f"[Generator] LLM API call error: {e}. Using fallback.")
        return STATIC_FALLBACK

def run_disguise_pipeline(input_file, output_dir):
    """
    Main pipeline:
    1. Load book text.
    2. Segment using jieba.posseg:
       - Collect person names (tag == 'nr') with count >= 15
       - Collect location names (tag == 'ns') with count >= 10
    3. Use LLM to cluster names and locations.
    4. Save to json.
    5. Build disguise replacement pairs.
       - characters: [角色_1], [角色_2], ...
       - locations: [地点_1], [地点_2], ...
    6. Run global replacement and save disguised text.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[Generator] Loading text from {input_file}...")
    text = load_book_text(input_file)
    print(f"[Generator] Loaded {len(text)} characters.")
    
    print("[Generator] Extracting entities using jieba.posseg...")
    char_counts = Counter()
    loc_counts = Counter()
    
    words = pseg.cut(text)
    for word, flag in words:
        if len(word) < 2:
            continue
        if flag == 'nr':
            char_counts[word] += 1
        elif flag == 'ns':
            loc_counts[word] += 1
            
    # 只改高频
    frequent_chars = [name for name, count in char_counts.items() if count >= 8]
    frequent_locs = [loc for loc, count in loc_counts.items() if count >= 5]
    
    print(f"[Generator] Found {len(frequent_chars)} high-frequency names, {len(frequent_locs)} high-frequency locations.")
    
    # 写入提取出的原始实体列表，便于审查
    extracted_path = os.path.join(output_dir, "extracted_names.json")
    with open(extracted_path, 'w', encoding='utf-8') as f:
        json.dump({
            "frequent_characters": [{"name": n, "count": char_counts[n]} for n in frequent_chars],
            "frequent_locations": [{"name": l, "count": loc_counts[l]} for l in frequent_locs]
        }, f, ensure_ascii=False, indent=2)
    print(f"[Generator] Saved raw extracted entities to {extracted_path}")

    # LLM 聚类
    clusters = cluster_entities_with_llm(frequent_chars, frequent_locs)
    
    alias_json_path = os.path.join(output_dir, "sanguo_aliases.json")
    with open(alias_json_path, 'w', encoding='utf-8') as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)
    print(f"[Generator] Saved clustered entities to {alias_json_path}")
    
    # 构建替换映射
    replace_pairs = []
    # 构建替换映射
    replace_pairs = []
    
    # 1. 角色替换 (characters)
    char_clusters = clusters.get("characters", {})
    already_clustered_chars = set()
    
    char_idx = 1
    for std_name, aliases in char_clusters.items():
        if not isinstance(aliases, list):
            aliases = [aliases]
        unique_aliases = set(aliases)
        unique_aliases.add(std_name)
        
        code = f"[角色_{char_idx}]"
        char_idx += 1
        for alias in unique_aliases:
            alias = alias.strip()
            if len(alias) >= 2:
                replace_pairs.append((alias, code))
                already_clustered_chars.add(alias)
                
    # 自动补充未被聚类的高频人名，保证覆盖度
    for char in frequent_chars:
        if char not in already_clustered_chars:
            code = f"[角色_{char_idx}]"
            char_idx += 1
            replace_pairs.append((char, code))
            char_clusters[char] = [char]
            already_clustered_chars.add(char)
                
    # 2. 地点替换 (locations)
    loc_clusters = clusters.get("locations", {})
    already_clustered_locs = set()
    
    loc_idx = 1
    for std_loc, aliases in loc_clusters.items():
        if not isinstance(aliases, list):
            aliases = [aliases]
        unique_aliases = set(aliases)
        unique_aliases.add(std_loc)
        
        code = f"[地点_{loc_idx}]"
        loc_idx += 1
        for alias in unique_aliases:
            alias = alias.strip()
            if len(alias) >= 2:
                replace_pairs.append((alias, code))
                already_clustered_locs.add(alias)
                
    # 自动补充未被聚类的高频地名
    for loc in frequent_locs:
        if loc not in already_clustered_locs:
            code = f"[地点_{loc_idx}]"
            loc_idx += 1
            replace_pairs.append((loc, code))
            loc_clusters[loc] = [loc]
            already_clustered_locs.add(loc)
            
    # 将最终完整的聚类表回写到 JSON
    clusters["characters"] = char_clusters
    clusters["locations"] = loc_clusters
    
    # 重新保存完整对照表
    with open(alias_json_path, 'w', encoding='utf-8') as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)
    print(f"[Generator] Saved updated complete alias clusters to {alias_json_path}")
                
    # 排序：字数长的别称先替换，防止短别称误伤长别称（例如先替"刘玄德"再替"刘备"）
    replace_pairs.sort(key=lambda x: len(x[0]), reverse=True)
    
    # 全局替换
    print("[Generator] Obfuscating entities in text...")
    disguised_text = text
    for original, code in replace_pairs:
        disguised_text = disguised_text.replace(original, code)
        
    disguised_txt_path = os.path.join(output_dir, "三国演义白话文_disguised.txt")
    with open(disguised_txt_path, 'w', encoding='utf-8') as f:
        f.write(disguised_text)
    print(f"[Generator] Saved disguised text to {disguised_txt_path}")
    
    return alias_json_path, disguised_txt_path

if __name__ == "__main__":
    input_p = r"E:/project/pyltp-books-master/pyltp-books-master/mybooks/Book/三国演义白话文"
    output_d = r"e:/project/advanced-rag/tests/temp_data"
    run_disguise_pipeline(input_p, output_d)



import os
import sys
import json
import random
import re
import requests

# Reconfigure stdout/stderr for Windows UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# 使用环境变量并提供默认路径
OUTPUT_PATH = "tests/temp_data/test_sanguo_dataset.json"
LLM_API_URL = "http://localhost:8080/v1/chat/completions"
MODEL_NAME = "qwen3.6-35b-a3b-opus-nothink"

# 实体过滤黑名单，防止常识和原名泄漏
BLACKLIST_ENTITIES = [
    "刘备", "关羽", "张飞", "曹操", "孙权", "诸葛亮", "周瑜", "吕布", "赵云", "司马懿",
    "袁绍", "董卓", "鲁肃", "魏延", "黄忠", "马超", "庞统", "陆逊", "邓艾", "姜维",
    "司马昭", "司马师", "曹丕", "袁术", "刘表", "刘璋", "王允", "蒋干", "孟获", "张松",
    "夏侯惇", "夏侯渊", "张辽", "徐晃", "张郃", "太史慈", "甘宁", "吕蒙", "马岱", "典韦",
    "荆州", "益州", "徐州", "兖州", "扬州", "冀州", "幽州", "并州", "凉州", "交州",
    "洛阳", "建业", "成都", "长安", "汉中", "许昌", "新野", "樊城", "赤壁", "官渡", "白帝城", "托孤"
]

def get_semantic_paragraph_chunks(txt_path, target_length=800):
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"未找到文档文件: '{txt_path}'。")
        
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = []
    current_length = 0
    for p in paragraphs:
        p_text = p.strip()
        if not p_text:
            continue
        current_chunk.append(p_text)
        current_length += len(p_text)
        if current_length >= target_length:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_length = 0
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    return chunks

def extract_json(text: str):
    """
    安全提取并解析 LLM 输出中的 JSON 对象。
    """
    text = re.sub(r'<think>[\s\S]*?(?:</think>|$)', '', text).strip()
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if match:
        text = match.group(1).strip()
    match_braces = re.search(r'(\{[\s\S]*\})', text)
    if match_braces:
        text = match_braces.group(1).strip()
    text = re.sub(r',\s*([\]}])', r'\1', text)
    return json.loads(text)

def generate_qa_pair(chunk, idx):
    prompt = f"""你是一个极其严谨的、专门用于测试 RAG 检索器召回能力的学术评测出题官。
请阅读以下从完全脱敏的三国演义片段（其中所有核心人物、核心地名都已被替换成了形如 `[角色_N]` 和 `[地点_M]` 的代号）：

【脱敏文本片段】：
{chunk}

请基于上述文本，设计一个必须要跨段落/跨句子进行逻辑链式推理的问题，并给出标准答案。

【出题铁律（违反任何一条则测试失败）】：
1. 答案（ground_truth）必须且只能是文本中出现的伪装代号，形如 `[角色_X]` 或 `[地点_Y]`（例如 `[角色_24]` 或 `[地点_5]`）。绝对不能是任何历史上真实的中文人名（如“马岱”、“刘备”、“关羽”等）或真实地名！
2. 问题本身中绝对不能出现任何形如 `[角色_X]`、`[地点_Y]` 的代号。
3. 问题本身也严禁使用任何著名的历史事件专有名词（如“赤壁之战”、“桃园结义”、“白帝城托孤”、“连环计”、“凤仪亭”、“割发代首”等历史典故或专属事件词汇）。必须完全基于此片段中的客观物理事实描述进行逻辑指代提问。
   * 错误示例：“在赤壁之战中被派去追杀孔明的都督是谁？”（泄露了赤壁之战和孔明两个常识词，且答案不是代号）
   * 正确示例：“在文中因‘万事皆备只欠东风’而生病躺在床上的那位角色，其病好后派去七星坛追杀那位作法借风谋士的都督，其对应的代号是什么？”（答案：`[角色_35]`）

请严格按照以下 JSON 格式输出，不要包含任何多余解释、Markdown 代码标记或思考过程：
{{
  "question": "隐式事实指代推理问题（不得出现任何代号，也不得出现任何真实历史人名/地名/典故名称）",
  "ground_truth": "标准答案（必须是类似于 `[角色_X]` 或 `[地点_Y]` 的规范代号，不能包含其他字）"
}}"""
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that outputs raw JSON content."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 1024
    }
    
    for attempt in range(1, 4):
        try:
            print(f"  正在请求 API (尝试 {attempt}/3)...")
            resp = requests.post(LLM_API_URL, json=payload, timeout=60)
            resp.raise_for_status()
            
            try:
                response_data = resp.json()
                content = response_data['choices'][0]['message']['content'].strip()
            except (ValueError, KeyError, IndexError) as err:
                print(f"  ⚠️ 尝试 {attempt} 接口响应数据结构解析失败: {err}")
                continue
                
            try:
                qa = extract_json(content)
                if "question" in qa and "ground_truth" in qa:
                    return {
                        "id": idx,
                        "source_context": chunk,
                        "question": qa["question"],
                        "ground_truth": qa["ground_truth"].strip()
                    }
                else:
                    print(f"  ⚠️ 尝试 {attempt} JSON 缺少必要的键: {qa}")
            except Exception as e:
                print(f"  ⚠️ 尝试 {attempt} JSON 解析或过滤失败: {e}. 原始内容摘要: {content[:200]}...")
                
        except requests.exceptions.RequestException as req_err:
            print(f"  ⚠️ 尝试 {attempt} 网络或 API 请求错误: {req_err}")
            
    return None

def main():
    txt_path = "tests/temp_data/三国演义白话文_disguised.txt"
    if not os.path.exists(txt_path):
        print(f"⚠️ {txt_path} 不存在，正在生成...")
        from tests.disguise_book_generator import run_disguise_pipeline
        input_book = "E:/project/pyltp-books-master/pyltp-books-master/mybooks/Book/三国演义白话文"
        run_disguise_pipeline(input_book, "tests/temp_data")

    try:
        print(f"📂 Loading disguised book from {txt_path}...")
        chunks = get_semantic_paragraph_chunks(txt_path, target_length=800)
        print(f"📄 Generated {len(chunks)} chunks from disguised book.")
    except Exception as err:
        print(f"❌ 运行中断: {err}")
        sys.exit(1)
    
    # Set seed for reproducible sampling
    random.seed(42)
    random.shuffle(chunks)
    print(f"🎲 Sampled {len(chunks)} chunks for question generation.")
    
    dataset = []
    idx = 1
    for chunk in chunks:
        if len(dataset) >= 10:
            break
        print(f"🤖 Generating Q&A pair [{len(dataset)+1}/10]...")
        qa_pair = generate_qa_pair(chunk, idx)
        if qa_pair:
            q_text = qa_pair.get("question", "")
            gt_text = qa_pair.get("ground_truth", "")
            
            # 1. 严格过滤：问题中不得包含任何形式代号
            if "[角色_" in q_text or "角色_" in q_text or "[地点_" in q_text or "地点_" in q_text:
                print(f"  ⚠️ 生成的问题中包含伪装代号 '{q_text}'，废弃。")
                continue
                
            # 2. 严格过滤：问题和答案中不得包含任何真实三国演义人名或地名
            has_blacklist = False
            for black_word in BLACKLIST_ENTITIES:
                if black_word in q_text or black_word in gt_text:
                    print(f"  ⚠️ 问题或答案中包含黑名单词汇 '{black_word}'，废弃。")
                    has_blacklist = True
                    break
            if has_blacklist:
                continue
                
            # 3. 严格过滤：答案必须是严格代号匹配，例如 `[角色_24]` 或 `[地点_5]`
            if not re.match(r'^\[(角色|地点)_\d+\]$', gt_text):
                print(f"  ⚠️ 答案格式不合规 (必须仅为 [角色_X] 或 [地点_Y]): '{gt_text}'，废弃。")
                continue
                
            dataset.append(qa_pair)
            print(f"  ✅ Generated successfully: Q: {qa_pair['question'][:50]}... A: {gt_text}")
            idx += 1
        else:
            print(f"  ❌ Failed to generate Q&A for chunk")
            
    # Write to dataset file
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"🎉 Process completed. Saved {len(dataset)} Q&A pairs to {OUTPUT_PATH}")

if __name__ == '__main__':
    main()

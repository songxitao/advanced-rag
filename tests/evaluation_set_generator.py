import os
import sys
import json
import random
import re
import requests
from docx import Document

# Reconfigure stdout/stderr for Windows UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


DOC_CHINESE = "E:/desktop/code/New folder/paper song.docx"
DOC_ENGLISH = "E:/project/DeepSeek-OCR/ocr_results/44221625_LI LEI/44221625_LI LEI_merged.docx"
OUTPUT_PATH = "tests/test_dataset.json"
LLM_API_URL = "http://localhost:8080/v1/chat/completions"
MODEL_NAME = "qwen3.6-35b-a3b-distilled-think"

def get_semantic_paragraph_chunks(doc_path, target_length=1000):
    doc = Document(doc_path)
    chunks = []
    current_chunk = []
    current_length = 0
    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue
        current_chunk.append(text)
        current_length += len(text)
        if current_length >= target_length:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_length = 0
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    return chunks

def extract_json(text: str):
    """
    Safely extract and parse JSON object from LLM output.
    Supports stripping <think> tags and markdown code blocks.
    """
    # Strip <think>...</think> tags and content
    text = re.sub(r'<think>[\s\S]*?</think>', '', text).strip()
    
    # Try to match ```json ... ``` or ``` ... ```
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if match:
        text = match.group(1).strip()
        
    # Extract the longest possible JSON object starting with { and ending with }
    match_braces = re.search(r'(\{[\s\S]*\})', text)
    if match_braces:
        text = match_braces.group(1).strip()
        
    return json.loads(text)

def generate_qa_pair(chunk, idx):
    prompt = f"""你是一个严谨的学术评测出题官。请阅读以下从论文中提取的文本片段（可能是中文或英文），为其设计一个具体的技术性问题，并给出该问题在原文中能够直接印证的标准答案（Ground Truth）。

【语言要求】：
如果原文是英文，请用英文出题和给出答案；如果是中文，请用中文出题和给出答案。

【限制要求】：
1. 问题必须针对文本中的核心技术细节、公式或实验结论，切忌泛泛而谈。
2. 标准答案必须完全忠实于原文，不得夹杂任何外部知识。
3. 请严格按照以下 JSON 格式输出，不要包含任何多余解释或 markdown 标记（如 ```json ... ```）：
{{
  "question": "问题内容",
  "ground_truth": "标准答案内容"
}}

【原文片段】：
{chunk}"""
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that outputs raw JSON content."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1024
    }
    
    for attempt in range(1, 4):
        try:
            print(f"  正在请求 API (尝试 {attempt}/3)...")
            resp = requests.post(LLM_API_URL, json=payload, timeout=60)
            if resp.status_code == 200:
                content = resp.json()['choices'][0]['message']['content'].strip()
                try:
                    qa = extract_json(content)
                    if "question" in qa and "ground_truth" in qa:
                        return {
                            "id": idx,
                            "source_context": chunk,
                            "question": qa["question"],
                            "ground_truth": qa["ground_truth"]
                        }
                    else:
                        print(f"  ⚠️ Attempt {attempt} JSON missing keys: {qa}")
                except Exception as e:
                    print(f"  ⚠️ Attempt {attempt} JSON parse failed: {e}. Raw content: {content[:200]}...")
            else:
                print(f"  ⚠️ Attempt {attempt} failed with HTTP status: {resp.status_code}")
        except Exception as e:
            print(f"  ⚠️ Attempt {attempt} exception: {e}")
    return None

def main():
    print("📂 Loading Chinese document...")
    chunks_cn = get_semantic_paragraph_chunks(DOC_CHINESE)
    print(f"📄 Generated {len(chunks_cn)} chunks from Chinese paper.")
    
    print("📂 Loading English document...")
    chunks_en = get_semantic_paragraph_chunks(DOC_ENGLISH)
    print(f"📄 Generated {len(chunks_en)} chunks from English paper.")
    
    # Set seed for reproducible sampling
    random.seed(42)
    
    # Sample 15 chunks from each
    sampled_cn = random.sample(chunks_cn, min(15, len(chunks_cn)))
    sampled_en = random.sample(chunks_en, min(15, len(chunks_en)))
    sampled_chunks = sampled_cn + sampled_en
    
    # Shuffle the combined list
    random.shuffle(sampled_chunks)
    print(f"🎲 Sampled {len(sampled_chunks)} chunks for question generation (CN: {len(sampled_cn)}, EN: {len(sampled_en)}).")
    
    dataset = []
    for idx, chunk in enumerate(sampled_chunks, 1):
        print(f"🤖 Generating Q&A pair [{idx}/{len(sampled_chunks)}]...")
        qa_pair = generate_qa_pair(chunk, idx)
        if qa_pair:
            dataset.append(qa_pair)
            print(f"  ✅ Generated successfully: {qa_pair['question'][:50]}...")
        else:
            print(f"  ❌ Failed to generate Q&A for chunk {idx}")
            
    # Write to dataset file
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"🎉 Process completed. Saved {len(dataset)} Q&A pairs to {OUTPUT_PATH}")

if __name__ == '__main__':
    main()

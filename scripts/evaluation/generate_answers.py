import os
import sys
import json
import requests

# Reconfigure stdout/stderr for Windows UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

INPUT_PATH = "tests/retrieval_results.json"
OUTPUT_PATH = "tests/answer_results.json"
LLM_API_URL = "http://localhost:8080/v1/chat/completions"
MODEL_NAME = "gemma4-mtp-nothink"

def query_llm_for_answer(question, context):
    """
    调用本地做题小模型 gemma4-mtp-nothink，依据上下文回答问题。
    """
    if not context or not context.strip():
        return "（未检索到相关参考资料，无法回答）"

    prompt = f"""你是一个参加学术考试的学生。请根据以下给定的【参考资料】回答【问题】。
    
【答题要求】：
1. 必须完全根据【参考资料】中提供的事实来回答问题，不要凭空捏造或加入外部知识。
2. 保持回答的专业性与精炼度。
3. 语言一致性：如果参考资料和问题是英文，请用英文回答；如果是中文，请用中文回答。

【参考资料】：
{context}

【问题】：
{question}"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. You must answer the question based only on the provided context."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 1024
    }

    for attempt in range(1, 4):
        try:
            resp = requests.post(LLM_API_URL, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            answer = data["choices"][0]["message"]["content"].strip()
            return answer
        except Exception as e:
            print(f"  ⚠️ 模型请求失败 (尝试 {attempt}/3): {e}")
            if attempt == 3:
                return f"（请求模型回答出错：{str(e)}）"

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sanguo', action='store_true', help='Use Sanguo dataset and 4-track context')
    args, unknown = parser.parse_known_args()
    
    # 默认值
    input_path = "tests/retrieval_results.json"
    output_path = "tests/answer_results.json"
    use_sanguo = args.sanguo or os.path.exists("tests/temp_data/retrieval_sanguo_results.json")
    
    if use_sanguo:
        input_path = "tests/temp_data/retrieval_sanguo_results.json"
        output_path = "tests/temp_data/answer_sanguo_results.json"
        print("💡 [Mode] Running in 4-Track Sanguo Mode...")
    else:
        print("💡 [Mode] Running in 2-Track Default Mode...")

    if not os.path.exists(input_path):
        print(f"❌ 检索结果文件不存在: {input_path}，请先运行 Stage 2 (run_retrieval.py)")
        sys.exit(1)

    print(f"📄 加载检索结果: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"🚀 开始让小模型 {MODEL_NAME} 进行双轨/四轨答题...")
    
    results = []
    total = len(data)
    for idx, item in enumerate(data, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        if use_sanguo:
            naive_context = item["naive_context"]
            traditional_context = item.get("traditional_context", "")
            ppr_context = item.get("ppr_context", "")
            walk_context = item.get("walk_context", "")
            
            print(f"\n📝 [{idx}/{total}] 问题: {question}")
            sys.stdout.flush()
            
            print("  🔍 正在生成 Naive RAG 的答案...")
            sys.stdout.flush()
            naive_answer = query_llm_for_answer(question, naive_context)
            
            print("  🔍 正在生成 Traditional RAG 的答案...")
            sys.stdout.flush()
            traditional_answer = query_llm_for_answer(question, traditional_context)
            
            print("  🔍 正在生成 PPR Graph RAG 的答案...")
            sys.stdout.flush()
            ppr_answer = query_llm_for_answer(question, ppr_context)
            
            print("  🔍 正在生成 Heuristic Walk Graph RAG 的答案...")
            sys.stdout.flush()
            walk_answer = query_llm_for_answer(question, walk_context)
            
            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "naive_context": naive_context,
                "traditional_context": traditional_context,
                "ppr_context": ppr_context,
                "walk_context": walk_context,
                "naive_answer": naive_answer,
                "traditional_answer": traditional_answer,
                "ppr_answer": ppr_answer,
                "walk_answer": walk_answer
            })
        else:
            naive_context = item["naive_context"]
            advanced_context = item["advanced_context"]
            
            print(f"\n📝 [{idx}/{total}] 问题: {question}")
            
            print("  🔍 正在生成 Naive RAG 的答案...")
            naive_answer = query_llm_for_answer(question, naive_context)
            
            print("  🔍 正在生成 Advanced RAG 的答案...")
            advanced_answer = query_llm_for_answer(question, advanced_context)
            
            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "naive_context": naive_context,
                "advanced_context": advanced_context,
                "naive_answer": naive_answer,
                "advanced_answer": advanced_answer
            })

    # 保存答题结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 答题结束！回答已成功保存至: {output_path}")

if __name__ == "__main__":
    main()

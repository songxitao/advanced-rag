import os
import sys
import json
import requests

# Reconfigure stdout/stderr for Windows UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

INPUT_PATH = "tests/answer_results.json"
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
    if not os.path.exists(INPUT_PATH):
        print(f"❌ 答题结果文件不存在: {INPUT_PATH}，无法进行补答。")
        sys.exit(1)

    print(f"📄 加载现有答题结果: {INPUT_PATH}")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    patched_count = 0
    total = len(data)

    print(f"🚀 开始扫描并补答空白题目...")
    for idx, item in enumerate(data, 1):
        question = item["question"]
        naive_context = item["naive_context"]
        advanced_context = item["advanced_context"]
        naive_answer = item.get("naive_answer", "")
        advanced_answer = item.get("advanced_answer", "")

        need_patch_naive = not naive_answer or not naive_answer.strip()
        need_patch_advanced = not advanced_answer or not advanced_answer.strip()

        if need_patch_naive or need_patch_advanced:
            print(f"\n📝 [{idx}/{total}] 发现未作答题: {question}")
            
            if need_patch_naive:
                print("  🔍 正在重试并补答 Naive RAG 的空白答案...")
                new_naive = query_llm_for_answer(question, naive_context)
                item["naive_answer"] = new_naive
                print(f"  ✅ 补齐 Naive: {new_naive[:40]}...")
                patched_count += 1

            if need_patch_advanced:
                print("  🔍 正在重试并补答 Advanced RAG 的空白答案...")
                new_adv = query_llm_for_answer(question, advanced_context)
                item["advanced_answer"] = new_adv
                print(f"  ✅ 补齐 Advanced: {new_adv[:40]}...")
                patched_count += 1

    if patched_count > 0:
        # 保存覆盖
        with open(INPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n🎉 成功补齐了 {patched_count} 个空白答案，并已更新覆盖: {INPUT_PATH}")
    else:
        print("\n✨ 扫描完成：未发现任何空白或未作答的题目！")

if __name__ == "__main__":
    main()

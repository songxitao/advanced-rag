import os
import sys
import json
import requests
import matplotlib.pyplot as plt
import numpy as np

# Reconfigure stdout/stderr for Windows UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

INPUT_PATH = "tests/answer_results.json"
RADAR_OUTPUT_PATH = "tests/evaluation_radar.png"
LLM_API_URL = "http://localhost:8080/v1/chat/completions"
MODEL_NAME = "qwen3.6-35b-a3b-nothink"

# 确保输出目录存在
os.makedirs("tests/outputs", exist_ok=True)

def evaluate_with_ragas(data):
    """
    尝试使用 Ragas 进行打分。如果环境缺少依赖或网络不通，则返回 None。
    """
    try:
        print("🔄 尝试初始化 Ragas 评估系统...")
        from langchain_openai import ChatOpenAI
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevance
        from datasets import Dataset

        # 本地裁判模型
        evaluator_llm = ChatOpenAI(
            model=MODEL_NAME,
            api_key="ignored",
            base_url="http://localhost:8080/v1",
            temperature=0.0
        )
        
        # 覆盖打分使用的 LLM 避免拉取外网
        faithfulness.llm = evaluator_llm
        answer_relevance.llm = evaluator_llm

        # 构造数据集
        naive_samples = {
            "question": [item["question"] for item in data],
            "contexts": [[item["naive_context"]] for item in data],
            "answer": [item["naive_answer"] for item in data],
            "ground_truth": [item["ground_truth"] for item in data]
        }
        adv_samples = {
            "question": [item["question"] for item in data],
            "contexts": [[item["advanced_context"]] for item in data],
            "answer": [item["advanced_answer"] for item in data],
            "ground_truth": [item["ground_truth"] for item in data]
        }

        naive_ds = Dataset.from_dict(naive_samples)
        adv_ds = Dataset.from_dict(adv_samples)

        print("📊 正在运行 Ragas 评估 (Naive RAG)...")
        naive_result = evaluate(naive_ds, metrics=[faithfulness, answer_relevance])
        
        print("📊 正在运行 Ragas 评估 (Advanced RAG)...")
        adv_result = evaluate(adv_ds, metrics=[faithfulness, answer_relevance])

        scores = {
            "Naive RAG": {
                "Faithfulness": float(naive_result.get("faithfulness", 0.0)),
                "Answer Relevance": float(naive_result.get("answer_relevance", 0.0)),
                "Context Relevance": float(naive_result.get("context_precision", 0.0))  # 默认降级或用 Ragas 其他指标
            },
            "Advanced RAG": {
                "Faithfulness": float(adv_result.get("faithfulness", 0.0)),
                "Answer Relevance": float(adv_result.get("answer_relevance", 0.0)),
                "Context Relevance": float(adv_result.get("context_precision", 0.0))
            }
        }
        return scores
    except Exception as e:
        print(f"⚠️ Ragas 评估系统不可用或报错 ({e})。将无缝启用本地 LLM-as-a-Judge 备份评分逻辑。")
        return None

def query_judge_score(question, context, ground_truth, answer):
    """
    调用本地 Qwen 裁判模型对单条回答进行打分 (0-10分分值)
    """
    prompt = f"""你是一个苛刻的学术论文评测裁判官。你需要根据给定的【参考资料】和【标准答案】，对考生给出的【考生答案】进行量化打分。

【评分标准】：
1. 忠实度 (Faithfulness) [满分 10分]：考生答案中包含的所有陈述是否都能在【参考资料】中找到依据？如果出现捏造、幻觉或使用外部知识，扣分。
2. 答案相关性 (Answer Relevance) [满分 10分]：考生的回答是否切中【问题】？是否精炼，有没有答非所问或废话？
3. 内容精确度 (Accuracy) [满分 10分]：考生的回答与【标准答案】在核心技术细节、公式或实验数据上是否完全吻合？

请以标准的 JSON 格式输出打分结果，不要包含任何 markdown 标记或多余解释。格式如下：
{{
  "faithfulness": 8.5,
  "relevance": 9.0,
  "accuracy": 8.0
}}

【评测细节】：
- 问题: {question}
- 参考资料: {context}
- 标准答案: {ground_truth}
- 考生答案: {answer}"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a precise evaluation assistant that outputs raw JSON scores."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0
    }

    for attempt in range(1, 4):
        try:
            resp = requests.post(LLM_API_URL, json=payload, timeout=60)
            resp.raise_for_status()
            raw_text = resp.json()["choices"][0]["message"]["content"].strip()
            
            # 清洗 think 标签及 markdown
            raw_text = re_clean_json(raw_text)
            scores = json.loads(raw_text)
            
            # 缩放到 0-1 范围，方便雷达图绘制
            return {
                "Faithfulness": float(scores.get("faithfulness", 0.0)) / 10.0,
                "Answer Relevance": float(scores.get("relevance", 0.0)) / 10.0,
                "Accuracy": float(scores.get("accuracy", 0.0)) / 10.0
            }
        except Exception as e:
            print(f"  ⚠️ 裁判打分失败 (尝试 {attempt}/3): {e}")
            if attempt == 3:
                # 失败退水默认分
                return {"Faithfulness": 0.5, "Answer Relevance": 0.5, "Accuracy": 0.5}

def re_clean_json(text):
    import re
    text = re.sub(r'<think>[\s\S]*?(?:</think>|$)', '', text).strip()
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if match:
        text = match.group(1).strip()
    match_braces = re.search(r'(\{[\s\S]*\})', text)
    if match_braces:
        text = match_braces.group(1).strip()
    text = re.sub(r',\s*([\]}])', r'\1', text)
    return text

def evaluate_with_local_judge(data, use_sanguo=False):
    """
    备份的 LLM 裁判批量评分系统
    """
    print(f"🚀 正在激活本地 Qwen 裁判对 {len(data)} 道题进行评分...")
    if use_sanguo:
        scores = {
            "Naive RAG": {"Faithfulness": 0.0, "Answer Relevance": 0.0, "Accuracy": 0.0},
            "Traditional RAG": {"Faithfulness": 0.0, "Answer Relevance": 0.0, "Accuracy": 0.0},
            "PPR Graph RAG": {"Faithfulness": 0.0, "Answer Relevance": 0.0, "Accuracy": 0.0},
            "Heuristic Walk Graph RAG": {"Faithfulness": 0.0, "Answer Relevance": 0.0, "Accuracy": 0.0}
        }
        total = len(data)
        for idx, item in enumerate(data, 1):
            print(f"  👨‍⚖️ 正在评判第 [{idx}/{total}] 题...")
            sys.stdout.flush()
            
            # Naive RAG
            s = query_judge_score(item["question"], item["naive_context"], item["ground_truth"], item["naive_answer"])
            for k, v in s.items(): scores["Naive RAG"][k] += v
            
            # Traditional RAG
            s = query_judge_score(item["question"], item.get("traditional_context", ""), item["ground_truth"], item.get("traditional_answer", ""))
            for k, v in s.items(): scores["Traditional RAG"][k] += v
            
            # PPR Graph RAG
            s = query_judge_score(item["question"], item.get("ppr_context", ""), item["ground_truth"], item.get("ppr_answer", ""))
            for k, v in s.items(): scores["PPR Graph RAG"][k] += v
            
            # Heuristic Walk Graph RAG
            s = query_judge_score(item["question"], item.get("walk_context", ""), item["ground_truth"], item.get("walk_answer", ""))
            for k, v in s.items(): scores["Heuristic Walk Graph RAG"][k] += v
            
        # 求平均分
        for m in scores:
            for k in scores[m]:
                scores[m][k] = round(scores[m][k] / total, 3)
        return scores
    else:
        naive_scores = {"Faithfulness": 0.0, "Answer Relevance": 0.0, "Accuracy": 0.0}
        adv_scores = {"Faithfulness": 0.0, "Answer Relevance": 0.0, "Accuracy": 0.0}
        total = len(data)
        for idx, item in enumerate(data, 1):
            print(f"  👨‍⚖️ 正在评判第 [{idx}/{total}] 题...")
            sys.stdout.flush()
            n_score = query_judge_score(
                item["question"], item["naive_context"], item["ground_truth"], item["naive_answer"]
            )
            for k, v in n_score.items():
                naive_scores[k] += v
            a_score = query_judge_score(
                item["question"], item["advanced_context"], item["ground_truth"], item["advanced_answer"]
            )
            for k, v in a_score.items():
                adv_scores[k] += v
        for k in naive_scores:
            naive_scores[k] = round(naive_scores[k] / total, 3)
            adv_scores[k] = round(adv_scores[k] / total, 3)
        return {
            "Naive RAG": naive_scores,
            "Advanced RAG": adv_scores
        }

def draw_radar_chart(scores, use_sanguo=False):
    """
    根据打分绘制雷达图
    """
    first_key = list(scores.keys())[0]
    labels = list(scores[first_key].keys())
    num_vars = len(labels)

    # 极坐标角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    
    # 启用中文支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    colors = {
        "Naive RAG": '#FF5722',
        "Advanced RAG": '#4CAF50',
        "Traditional RAG": '#FFC107',
        "PPR Graph RAG": '#00BCD4',
        "Heuristic Walk Graph RAG": '#4CAF50'
    }

    for name, vals_dict in scores.items():
        vals = list(vals_dict.values())
        vals += vals[:1]
        color = colors.get(name, '#9C27B0')
        ax.plot(angles, vals, color=color, linewidth=2, label=name)
        ax.fill(angles, vals, color=color, alpha=0.1)

    # 设置刻度标签
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    
    # 设置径向网格线 (RAG 评分 0 到 1.0)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
    plt.ylim(0, 1.0)

    title = "RAG 各变体量化评测结果 (消融实验)" if use_sanguo else "Naive RAG vs Advanced RAG 量化评测结果"
    plt.title(title, size=14, color='#333333', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.tight_layout()
    plt.savefig(RADAR_OUTPUT_PATH, dpi=150)
    plt.savefig("tests/outputs/evaluation_radar.png", dpi=150)
    print(f"🎨 雷达图已保存至: {RADAR_OUTPUT_PATH} (与 tests/outputs/evaluation_radar.png)")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sanguo', action='store_true', help='Use Sanguo dataset and 4-track context')
    args, unknown = parser.parse_known_args()
    
    input_path = "tests/answer_results.json"
    use_sanguo = args.sanguo or os.path.exists("tests/temp_data/answer_sanguo_results.json")
    
    if use_sanguo:
        input_path = "tests/temp_data/answer_sanguo_results.json"
        print("💡 [Evaluation Mode] Running in 4-Track Sanguo Mode...")
    else:
        print("💡 [Evaluation Mode] Running in 2-Track Default Mode...")

    if not os.path.exists(input_path):
        print(f"❌ 答题结果文件不存在: {input_path}，请先运行 Stage 3 (generate_answers.py)")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. 尝试使用 Ragas 评估 (如果有 Sanguo 数据集，因为结构不同直接降级使用本地 Qwen 裁判)
    scores = None
    if not use_sanguo:
        scores = evaluate_with_ragas(data)

    # 2. 如果不可用或使用 Sanguo，则回退到本地 Qwen 裁判
    if scores is None:
        scores = evaluate_with_local_judge(data, use_sanguo=use_sanguo)

    print("\n" + "="*40)
    print("        🏆 评测结果报告 🏆")
    print("="*40)
    for model_name, sub_scores in scores.items():
        print(f" {model_name} 平均得分:")
        for k, v in sub_scores.items():
            print(f"  - {k}: {v * 10.0:.1f} / 10.0")
        print("-"*40)
    print("="*40)

    # 保存最终得分到 json
    scores_out_path = "tests/temp_data/evaluation_sanguo_scores.json" if use_sanguo else "tests/evaluation_scores.json"
    with open(scores_out_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    # 3. 绘制雷达图
    draw_radar_chart(scores, use_sanguo=use_sanguo)
    print("\n🎉 评测全部完成！您可以查看 tests/evaluation_radar.png 对比量化图形。")

if __name__ == "__main__":
    main()

# RAG Double-Track Quantitative Evaluation Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a rock-solid, segmented offline pipeline to quantitatively evaluate Naive RAG vs Advanced RAG using local LLMs (Qwen 35B for generating/judging and Gemma 12B for answering), outputting a comparative radar chart.

**Architecture:** A step-by-step CLI pipeline manager that decouples: 
1. Neutral test dataset generation from a raw docx.
2. Context retrieval using Naive RAG (local module) and Advanced RAG (local coordinator).
3. Answer generation using Gemma 12B via Llama-Server.
4. Metric scoring using Ragas with Qwen 35B as the judge, culminating in a radar chart plot.

**Tech Stack:** Python 3, `ragas`, `matplotlib`, `pandas`, `python-docx` (`docx`), `requests`, `openai`.

---

## 🛠️ Task Breakdown

### Task 0: Environment setup & Dependency installation

**Files:**
- Create: `tests/check_env.py`

- [ ] **Step 1: Write the environment dependency validation script**
  Create `tests/check_env.py` with:
  ```python
  import sys
  
  def check():
      modules = ['ragas', 'matplotlib', 'pandas', 'docx', 'requests', 'openai', 'jieba']
      missing = []
      for m in modules:
          try:
              __import__(m)
              print(f"✅ {m} is available")
          except ImportError:
              print(f"❌ {m} is missing")
              missing.append(m)
      if missing:
          print(f"\nMissing dependencies: {missing}")
          sys.exit(1)
      else:
          print("\nAll dependencies are successfully installed!")
          sys.exit(0)
  
  if __name__ == '__main__':
      check()
  ```

- [ ] **Step 2: Run dependency check to verify failures**
  Run: `cmd /c ""D:\program files\Miniconda\Scripts\activate.bat"" && conda activate translator_env && python tests/check_env.py`
  Expected: Exit code 1 with `ragas` and `matplotlib` and `jieba` missing.

- [ ] **Step 3: Install missing dependencies**
  Run: `cmd /c ""D:\program files\Miniconda\Scripts\activate.bat"" && conda activate translator_env && pip install ragas matplotlib jieba`

- [ ] **Step 4: Run dependency check to verify success**
  Run: `cmd /c ""D:\program files\Miniconda\Scripts\activate.bat"" && conda activate translator_env && python tests/check_env.py`
  Expected: Exit code 0 with `All dependencies are successfully installed!`

- [ ] **Step 5: Commit**
  ```bash
  git add tests/check_env.py
  git commit -m "chore: add dependency check script and install libraries"
  ```

---

### Task 1: Pipeline Control Manager

**Files:**
- Create: `tests/run_pipeline.py`

- [ ] **Step 1: Write failing test for config values**
  Create `tests/test_pipeline_config.py`:
  ```python
  import os
  
  def test_config():
      # Verify that raw document exists
      raw_doc_path = "E:/desktop/code/New folder/paper song.docx"
      assert os.path.exists(raw_doc_path), f"Raw document not found at {raw_doc_path}"
  ```
  Run: `pytest tests/test_pipeline_config.py`
  Expected: PASS (if file exists).

- [ ] **Step 2: Create pipeline manager script**
  Create `tests/run_pipeline.py` with the CLI menu interface:
  ```python
  import sys
  import os
  import subprocess
  
  def print_menu():
      print("=" * 65)
      print("📊 本地 RAG 双轨量化评测控制中心 (Evaluation Pipeline)")
      print("=" * 65)
      print("[1] 阶段一：中立自动出题 (调用 Qwen35B-Think 生成 test_dataset.json)")
      print("[2] 阶段二：双轨检索提取 (数据对齐并生成 retrieval_results.json)")
      print("[3] 阶段三：小模型解答 (调用 gemma4-mtp-nothink 生成 answer_results.json)")
      print("[4] 阶段四：裁判量化评分 (运行 Ragas 并导出雷达图)")
      print("-" * 65)
      print("[A] 顺序运行全部阶段")
      print("[Q] 退出评测控制中心")
      print("=" * 65)
  
  def run_step(step_num):
      scripts = {
          "1": "tests/evaluation_set_generator.py",
          "2": "tests/run_retrieval.py",
          "3": "tests/generate_answers.py",
          "4": "tests/evaluate_results.py"
      }
      script = scripts.get(step_num)
      if not script:
          return
      print(f"\n🚀 正在启动 阶段 {step_num}: {script}...")
      result = subprocess.run([sys.executable, script], shell=True)
      if result.returncode == 0:
          print(f"✅ 阶段 {step_num} 执行成功！")
      else:
          print(f"❌ 阶段 {step_num} 执行失败，退出码: {result.returncode}")
  
  def main():
      while True:
          print_menu()
          choice = input("请输入您的选择 [1-4, A, Q]: ").strip().upper()
          if choice == 'Q':
              print("👋 已退出评测控制中心。")
              break
          elif choice in ['1', '2', '3', '4']:
              run_step(choice)
          elif choice == 'A':
              for step in ['1', '2', '3', '4']:
                  run_step(step)
          else:
              print("⚠️ 无效输入，请重新选择！")
  
  if __name__ == '__main__':
      main()
  ```

- [ ] **Step 3: Test command execution of a single dummy sub-stage**
  Temporarily create a dummy script `tests/evaluation_set_generator.py`:
  ```python
  print("Dummy step 1 active")
  ```
  Run: `python tests/run_pipeline.py` (enter 1, then Q).
  Expected: Menu prints, executing step 1 outputs `Dummy step 1 active` and returns successfully.

- [ ] **Step 4: Clean up dummy script and commit**
  Remove `tests/test_pipeline_config.py`. Keep `tests/run_pipeline.py`.
  ```bash
  git add tests/run_pipeline.py
  git commit -m "feat: add pipeline control manager menu script"
  ```

---

### Task 2: Stage 1: Neutral Test Dataset Generator

**Files:**
- Create: `tests/evaluation_set_generator.py`

- [ ] **Step 1: Write test for docx chunk splitting**
  Create `tests/test_generator_logic.py`:
  ```python
  from docx import Document
  import os
  
  def test_chunking():
      doc_path = "E:/desktop/code/New folder/paper song.docx"
      doc = Document(doc_path)
      full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
      assert len(full_text) > 0
      
      # Test 1000 char split
      chunks = [full_text[i:i+1000] for i in range(0, len(full_text), 1000)]
      assert len(chunks) > 0
      assert len(chunks[0]) <= 1000
  ```
  Run: `pytest tests/test_generator_logic.py`
  Expected: PASS

- [ ] **Step 2: Implement `tests/evaluation_set_generator.py`**
  Write the question generator script that connects to `http://localhost:8080/v1/chat/completions` using the model `qwen3.6-35b-a3b-distilled-think` with system prompt to extract questions and ground truths.
  ```python
  import os
  import json
  import random
  import requests
  from docx import Document
  
  DOC_PATH = "E:/desktop/code/New folder/paper song.docx"
  OUTPUT_PATH = "tests/test_dataset.json"
  LLM_API_URL = "http://localhost:8080/v1/chat/completions"
  MODEL_NAME = "qwen3.6-35b-a3b-distilled-think"
  
  def split_document(doc_path, chunk_size=1000):
      doc = Document(doc_path)
      full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
      chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
      return chunks
  
  def generate_qa_pair(chunk, idx):
      prompt = f"""你是一个严谨的学术评测出题官。请阅读以下从论文中提取的文本片段，为其设计一个具体的技术性问题，并给出该问题在原文中能够直接印证的标准答案（Ground Truth）。
  
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
      
      for attempt in range(3):
          try:
              resp = requests.post(LLM_API_URL, json=payload, timeout=60)
              if resp.status_code == 200:
                  content = resp.json()['choices'][0]['message']['content'].strip()
                  # Clean potential markdown block formatting
                  if content.startswith("```"):
                      content = content.replace("```json", "").replace("```", "").strip()
                  qa = json.loads(content)
                  if "question" in qa and "ground_truth" in qa:
                      return {
                          "id": idx,
                          "source_context": chunk,
                          "question": qa["question"],
                          "ground_truth": qa["ground_truth"]
                      }
              print(f"⚠️ Chunk {idx} generation failed (Attempt {attempt+1}): HTTP {resp.status_code}")
          except Exception as e:
              print(f"⚠️ Chunk {idx} exception (Attempt {attempt+1}): {e}")
      return None
  
  def main():
      print("📂 Loading document...")
      chunks = split_document(DOC_PATH)
      print(f"📄 Generated {len(chunks)} chunks of size 1000.")
      
      # Sample up to 30 chunks randomly
      sampled_chunks = random.sample(chunks, min(30, len(chunks)))
      print(f"🎲 Sampled {len(sampled_chunks)} chunks for question generation.")
      
      dataset = []
      for idx, chunk in enumerate(sampled_chunks, 1):
          print(f"🤖 Generating Q&A pair [{idx}/{len(sampled_chunks)}]...")
          qa_pair = generate_qa_pair(chunk, idx)
          if qa_pair:
              dataset.append(qa_pair)
          else:
              print(f"❌ Failed to generate Q&A for chunk {idx}")
              
      with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
          json.dump(dataset, f, ensure_ascii=False, indent=2)
      print(f"✅ Saved {len(dataset)} Q&A pairs to {OUTPUT_PATH}")
  
  if __name__ == '__main__':
      main()
  ```

- [ ] **Step 3: Run generator logic to output dataset**
  Run: `cmd /c ""D:\program files\Miniconda\Scripts\activate.bat"" && conda activate translator_env && python tests/evaluation_set_generator.py`
  Expected: JSON file generated at `tests/test_dataset.json` containing the questions.

- [ ] **Step 4: Clean up test files and commit**
  Remove `tests/test_generator_logic.py`.
  ```bash
  git add tests/evaluation_set_generator.py
  git commit -m "feat: add test dataset Q&A generator script"
  ```

---

### Task 3: Stage 2: RAG Context Retrieval

**Files:**
- Create: `tests/run_retrieval.py`

- [ ] **Step 1: Write retrieval test to check integration import**
  Create `tests/test_retrieval_imports.py`:
  ```python
  import sys
  import os
  
  def test_imports():
      # Add naive RAG path
      sys.path.insert(0, "E:/project/rag")
      from rag_engine import RAGEngine
      engine = RAGEngine()
      assert engine is not None
      
      # Add advanced RAG path
      from src.coordinator import RAGCoordinator
      assert RAGCoordinator is not None
  ```
  Run: `pytest tests/test_retrieval_imports.py`
  Expected: PASS

- [ ] **Step 2: Implement RAG Context Retrieval script**
  Create `tests/run_retrieval.py` to check document alignment in Naive RAG and retrieve from both systems:
  ```python
  import os
  import sys
  import json
  
  # Inject old RAG system path
  sys.path.insert(0, "E:/project/rag")
  from rag_engine import RAGEngine
  
  from src.loader import DocumentLoader
  from src.splitter import SemanticParentChildSplitter
  from src.embedding import LocalEmbeddingService
  from src.database import ChromaAdapter
  from src.reranker import RerankerService
  from src.coordinator import RAGCoordinator
  
  DOC_PATH = "E:/desktop/code/New folder/paper song.docx"
  DATASET_PATH = "tests/test_dataset.json"
  OUTPUT_PATH = "tests/retrieval_results.json"
  
  def align_naive_rag():
      print("🔄 Aligning Naive RAG database...")
      engine = RAGEngine()
      stats = engine.get_stats()
      doc_count = stats.get("总文档块数", 0)
      
      # Check if paper song.docx is already indexed
      coll = engine.collection
      metadatas = coll.get()['metadatas']
      indexed_files = set(m.get('filename') for m in metadatas if m)
      
      if "paper song.docx" not in indexed_files:
          print("📥 paper song.docx is missing from Naive RAG. Indexing it now...")
          engine.add_file(DOC_PATH)
          print("✅ Indexed successfully in Naive RAG.")
      else:
          print("✅ paper song.docx is already indexed in Naive RAG.")
      return engine
  
  def get_advanced_coordinator():
      print("🔄 Initializing Advanced RAG coordinator...")
      db_dir = "./vector_db"
      loader = DocumentLoader()
      # Run on CPU to save VRAM
      embedding_service = LocalEmbeddingService(device="cpu")
      splitter = SemanticParentChildSplitter(embedding_service=embedding_service, threshold=None, child_size=150)
      db_adapter = ChromaAdapter(db_dir=db_dir)
      reranker_service = RerankerService(device="cpu")
  
      return RAGCoordinator(
          loader=loader,
          splitter=splitter,
          embedding_service=embedding_service,
          db_adapter=db_adapter,
          reranker=reranker_service
      )
  
  def main():
      if not os.path.exists(DATASET_PATH):
          print(f"❌ Test dataset not found at {DATASET_PATH}. Please run stage 1 first.")
          sys.exit(1)
          
      with open(DATASET_PATH, 'r', encoding='utf-8') as f:
          dataset = json.load(f)
          
      # 1. Align databases
      naive_engine = align_naive_rag()
      advanced_coord = get_advanced_coordinator()
      
      # 2. Retrieve contexts
      results = []
      for idx, item in enumerate(dataset, 1):
          q = item["question"]
          print(f"🔍 Retrieving context for question [{idx}/{len(dataset)}]: {q[:30]}...")
          
          # Naive RAG context (Top 5 reranked context)
          naive_context = naive_engine.search_with_context(q, top_k=5)
          
          # Advanced RAG context (Top 5 parent-child + reranked context)
          advanced_context = advanced_coord.query(q)
          
          results.append({
              "id": item["id"],
              "question": q,
              "ground_truth": item["ground_truth"],
              "naive_context": naive_context,
              "advanced_context": advanced_context
          })
          
      with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
          json.dump(results, f, ensure_ascii=False, indent=2)
      print(f"✅ Saved retrieval results of {len(results)} queries to {OUTPUT_PATH}")
  
  if __name__ == '__main__':
      main()
  ```

- [ ] **Step 3: Run dual-RAG retrieval**
  Run: `cmd /c ""D:\program files\Miniconda\Scripts\activate.bat"" && conda activate translator_env && python tests/run_retrieval.py`
  Expected: Prints RAG logs, performs context retrieval, outputs `tests/retrieval_results.json`.

- [ ] **Step 4: Clean up test files and commit**
  Remove `tests/test_retrieval_imports.py`.
  ```bash
  git add tests/run_retrieval.py
  git commit -m "feat: add RAG retrieval and data alignment script"
  ```

---

### Task 4: Stage 3: LLM Answering

**Files:**
- Create: `tests/generate_answers.py`

- [ ] **Step 1: Write test to verify Gemma completions**
  Create `tests/test_gemma_api.py`:
  ```python
  import requests
  
  def test_gemma():
      payload = {
          "model": "gemma4-mtp-nothink",
          "messages": [{"role": "user", "content": "Ping"}],
          "max_tokens": 10
      }
      resp = requests.post("http://localhost:8080/v1/chat/completions", json=payload)
      assert resp.status_code == 200
  ```
  Run: `pytest tests/test_gemma_api.py`
  Expected: PASS

- [ ] **Step 2: Implement LLM Answering script**
  Create `tests/generate_answers.py` to query Gemma 12B on Llama-Server:
  ```python
  import os
  import sys
  import json
  import requests
  
  INPUT_PATH = "tests/retrieval_results.json"
  OUTPUT_PATH = "tests/answer_results.json"
  LLM_API_URL = "http://localhost:8080/v1/chat/completions"
  MODEL_NAME = "gemma4-mtp-nothink"
  
  def ask_gemma(context, question):
      prompt = f"""请你扮演一个专业的答题助手。请结合我提供的【参考资料】，准确回答【问题】。
  
  【参考资料】：
  {context}
  
  【问题】：
  {question}
  
  【答题要求】：
  1. 必须优先使用【参考资料】中的事实进行回答，答案需要准确、精炼。
  2. 若资料中未提及相关信息，请直接回答：“参考资料中未提及相关信息，无法回答。”"""
  
      payload = {
          "model": MODEL_NAME,
          "messages": [
              {"role": "user", "content": prompt}
          ],
          "temperature": 0.7,
          "max_tokens": 512
      }
      
      try:
          resp = requests.post(LLM_API_URL, json=payload, timeout=60)
          if resp.status_code == 200:
              return resp.json()['choices'][0]['message']['content'].strip()
          else:
              print(f"⚠️ Gemma request failed: HTTP {resp.status_code}")
      except Exception as e:
          print(f"⚠️ Gemma request exception: {e}")
      return "（生成答案失败）"
  
  def main():
      if not os.path.exists(INPUT_PATH):
          print(f"❌ Input retrieval file not found at {INPUT_PATH}. Please run stage 2 first.")
          sys.exit(1)
          
      with open(INPUT_PATH, 'r', encoding='utf-8') as f:
          data = json.load(f)
          
      results = []
      for idx, item in enumerate(data, 1):
          print(f"🤖 Solver generating answers [{idx}/{len(data)}]...")
          naive_ans = ask_gemma(item["naive_context"], item["question"])
          adv_ans = ask_gemma(item["advanced_context"], item["question"])
          
          results.append({
              "id": item["id"],
              "question": item["question"],
              "ground_truth": item["ground_truth"],
              "naive_context": item["naive_context"],
              "naive_answer": naive_ans,
              "advanced_context": item["advanced_context"],
              "advanced_answer": adv_ans
          })
          
      with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
          json.dump(results, f, ensure_ascii=False, indent=2)
      print(f"✅ Saved solver answers to {OUTPUT_PATH}")
  
  if __name__ == '__main__':
      main()
  ```

- [ ] **Step 3: Run answers generation**
  Run: `cmd /c ""D:\program files\Miniconda\Scripts\activate.bat"" && conda activate translator_env && python tests/generate_answers.py`
  Expected: Prints response generation, creates `tests/answer_results.json`.

- [ ] **Step 4: Clean up test files and commit**
  Remove `tests/test_gemma_api.py`.
  ```bash
  git add tests/generate_answers.py
  git commit -m "feat: add solver answering generation script using Gemma"
  ```

---

### Task 5: Stage 4: Ragas Scoring & Radar Chart Visualization

**Files:**
- Create: `tests/evaluate_results.py`

- [ ] **Step 1: Write test to verify matplotlib configuration**
  Create `tests/test_matplotlib.py`:
  ```python
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  
  def test_plot():
      fig, ax = plt.subplots()
      ax.plot([1, 2], [3, 4])
      fig.savefig("tests/test_plot.png")
      import os
      assert os.path.exists("tests/test_plot.png")
      os.remove("tests/test_plot.png")
  ```
  Run: `pytest tests/test_matplotlib.py`
  Expected: PASS

- [ ] **Step 2: Implement scoring and visualization script**
  Create `tests/evaluate_results.py` using Ragas API and local Qwen evaluator:
  ```python
  import os
  import sys
  import json
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  
  # Configure Chinese font for matplotlib
  plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
  plt.rcParams['axes.unicode_minus'] = False
  
  from datasets import Dataset
  from langchain_openai import ChatOpenAI
  from ragas import evaluate
  from ragas.metrics import faithfulness, answer_relevance, context_recall, context_precision
  
  INPUT_PATH = "tests/answer_results.json"
  SCORE_PATH = "tests/evaluation_scores.json"
  RADAR_PATH = "tests/outputs/evaluation_radar.png"
  
  def main():
      if not os.path.exists(INPUT_PATH):
          print(f"❌ Input file not found at {INPUT_PATH}. Please run stage 3 first.")
          sys.exit(1)
          
      with open(INPUT_PATH, 'r', encoding='utf-8') as f:
          data = json.load(f)
          
      # 1. Format datasets for Ragas
      naive_list = []
      adv_list = []
      
      for item in data:
          # Convert multi-string context format into List[str] as expected by Ragas
          naive_list.append({
              "question": item["question"],
              "contexts": [item["naive_context"]],
              "answer": item["naive_answer"],
              "ground_truth": item["ground_truth"]
          })
          adv_list.append({
              "question": item["question"],
              "contexts": [item["advanced_context"]],
              "answer": item["advanced_answer"],
              "ground_truth": item["ground_truth"]
          })
          
      naive_ds = Dataset.from_list(naive_list)
      adv_ds = Dataset.from_list(adv_list)
      
      # 2. Setup local Qwen judge
      judge_llm = ChatOpenAI(
          model="qwen3.6-35b-a3b-distilled-think",
          base_url="http://localhost:8080/v1",
          api_key="none",
          temperature=0.0
      )
      
      # Bind local LLM to ragas metrics
      metrics = [faithfulness, answer_relevance, context_recall, context_precision]
      for m in metrics:
          m.llm = judge_llm
          
      print("⚖️ Evaluating Naive RAG results using Ragas...")
      naive_results = evaluate(naive_ds, metrics=metrics)
      print("⚖️ Evaluating Advanced RAG results using Ragas...")
      adv_results = evaluate(adv_ds, metrics=metrics)
      
      scores = {
          "naive_rag": dict(naive_results),
          "advanced_rag": dict(adv_results)
      }
      
      # Save scores
      with open(SCORE_PATH, 'w', encoding='utf-8') as f:
          json.dump(scores, f, ensure_ascii=False, indent=2)
      print(f"✅ Saved scores to {SCORE_PATH}")
      
      # 3. Draw Radar Chart
      labels = ['Faithfulness\n(忠实度)', 'Answer Relevance\n(答案相关度)', 
                'Context Recall\n(检索召回率)', 'Context Precision\n(检索精准度)']
      num_vars = len(labels)
      
      naive_vals = [scores["naive_rag"].get(m.name, 0.0) for m in metrics]
      adv_vals = [scores["advanced_rag"].get(m.name, 0.0) for m in metrics]
      
      # Complete the circular loop
      naive_vals += naive_vals[:1]
      adv_vals += adv_vals[:1]
      
      angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
      angles += angles[:1]
      
      fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
      
      # Draw Naive RAG polygon
      ax.plot(angles, naive_vals, color='#FF5733', linewidth=2, label='Naive RAG (对照组)')
      ax.fill(angles, naive_vals, color='#FF5733', alpha=0.25)
      
      # Draw Advanced RAG polygon
      ax.plot(angles, adv_vals, color='#1E8449', linewidth=2, label='Advanced RAG (实验组)')
      ax.fill(angles, adv_vals, color='#1E8449', alpha=0.25)
      
      ax.set_theta_offset(np.pi / 2)
      ax.set_theta_direction(-1)
      
      # Set ticks
      ax.set_thetagrids(np.degrees(angles[:-1]), labels)
      ax.set_ylim(0, 1.0)
      ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], angle=45, color='grey', size=8)
      
      plt.title("Naive RAG vs Advanced RAG 双轨量化评测雷达对比图", y=1.1, fontsize=14, fontweight='bold')
      plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
      
      os.makedirs(os.path.dirname(RADAR_PATH), exist_ok=True)
      plt.savefig(RADAR_PATH, dpi=150, bbox_inches='tight')
      print(f"🎨 Saved radar chart to {RADAR_PATH}")
  
  if __name__ == '__main__':
      main()
  ```

- [ ] **Step 3: Run scoring and chart creation**
  Run: `cmd /c ""D:\program files\Miniconda\Scripts\activate.bat"" && conda activate translator_env && python tests/evaluate_results.py`
  Expected: Prints Ragas evaluation logs, creates radar chart PNG at `tests/outputs/evaluation_radar.png`.

- [ ] **Step 4: Clean up test files and commit**
  Remove `tests/test_matplotlib.py`.
  ```bash
  git add tests/evaluate_results.py
  git commit -m "feat: add Ragas evaluation and radar plotting script"
  ```

# RAG Double-Track Quantitative Evaluation Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a rock-solid, segmented offline pipeline to quantitatively evaluate Naive RAG vs Advanced RAG using local LLMs (Qwen 35B for generating/judging and Gemma 12B for answering), outputting a comparative radar chart, using a dual-document bilingual dataset with paragraph-level semantic splitting.

**Architecture:** A step-by-step CLI pipeline manager that decouples: 
1. Neutral bilingual test dataset generation using natural paragraph-level semantic splitting.
2. Context retrieval using Naive RAG (local module) and Advanced RAG (local coordinator), with automatic dual-document database alignment.
3. Answer generation using Gemma 12B via Llama-Server.
4. Metric scoring using Ragas with Qwen 35B as the judge, culminating in a radar chart plot.

**Tech Stack:** Python 3, `ragas`, `matplotlib`, `pandas`, `python-docx` (`docx`), `requests`, `openai`.

---

## 🛠️ Task Breakdown

### Task 0: Environment setup & Dependency installation (Completed)

---

### Task 1: Pipeline Control Manager (Completed)

---

### Task 2: Stage 1: Neutral Test Dataset Generator

**Files:**
- Create: `tests/evaluation_set_generator.py`

- [ ] **Step 1: Write test for natural paragraph semantic chunk splitting**
  Create `tests/test_generator_logic.py`:
  ```python
  from docx import Document
  import os
  
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
  
  def test_semantic_chunking():
      doc_path = "E:/desktop/code/New folder/paper song.docx"
      chunks = get_semantic_paragraph_chunks(doc_path)
      assert len(chunks) > 0
      # Ensure natural paragraph boundaries are preserved (no mid-sentence chops)
      assert "\n\n" in chunks[0] or len(chunks[0]) > 0
  ```
  Run: `pytest tests/test_generator_logic.py`
  Expected: PASS

- [ ] **Step 2: Implement `tests/evaluation_set_generator.py`**
  Write the question generator script that reads from both the Chinese and English papers, samples 15 chunks from each, and generates Q&As using Qwen 35B.
  ```python
  import os
  import json
  import random
  import requests
  from docx import Document
  
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
  
  def generate_qa_pair(chunk, idx):
      prompt = f"""你是一个严谨的学术评测出题官。请阅读以下从论文中提取的文本片段（可能是中文或英文），为其设计一个具体的技术性问题，并给出该问题在原文中能够直接印证的标准答案（Ground Truth）。如果原文是英文，请用英文出题和给出答案；如果是中文，请用中文出题和给出答案。
  
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
      print("📂 Loading Chinese document...")
      chunks_cn = get_semantic_paragraph_chunks(DOC_CHINESE)
      print(f"📄 Generated {len(chunks_cn)} chunks from Chinese paper.")
      
      print("📂 Loading English document...")
      chunks_en = get_semantic_paragraph_chunks(DOC_ENGLISH)
      print(f"📄 Generated {len(chunks_en)} chunks from English paper.")
      
      # Sample 15 chunks from each randomly
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
  Create `tests/run_retrieval.py` to align both documents in Naive and Advanced databases and retrieve context:
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
  
  DOC_CHINESE = "E:/desktop/code/New folder/paper song.docx"
  DOC_ENGLISH = "E:/project/DeepSeek-OCR/ocr_results/44221625_LI LEI/44221625_LI LEI_merged.docx"
  DATASET_PATH = "tests/test_dataset.json"
  OUTPUT_PATH = "tests/retrieval_results.json"
  
  def align_naive_rag():
      print("🔄 Aligning Naive RAG database...")
      engine = RAGEngine()
      
      # Get currently indexed filenames
      metadatas = engine.collection.get()['metadatas']
      indexed_files = set(m.get('filename') for m in metadatas if m)
      
      for name, path in [("paper song.docx", DOC_CHINESE), ("44221625_LI LEI_merged.docx", DOC_ENGLISH)]:
          if name not in indexed_files:
              print(f"📥 {name} is missing from Naive RAG. Indexing it now...")
              engine.add_file(path)
              print(f"✅ Indexed {name} successfully in Naive RAG.")
          else:
              print(f"✅ {name} is already indexed in Naive RAG.")
      return engine
  
  def align_advanced_rag(coordinator):
      print("🔄 Aligning Advanced RAG database...")
      # Get currently indexed filenames in Advanced RAG
      metadatas = coordinator.db_adapter.collection.get()['metadatas']
      indexed_files = set(m.get('filename') for m in metadatas if m)
      
      for name, path in [("paper song.docx", DOC_CHINESE), ("44221625_LI LEI_merged.docx", DOC_ENGLISH)]:
          if name not in indexed_files:
              print(f"📥 {name} is missing from Advanced RAG. Indexing it now...")
              coordinator.add_file(path)
              print(f"✅ Indexed {name} successfully in Advanced RAG.")
          else:
              print(f"✅ {name} is already indexed in Advanced RAG.")
  
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
      align_advanced_rag(advanced_coord)
      
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
  (No changes needed from previous version of the plan)

- [ ] **Step 2: Implement LLM Answering script**
  (No changes needed from previous version of the plan)

- [ ] **Step 3: Run answers generation**
  (No changes needed from previous version of the plan)

- [ ] **Step 4: Clean up test files and commit**
  (No changes needed from previous version of the plan)

---

### Task 5: Stage 4: Ragas Scoring & Radar Chart Visualization

**Files:**
- Create: `tests/evaluate_results.py`

- [ ] **Step 1: Write test to verify matplotlib configuration**
  (No changes needed from previous version of the plan)

- [ ] **Step 2: Implement scoring and visualization script**
  (No changes needed from previous version of the plan)

- [ ] **Step 3: Run scoring and chart creation**
  (No changes needed from previous version of the plan)

- [ ] **Step 4: Clean up test files and commit**
  (No changes needed from previous version of the plan)

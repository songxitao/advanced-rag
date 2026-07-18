# scratch

> Scratch directory for environment diagnostics, component-level unit tests, and an end-to-end integration demo of the semantic RAG engine.

This directory contains three categories of scripts: (1) health-check tools that verify the runtime environment — installed Python packages, ChromaDB vector stores, and the local LLM router; (2) small isolated tests for individual libraries and fallback behavior (mistune rendering, MinerU PDF-parse degradation); and (3) a full end-to-end demo script (query_thesis_rag.py) that exercises the complete pipeline: document loading, semantic parent-child splitting, embedding, ChromaDB indexing, graph-based retrieval with heuristic walk, and reranking. The simulated splitter test doubles as a unit test for the chunking logic.

## Files

### `scratch/check_dependencies.py`

Verify that all third-party packages required by the project are installed.

- `main` (function) - Iterates over a hardcoded list of module names, imports each, and prints INSTALLED/NOT INSTALLED.

### `scratch/check_db.py`

Inspect the Advanced RAG ChromaDB store (e.g., collection counts, indexed filenames).

- `main` (function) - Connects to PersistentClient at e:/project/advanced-rag/vector_db, lists collections with document counts, and prints the set of indexed filenames from metadata.

### `scratch/check_naive_db.py`

Inspect the Naive RAG ChromaDB store for comparison against the advanced store.

- `main` (function) - Connects to e:/project/rag/vector_db, lists collections, and prints indexed filenames from rag_solo_docs.

### `scratch/check_llm.py`

Confirm the local LLM router (Ollama-compatible) is reachable and list available models.

- `main` (function) - GETs http://localhost:8080/v1/models, prints status and model list.

### `scratch/find_source_path.py`

Debug tool to find which source files a given document (e.g., paper song.docx) was indexed from.

- `main` (function) - Queries advanced_rag_collection metadata for entries whose filename matches the target, prints the set of source_paths.

### `scratch/test_llm_completion.py`

Smoke-test LLM completion API: list models then send a chat completion request.

- `main` (function) - GETs /v1/models, then POSTs a user message to /v1/chat/completions with model qwen3.6-35b-a3b-opus and prints the response.

### `scratch/test_qwen_models.py`

Test two specific Qwen distilled models (nothink vs think variants) against the local router.

- `test_model` (function) - Sends a chat completion request for a given model name and prints the assistant reply.
- `main` (function) - Tests both qwen3.6-35b-a3b-distilled-nothink and distilled-think models sequentially.

### `scratch/test_mineru_fallback.py`

Verify that DocumentLoader degrades gracefully to fitz when MinerU is unavailable.

- `create_temp_pdf` (function) - Creates a minimal temp PDF via PyMuPDF (fitz) for the loader test.
- `main` (function) - Writes a temp PDF, calls DocumentLoader.load() on it, prints parsed text, and cleans up.

### `scratch/test_mistune.py`

Quick sanity check that the mistune Markdown parser and its plugins are importable.

- `__main__` (module-level) - Prints mistune version, dir(mistune), and dir(mistune.plugins).

### `scratch/test_renderer.py`

Standalone test of the AST-to-Markdown rendering pipeline (render_inline + render_ast_node_to_md). Truncated in source.

- `render_inline` (function) - Recursively converts mistune inline AST children back to Markdown text (text, emphasis, strong, codespan, link, image, raw_html).
- `render_ast_node_to_md` (function) - Converts a block-level AST node (heading, paragraph, code block, list, table) to Markdown. Truncated before the table implementation completes.

### `scratch/test_simulated_splitter.py`

Unit test for the chunking logic using a mock embedding service that returns deterministic vectors.

- `MockEmbeddingService` (class) - Fake embedding service returning [1,0,0] for text containing '第一部分' and [0,0,1] otherwise; exposes batch mode.
- `_cosine_similarity` (function) - Standard cosine similarity between two float vectors.
- `_split_parent_to_chunks` (function) - Splits a parent text into child-sized chunks at punctuation boundaries.

### `scratch/query_thesis_rag.py`

Full end-to-end demo: ingest three papers, build the GNN co-occurrence graph, then run heuristic-walk graph-enhanced retrieval with reranking.

- `main` (function) - 1) Initializes RAGCoordinator (loader, splitter, embedding service, ChromaAdapter, reranker) against a fresh db_dir. 2) Indexes three paper .docx files via coordinator.add_file(). 3) Creates GraphPostRetriever and runs three queries with graph_search_mode='heuristic_walk', printing retrieved context per query.

## Key Concepts

- **Semantic parent-child chunking**: Documents are split into large parent chunks and smaller child chunks via SemanticParentChildSplitter; embeddings of both levels enable retrieval at two granularities.
- **Graph-enhanced retrieval (Heuristic Walk)**: Entity co-occurrence edges built during indexing are traversed by GraphPostRetriever using heuristic walk mode, so a query can reach semantically linked passages that keyword search would miss.
- **Two-stage retrieval**: Initial dense-graph retrieval is followed by cross-encoder reranking (RerankerService) to re-order candidates before returning context to the LLM.
- **MinerU fallback degradation**: DocumentLoader attempts MinerU PDF parsing first; on failure it logs a warning and falls back to fitz, ensuring the pipeline stays operational when the external parser is down.
- **Offline-first model loading**: 

## Internal Relationships

- `scratch/query_thesis_rag.py` → `src/loader.py`: query_thesis_rag.py imports DocumentLoader to ingest .docx files.
- `scratch/query_thesis_rag.py` → `src/splitter.py`: Imports SemanticParentChildSplitter for parent-child chunking.
- `scratch/query_thesis_rag.py` → `src/embedding.py`: Imports LocalEmbeddingService for dense embeddings (CUDA-accelerated).
- `scratch/query_thesis_rag.py` → `src/database.py`: Imports ChromaAdapter to persist/retrieve vectors from a persistent ChromaDB store.
- `scratch/query_thesis_rag.py` → `src/reranker.py`: Imports RerankerService for cross-encoder reranking after initial retrieval.
- `scratch/query_thesis_rag.py` → `src/coordinator.py`: Imports RAGCoordinator as the orchestration layer that ties loader→splitter→embedding→db together and builds the entity co-occurrence graph.
- `scratch/query_thesis_rag.py` → `src/graph_search.py`: Imports GraphPostRetriever; its query_graph_enhanced() method performs heuristic-walk graph traversal + reranking.
- `scratch/check_db.py` → `check_naive_db.py`: Mirror scripts — same structure, different DB paths (advanced vs naive) and collection names, used for side-by-side comparison.
- `scratch/test_mistune.py` → `test_renderer.py`: test_mistune.py verifies the parser is importable; test_renderer.py exercises the render_inline/render_ast_node_to_md functions that consume its AST output.
- `scratch/test_simulated_splitter.py` → `test_renderer.py`: Shares the same render_inline and render_ast_node_to_md function definitions (duplicated in both files); test_simulated_splitter.py also defines MockEmbeddingService as a unit-test double.

# Advanced RAG Engine

> A semantic chunking and adaptive reranking retrieval-augmented generation engine with graph search support.

This project implements a local, offline-capable RAG pipeline written in Python. It ingests documents (PDF, Word, Excel, Markdown) by converting them to standard Markdown via MarkItDown, then splits the text into child and parent chunks using a semantic splitter. Chunks are deduplicated with a sliding-window hash, embedded on GPU/CPU, and stored in Chroma.

Retrieval runs two channels in parallel: dense vector search (Chroma) and sparse BM25. Results pass through a CrossEncoder reranker, then an adaptive 'semantic cliff' cutoff trims low-scoring candidates before returning parent chunks as context for a local LLM. A graph-search mode adds PPR-based subgraph retrieval on top of the hybrid pipeline.

The codebase ships with a FastAPI server (src/app.py), a full test suite under tests/, and VitePress docs in docs/.

## Key Features

- Parent-child semantic chunking (150-word child / 800-word parent)
- Hybrid retrieval: dense Chroma + sparse BM25, run concurrently
- CrossEncoder reranking with adaptive semantic-cliff cutoff
- Sliding-window hash dedup before embedding
- GPU/CPU device selection; models loaded from offline HF cache
- Graph-enhanced retrieval (PPR-weighted subgraph) via /retrieve_graph endpoint
- FastAPI server with Swagger UI

## Getting Started

1. conda create -n advanced-rag python=3.12 && conda activate advanced-rag
2. pip install -r requirements.txt
3. conda create -n markitdown-env python=3.12 && conda activate markitdown-env && pip install markitdown[all]
4. Set HF_HOME and SENTENCE_TRANSFORMERS_HOME env vars to a local cache dir, then run: python src/app.py

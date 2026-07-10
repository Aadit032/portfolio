---
title: 'Recall-OS'
description: 'RecallOS is an AI-native enterprise knowledge operating system that allows organizations to ingest, organize, search and reason over every piece of company knowledge.'
pubDate: 2026-07-04
heroImage: '../../assets/recallos_1.png'
---

AI-native knowledge operating system for multimodal enterprise search and long-term organizational memory.

RecallOS ingests documents, images, videos, audio and conversations into a unified knowledge base that supports semantic search, hybrid retrieval, memory, and agentic workflows.

## Features

- 📄 Multimodal ingestion (PDFs, PPTs, Images, Audio, Video)
- 🔍 Hybrid retrieval (SPLADE + dense vector search)
- 🧠 Long-term organizational memory
- 🤖 Agent-powered search and reasoning
- 📚 Source-grounded responses with citations
- ⚡ Distributed ingestion pipeline

### Tech Stack

| Layer            | Technology             |
| ---------------- | ---------------------- |
| Frontend         | Next.js                |
| Backend          | Express                |
| Workers          | Redis Streams          |
| Storage          | MinIO (S3)             |
| Database         | PostgreSQL             |
| Vector DB        | Qdrant                 |
| Sparse Retrieval | SPLADE                 |
| Parsing          | LlamaParse             |
| Audio            | Whisper + CLAP         |
| Vision           | Vision-Language Models |
| LLM              | Provider Agnostic      |


### Architecture
```
User Upload
      │
      ▼
   MinIO (S3)
      │
      ▼
 Redis Streams
      │
      ▼
Distributed Workers
      │
      ▼
 LlamaParse / Whisper / Vision
      │
      ▼
 Semantic Chunking
      │
      ▼
 Chunk Enrichment
      │
      ▼
Dense Embeddings + SPLADE
      │
 ┌────┴─────┐
 ▼          ▼
Qdrant   Sparse Index
      │
      ▼
 PostgreSQL Metadata
```


### Retrieval Pipeline
```
User Query
      │
      ▼
Query Rewriting
      │
 ┌────┴─────┐
 ▼          ▼
Dense Search    SPLADE Search
(Qdrant)        (Sparse)
      │          │
      └────┬─────┘
           ▼
    Reciprocal Rank Fusion
           ▼
      Cross-Encoder
        Reranker
           ▼
        Context
           ▼
           LLM
```

### Multimodal Retrieval

#### Documents
1. Semantic chunking
2. Context enrichment
3. Dense + SPLADE indexing
#### Images
1. Vision-language embeddings
2. OCR + metadata
3. Text ↔ Image retrieval in a shared embedding space

#### Audio
1. Whisper transcription
2. CLAP embeddings for shared audio-text retrieval

#### Video
1. Transcript indexing
2. Keyframe extraction
3. Vision-language embeddings

### Principles

1. Memory-first
2. Source-grounded
3. Agent-native
4. Horizontally scalable
5. Multimodal by design
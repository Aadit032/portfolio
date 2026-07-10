---
title: 'Second brain'
description: 'AI-powered personal knowledge management platform that combines notes into a single workspace, enabling fast, context-aware conversations with your own knowledge.'
pubDate: 2026-02-01
heroImage: '../../assets/sb.png'
---


> An AI-powered knowledge management platform that transforms personal notes and documents into a searchable, context-aware second brain.

Recall centralizes notes, links, and documents into a unified workspace where users can search, retrieve, and chat with their knowledge using large language models and semantic retrieval.

---

## Features

- 🧠 AI-powered Q&A over personal knowledge
- 🔍 Semantic search using vector embeddings
- 📝 Centralized workspace for notes and links
- 💬 Context-aware conversations with LLMs
- 📚 Source-grounded retrieval
- 📊 Application monitoring with Prometheus

---

## Tech Stack

| Layer | Technology |
|--------|------------|
| Frontend | React + TypeScript |
| Backend | Node.js + Express |
| Database | PostgreSQL |
| Vector Database | Qdrant |
| AI | OpenAI / LLM APIs |
| Embeddings | Mistral API |
| Monitoring | Prometheus |

---

## Architecture

```text
            User
              │
              ▼
         React Frontend
              │
              ▼
        Express Backend
              │
      ┌───────┴────────┐
      ▼                ▼
 PostgreSQL        Qdrant
 (Metadata)      (Embeddings)
      │                │
      └───────┬────────┘
              ▼
        Retrieval Pipeline
              │
              ▼
             LLM
              │
              ▼
      Context-Aware Answer
```

---

## Retrieval Pipeline

When a user asks a question:

```text
User Query
      │
      ▼
Generate Embedding
      │
      ▼
Semantic Search
(Qdrant)
      │
      ▼
Retrieve Relevant Notes
      │
      ▼
Build Context
      │
      ▼
Large Language Model
      │
      ▼
Grounded Response
```

By retrieving only the most relevant notes before generation, Recall produces accurate, source-aware responses while significantly reducing latency compared to sending the entire knowledge base to the model.

---

## Monitoring

Application health is continuously monitored using Prometheus.

Collected metrics include:

- API latency
- Request throughput
- Error rates
- Endpoint availability
- System performance

Monitoring spans **15+ API endpoints**, providing visibility into application health and performance.

---

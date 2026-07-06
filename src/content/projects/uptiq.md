---
title: 'Uptiq'
description: 'A distributed uptime monitoring platform that checks website availability across multiple regions, providing real-time status.'
pubDate: 2026-02-01
heroImage: '../../assets/blog-placeholder-5.jpg'
---

- Engineered a distributed uptime monitoring pipeline tracking 20+ websites with sub-second
health checks, recording latency (ms) and availability status.
- Implemented a Redis Streams–based producer–consumer architecture with a pusher and
multiple regional consumer groups for horizontal scalability.
- Processed monitoring events with at-least-once delivery guarantees, ensuring fault-tolerant
execution under failures.
- Built a real-time monitoring dashboard displaying per-website uptime %, current status
(up/down), and historical availability trends, enabling quick incident detection and analysis.
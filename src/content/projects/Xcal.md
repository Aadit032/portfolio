---
title: 'Xcal'
description: 'Realtime Excalidraw'
pubDate: 2026-02-01
heroImage: '../../assets/xcal.png'
---

> A real-time collaborative whiteboard built with the HTML Canvas API, RoughJS, and WebSockets.

Xcal is a browser-based whiteboard that enables multiple users to draw and collaborate on the same canvas in real time. Inspired by Excalidraw, it features hand-drawn rendering, low-latency synchronization, and a custom canvas engine built from scratch.

---

## Features

- ✏️ Freehand drawing with smooth stroke interpolation
- ⬜ Rectangle, Circle, Line, Arrow, and Text tools
- 🎨 Hand-drawn rendering powered by RoughJS
- 🔄 Real-time collaboration using WebSockets
- 👥 Room-based collaborative editing
- ⚡ Optimized rendering pipeline for smooth interactions
- 🖱️ Shape selection, movement, and transformations

---

## Tech Stack

| Layer | Technology |
|--------|------------|
| Frontend | React |
| Rendering | HTML Canvas API |
| Drawing Engine | RoughJS |
| Backend | Node.js + Express |
| Realtime | WebSockets |
| Language | TypeScript |

---

## Architecture

```text
            User A             User B
               │                  │  
               └──────────┬───────┴
                          │                 
                    WebSocket Server
                          │
                 Room-based Event Broadcast
                          │
        ┌─────────────────┴─────────────────┐
        ▼                                   ▼
 Canvas State Sync                  Drawing Events
        │                                   │
        └──────────────┬────────────────────┘
                       ▼
              Canvas Rendering Engine
```

---

## Drawing Engine

The canvas engine is implemented directly on top of the HTML Canvas API.

Supported primitives:

- Rectangle
- Circle
- Line
- Arrow
- Freehand Pencil

Each shape is represented as structured data, allowing the canvas to be fully reconstructed on every redraw while supporting selection, movement, and editing.

---

## Rendering Pipeline

To maintain smooth interactions, the rendering system:

- Redraws only when necessary
- Minimizes expensive canvas operations
- Batches user interactions
- Efficiently replays canvas state

This allows the application to maintain approximately **45 FPS** during collaborative drawing sessions.

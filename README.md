# NeuraLearn

**AI-powered learning path generator** that turns any skill into an interactive knowledge graph with optimised study routes — powered by spectral graph theory, ant colony optimisation, and stochastic diffusion search.

Type a topic. Get a complete curriculum. See how everything connects.

![Python](https://img.shields.io/badge/Python-3.14-blue)
![React](https://img.shields.io/badge/React-19-61DAFB)
![TypeScript](https://img.shields.io/badge/TypeScript-5.7-3178C6)
![Tests](https://img.shields.io/badge/Tests-5_suites_(218_asserts)-brightgreen)

---

## What It Does

1. **You type a skill** — "Machine Learning", "Piano", "Web Development", anything
2. **NeuraLearn scrapes the web** concurrently (Wikipedia, GeeksforGeeks, GitHub, DuckDuckGo) to discover subtopics and resources
3. **Builds a knowledge graph** — a DAG where nodes are topics and edges are prerequisite relationships, with real fan-in/fan-out structure
4. **Finds the best learning path** using Ant Colony Optimisation with spectral heuristics — not just any valid order, but the one that minimises cognitive load
5. **Visualises everything** as an interactive graph you can explore, with multiple path options, progress tracking, gamification, and study tools

---

## Features

### Core

- **Knowledge Graph Generation** — webscrape → DAG → spectral layout → ACO-optimised paths, all in < 3 seconds
- **Multiple Learning Paths** — prerequisite-respecting routes at beginner / intermediate / advanced difficulty
- **Sub-Graph Deep Dives** — click any topic node to generate a detailed breakdown of that sub-skill
- **Server-Side Mastery Tracking** — prerequisite-validated completion with cascade unmastering
- **Shortest Path Finder** — find the minimum-topic route to reach any target node
- **Live Spell Correction** — SDS-powered real-time spell-check as you type on the home page
- **GAT Difficulty Analysis** — Graph Attention Network predicts per-topic difficulty based on graph structure and mastery state, powering smart "next topic" recommendations

### University Student Features

- **Pomodoro Timer** — 25/5/15-minute focus/break modes with SVG progress ring, auto-mode-switch, and session counter
- **Gamification System** — XP (100 per topic, 25 per Pomodoro), quadratic levelling, 8 achievements, streak tracking
- **Anki Flashcard Export** — one-click TSV download of all topics for spaced repetition
- **Study Time Estimation** — per-topic minute estimates based on difficulty, with total/remaining hours
- **Dark Mode** — full dark theme toggle across header, sidebar, graph, and toasts
- **Graph Analytics Dashboard** — spectral gap, algebraic connectivity, ACO convergence chart, degree distribution
- **Study Streak Calendar** — GitHub-style heatmap tracking daily study activity with current/longest streak stats
- **Weekly Study Goals** — adjustable weekly target with auto-reset, progress bar, and localStorage persistence
- **Topic Notes** — per-topic rich-text note editor with save/delete, persisted per skill
- **Smart Recommendations** — GAT-powered personalised "next topic" suggestions with difficulty indicators

---

## How It Works — The Algorithms

### Spectral Graph Theory (`graph.py` — 1006 lines)

The knowledge graph isn't just a data structure — it's analysed using the **Laplacian spectrum** to understand its topology:

- **Fiedler vector** (2nd eigenvector of the Laplacian) determines natural graph partitions and drives node layout
- **Spectral embedding** maps topics into Euclidean space where distance = topological dissimilarity
- **Algebraic connectivity** (λ₂) measures how tightly connected the curriculum is
- **Spectral clustering** groups related topics without manual labelling
- **ArpackNoConvergence resilience** — graceful fallback when eigensolvers partially converge

### Ant Colony Optimisation (`ACO.py` — 495 lines)

Finding the optimal study order is NP-hard. ACO solves it with swarm intelligence:

- 50 virtual ants walk the graph per iteration, guided by **pheromone trails** and a **spectral heuristic**
- Pheromone is **warm-started using Fiedler distances** — spectrally close topics get higher initial pheromone, converging 2–5× faster
- **Spectral distance** replaces naive edge-hop relatedness for the cost matrix
- Prerequisite constraints are enforced via **vectorised boolean matrix multiplication**
- All matrix ops (cost, heuristic, pheromone deposit) are **fully vectorised with NumPy**
- **Early stopping** when convergence plateaus

### Stochastic Diffusion Search (`SDS.py` — 559 lines)

Real-time spell correction as you type, backed by a 100k+ word dictionary:

- **Vectorised pre-filter** (character frequency + bigram overlap + prefix matching) narrows candidates from 100k → 200
- **SDS agents** test random micro-features (fuzzy character position matching) and cluster on the best match
- Final ranking combines SDS hits, character similarity, and **Levenshtein edit distance**
- All operations on padded NumPy uint8 matrices — zero Python loops over the dictionary

### Web Scraping (`Webscraping.py` — 524 lines)

Structured curriculum extraction from multiple sources:

- **Concurrent fetching** (4 sources in parallel via ThreadPoolExecutor)
- **DAG-aware prerequisite assignment** — foundational topics have no prereqs, higher levels depend on 1–2 topics from the level below, creating realistic fan-in/fan-out

### GAT Difficulty Analysis (`difficulty_gnn.py` — 443 lines)

Self-supervised Graph Attention Network that predicts per-topic difficulty:

- **3-layer GATConv** architecture (10 → 64 → 64 → 1) with multi-head attention
- **10 structural features** per node: in/out-degree, topological depth, prerequisite ratio, spectral coordinates (Fiedler), mastery state, neighbourhood mastery
- **Self-supervised** — no external training data; uses fixed weight initialisation tuned to structural heuristics
- Mastered neighbours reduce effective difficulty, enabling personalised recommendations

---

## File Structure

```text
neuralearn/
├── src/
│   ├── Backend/
│   │   ├── graph.py            # Knowledge graph + spectral analysis (1006 lines)
│   │   ├── ACO.py              # Ant Colony Optimisation with spectral heuristics (495 lines)
│   │   ├── SDS.py              # Stochastic Diffusion Search spell corrector (559 lines)
│   │   ├── difficulty_gnn.py   # GAT difficulty predictor (443 lines)
│   │   ├── Webscraping.py      # Concurrent multi-source web scraper (524 lines)
│   │   ├── api.py              # Flask REST API — 11 endpoints, thread-safe (1000 lines)
│   │   ├── test_all.py         # 5 test suites, 218 assertions (1190 lines)
│   │   └── words in english/   # A–Z dictionary CSVs for spell correction
│   │
│   └── frontend/
│       ├── src/
│       │   ├── app/
│       │   │   ├── pages/
│       │   │   │   ├── Home.tsx             # Landing page with live spell-check
│       │   │   │   └── KnowledgeGraph.tsx   # Interactive graph + 5-tab sidebar
│       │   │   ├── components/
│       │   │   │   ├── CustomNode.tsx       # ReactFlow node with level/cluster badges
│       │   │   │   ├── PomodoroTimer.tsx    # 25/5/15-min study timer with SVG ring
│       │   │   │   ├── GamificationPanel.tsx # XP, levels, achievements, streaks
│       │   │   │   ├── LearningInsightsPanel.tsx # Curriculum cohesion & bottleneck cards
│       │   │   │   ├── SmartRecommendation.tsx # GAT-powered next-topic suggestions
│       │   │   │   ├── StudyStreakCalendar.tsx # GitHub-style study heatmap
│       │   │   │   ├── TopicNotes.tsx       # Per-topic note editor
│       │   │   │   ├── WeeklyStudyGoal.tsx  # Weekly target tracker
│       │   │   │   └── ui/                  # shadcn/ui component library
│       │   │   ├── utils/
│       │   │   │   └── api.ts               # Typed API client (10 functions, timeout handling)
│       │   │   ├── App.tsx
│       │   │   └── routes.tsx
│       │   └── main.tsx
│       ├── index.html
│       ├── vite.config.ts
│       └── package.json
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## Tech Stack

| Layer | Tech |
| ------- | ------ |
| **Graph Engine** | Python 3.14, NumPy, SciPy (sparse Laplacians, ARPACK eigensolver), PyTorch Geometric |
| **Optimisation** | Ant Colony Optimisation (vectorised), Stochastic Diffusion Search |
| **API** | Flask 3.1, flask-cors, thread-safe LRU graph store |
| **Frontend** | React 19, TypeScript 5.7, Vite 6.4, ReactFlow, Tailwind CSS v4, shadcn/ui, Motion, Sonner |
| **Scraping** | requests, BeautifulSoup4, MediaWiki API, concurrent.futures |

---

## Quick Start

```bash
# Clone
git clone https://github.com/AKillerPanda/NeuraLearn.git
cd NeuraLearn

# Backend
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install flask flask-cors torch torch-geometric numpy scipy scikit-learn pandas beautifulsoup4 requests matplotlib seaborn

cd src/Backend
python api.py                   # → http://localhost:5000

# Frontend (new terminal)
cd src/frontend
npm install
npm run dev                     # → http://localhost:5173
```

> **Note:** Always run `api.py` using the **venv Python** (`.venv\Scripts\python.exe api.py`) to ensure all dependencies are available.

---

## Run Tests

```bash
cd src/Backend
python -m pytest test_all.py -v   # 5 suites, 218 assertions
```

---

## API Endpoints

| Method | Endpoint | Description |
| -------- | ---------- | ------------- |
| `POST` | `/api/generate` | Build knowledge graph + learning paths for a skill |
| `POST` | `/api/sub-graph` | Generate sub-graph for a subtopic |
| `POST` | `/api/spell-check` | SDS-powered spell correction |
| `POST` | `/api/master` | Mark a topic as mastered (prerequisite-validated) |
| `POST` | `/api/unmaster` | Un-master a topic with cascade to dependents |
| `POST` | `/api/shortest-path` | Find minimum-topic route to a target node |
| `GET` | `/api/progress/:skill` | Get current mastery progress for a graph |
| `GET` | `/api/flashcards/:skill` | Export Anki-compatible flashcards (TSV) |
| `GET` | `/api/study-stats/:skill` | Study time estimates & difficulty breakdown |
| `GET` | `/api/difficulty/:skill` | GAT-based per-topic difficulty scores & recommendations |
| `GET` | `/api/health` | Health check |

---

## License

See [LICENSE](LICENSE).

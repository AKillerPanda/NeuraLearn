# NeuraLearn

**AI-powered learning path generator** that turns any skill into an interactive knowledge graph with optimised study routes — powered by spectral graph theory, ant colony optimisation, and stochastic diffusion search.

Type a topic. Get a complete curriculum. See how everything connects.

![Python](https://img.shields.io/badge/Python-3.14-blue)
![React](https://img.shields.io/badge/React-19-61DAFB)
![TypeScript](https://img.shields.io/badge/TypeScript-5.7-3178C6)
![Tests](https://img.shields.io/badge/Tests-82%2F82-brightgreen)

---

## What It Does

1. **You type a skill** — "Machine Learning", "Piano", "Web Development", anything
2. **NeuraLearn scrapes the web** concurrently (Wikipedia, GeeksforGeeks, GitHub, DuckDuckGo) to discover subtopics and resources
3. **Builds a knowledge graph** — a DAG where nodes are topics and edges are prerequisite relationships, with real fan-in/fan-out structure
4. **Finds the best learning path** using Ant Colony Optimisation with spectral heuristics — not just any valid order, but the one that minimises cognitive load
5. **Visualises everything** as an interactive graph you can explore, with multiple path options and progress tracking

---

## How It Works — The Algorithms

### Spectral Graph Theory (`graph.py`)
The knowledge graph isn't just a data structure — it's analysed using the **Laplacian spectrum** to understand its topology:
- **Fiedler vector** (2nd eigenvector of the Laplacian) determines natural graph partitions and drives node layout
- **Spectral embedding** maps topics into Euclidean space where distance = topological dissimilarity
- **Algebraic connectivity** (λ₂) measures how tightly connected the curriculum is
- **Spectral clustering** groups related topics without manual labelling

### Ant Colony Optimisation (`ACO.py`)
Finding the optimal study order is NP-hard. ACO solves it with swarm intelligence:
- 50 virtual ants walk the graph per iteration, guided by **pheromone trails** and a **spectral heuristic**
- Pheromone is **warm-started using Fiedler distances** — spectrally close topics get higher initial pheromone, converging 2–5× faster
- **Spectral distance** replaces naive edge-hop relatedness for the cost matrix
- Prerequisite constraints are enforced via **vectorised boolean matrix multiplication**
- All matrix ops (cost, heuristic, pheromone deposit) are **fully vectorised with NumPy**

### Stochastic Diffusion Search (`SDS.py`)
Real-time spell correction as you type, backed by a 100k+ word dictionary:
- **Vectorised pre-filter** (character frequency + bigram overlap + prefix matching) narrows candidates from 100k → 200
- **SDS agents** test random micro-features (fuzzy character position matching) and cluster on the best match
- Final ranking combines SDS hits, character similarity, and **Levenshtein edit distance**
- All operations on padded NumPy uint8 matrices — zero Python loops over the dictionary

### Web Scraping (`Webscraping.py`)
Structured curriculum extraction from multiple sources:
- **Concurrent fetching** (4 sources in parallel via ThreadPoolExecutor)
- **DAG-aware prerequisite assignment** — foundational topics have no prereqs, higher levels depend on 1–2 topics from the level below, creating realistic fan-in/fan-out

---

## File Structure

```
neuralearn/
├── src/
│   ├── Backend/
│   │   ├── graph.py            # Knowledge graph + spectral analysis (Laplacian, Fiedler, clustering)
│   │   ├── ACO.py              # Ant Colony Optimisation with spectral heuristics
│   │   ├── SDS.py              # Stochastic Diffusion Search spell corrector
│   │   ├── Webscraping.py      # Concurrent multi-source web scraper
│   │   ├── api.py              # Flask REST API (spectral layout engine)
│   │   ├── test_all.py         # 82 tests covering all modules
│   │   └── words in english/   # A–Z dictionary CSVs for spell correction
│   │
│   └── frontend/
│       ├── src/
│       │   ├── app/
│       │   │   ├── pages/
│       │   │   │   ├── Home.tsx             # Landing page with live spell-check
│       │   │   │   └── KnowledgeGraph.tsx   # Interactive graph + path explorer
│       │   │   ├── components/
│       │   │   │   ├── CustomNode.tsx       # ReactFlow node with level badges
│       │   │   │   └── ui/                  # shadcn/ui component library
│       │   │   ├── utils/
│       │   │   │   └── api.ts               # Typed API client with timeout handling
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
|-------|------|
| **Graph Engine** | Python, NumPy, SciPy (sparse Laplacians, ARPACK eigensolver), PyTorch Geometric |
| **Optimisation** | Ant Colony Optimisation (vectorised), Stochastic Diffusion Search |
| **API** | Flask, flask-cors |
| **Frontend** | React 19, TypeScript, Vite, ReactFlow, Tailwind CSS v4, shadcn/ui, Motion |
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

---

## Run Tests

```bash
cd src/Backend
python test_all.py              # 82/82 tests
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/generate` | Build knowledge graph + learning paths for a skill |
| `POST` | `/api/spell-check` | SDS-powered spell correction |
| `POST` | `/api/sub-graph` | Generate sub-graph for a subtopic |
| `GET` | `/api/health` | Health check |

---

## License

See [LICENSE](LICENSE).

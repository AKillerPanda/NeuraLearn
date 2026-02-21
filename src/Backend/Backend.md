# Backend

The brains of NeuraLearn — four core algorithm modules, a REST API, and a comprehensive test suite.

## Files

| File | What It Does |
|------|-------------|
| **`graph.py`** | Knowledge graph engine. Stores topics as a DAG with prerequisite edges. Includes spectral graph theory — Laplacian matrices, Fiedler vector for graph partitioning, spectral embedding/clustering, algebraic connectivity (λ₂), and vectorised topological depth. All spectral results are cached and auto-invalidated on structural changes. |
| **`ACO.py`** | Ant Colony Optimisation for finding the optimal learning path. Cost matrix uses spectral distances (not just direct edges) so the ants understand global graph topology. Pheromone is warm-started via Fiedler vector distances for 2–5× faster convergence. Fully vectorised with NumPy — no Python loops in the hot path. |
| **`SDS.py`** | Stochastic Diffusion Search for real-time spell correction. Pre-filters 100k+ dictionary words down to ~200 candidates using vectorised character frequency, bigram overlap, and prefix matching on padded NumPy matrices. SDS agents then cluster on the best fuzzy match. Final ranking uses Levenshtein edit distance. |
| **`Webscraping.py`** | Concurrent multi-source scraper (Wikipedia MediaWiki API, GeeksforGeeks, GitHub awesome lists, DuckDuckGo). All four sources are fetched in parallel via `ThreadPoolExecutor`. Outputs a DAG-aware spec where foundational topics have no prereqs and higher levels fan-in from 1–2 topics below. |
| **`api.py`** | Flask REST API serving the frontend. Layout engine uses the Fiedler vector for natural x-axis positioning and topological depth for y-axis. Exposes `/api/generate`, `/api/spell-check`, `/api/sub-graph`, and `/api/health`. |
| **`test_all.py`** | 82 tests covering graph construction, spectral methods (Laplacian, Fiedler, clustering, Betti numbers), ACO matrix shapes/convergence, SDS spell correction accuracy, webscraping data structures, and end-to-end integration. |
| **`words in english/`** | A–Z CSV files (~100k words) used by SDS for the spell correction dictionary. Offensive words are filtered out at load time. |

## Key Design Decisions

- **Spectral over heuristic** — Graph analysis uses actual eigenvalues of the Laplacian rather than hand-tuned rules. This means the system adapts to any graph shape automatically.
- **Vectorisation everywhere** — NumPy broadcasting replaces Python loops in all hot paths (ACO walks, SDS agents, matrix construction). Typical speedup: 10–50×.
- **Cache-and-invalidate** — Expensive computations (topo sort, Laplacian, eigenvectors, spectral embedding) are cached and only recomputed when the graph structure changes.
- **DAG not chain** — The webscraper builds real prerequisite DAGs with fan-in/fan-out, not linear chains. Topics at the same level can be studied in parallel.
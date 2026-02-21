# Backend

The brains of NeuraLearn — four core algorithm modules, a REST API with 10 endpoints, and a comprehensive test suite.

## Files

| File | Lines | What It Does |
|------|------:|-------------|
| **`graph.py`** | 903 | Knowledge graph engine. Stores topics as a DAG with prerequisite edges. Includes spectral graph theory — Laplacian matrices, Fiedler vector for graph partitioning, spectral embedding/clustering, algebraic connectivity (λ₂), and vectorised topological depth. All spectral results are cached and auto-invalidated on structural changes. Handles `ArpackNoConvergence` gracefully with partial-convergence fallback. Resilient `from_spec()` skips missing prerequisites instead of crashing. |
| **`ACO.py`** | 495 | Ant Colony Optimisation for finding the optimal learning path. Cost matrix uses spectral distances (not just direct edges) so the ants understand global graph topology. Pheromone is warm-started via Fiedler vector distances for 2–5× faster convergence. Fully vectorised with NumPy — no Python loops in the hot path. Early stopping when convergence plateaus. Matplotlib import is lazy (only when plotting). |
| **`SDS.py`** | 557 | Stochastic Diffusion Search for real-time spell correction. Pre-filters 100k+ dictionary words down to ~200 candidates using vectorised character frequency, bigram overlap, and prefix matching on padded NumPy matrices. SDS agents then cluster on the best fuzzy match. Final ranking uses Levenshtein edit distance. |
| **`Webscraping.py`** | 524 | Concurrent multi-source scraper (Wikipedia MediaWiki API, GeeksforGeeks, GitHub awesome lists, DuckDuckGo). All four sources are fetched in parallel via `ThreadPoolExecutor`. Outputs a DAG-aware spec where foundational topics have no prereqs and higher levels fan-in from 1–2 topics below. |
| **`api.py`** | 858 | Flask REST API serving the frontend. **10 endpoints** covering graph generation, sub-graphs, mastery tracking (with prerequisite validation and cascade unmastering), shortest path, spell-check, flashcard export, and study stats. Layout engine uses the Fiedler vector for x-axis positioning and topological depth for y-axis (Sugiyama-style with barycenter heuristic). Thread-safe LRU graph store (max 50 graphs) with `threading.Lock`. Study time estimation per difficulty level. |
| **`test_all.py`** | 1171 | 5 test suites with 213 assertions covering graph construction, spectral methods (Laplacian, Fiedler, clustering, Betti numbers), ACO matrix shapes/convergence, SDS spell correction accuracy, webscraping data structures, and end-to-end integration. |
| **`words in english/`** | — | A–Z CSV files (~100k words) used by SDS for the spell correction dictionary. Offensive words are filtered out at load time. |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/generate` | Build knowledge graph + learning paths for a skill |
| `POST` | `/api/sub-graph` | Generate sub-graph for a subtopic |
| `POST` | `/api/spell-check` | SDS-powered spell correction |
| `POST` | `/api/master` | Mark a topic as mastered (prerequisite-validated) |
| `POST` | `/api/unmaster` | Un-master a topic with cascade to dependents |
| `POST` | `/api/shortest-path` | Find minimum-topic route to a target node |
| `GET`  | `/api/progress/:skill` | Get current mastery progress for a graph |
| `GET`  | `/api/flashcards/:skill` | Export Anki-compatible flashcards (TSV) |
| `GET`  | `/api/study-stats/:skill` | Study time estimates & difficulty breakdown |
| `GET`  | `/api/health` | Health check |

## Key Design Decisions

- **Spectral over heuristic** — Graph analysis uses actual eigenvalues of the Laplacian rather than hand-tuned rules. This means the system adapts to any graph shape automatically.
- **Vectorisation everywhere** — NumPy broadcasting replaces Python loops in all hot paths (ACO walks, SDS agents, matrix construction). Typical speedup: 10–50×.
- **Cache-and-invalidate** — Expensive computations (topo sort, Laplacian, eigenvectors, spectral embedding) are cached and only recomputed when the graph structure changes.
- **DAG not chain** — The webscraper builds real prerequisite DAGs with fan-in/fan-out, not linear chains. Topics at the same level can be studied in parallel.
- **Thread-safe concurrency** — `threading.Lock` protects the shared graph store so Flask's reloader and concurrent requests don't corrupt state.
- **LRU eviction** — The graph store uses `OrderedDict` with a max-50 cap to prevent unbounded memory growth.
- **Resilient eigensolvers** — All `eigsh` calls catch `ArpackNoConvergence` and fall back to partial results, so large/complex graphs never crash.
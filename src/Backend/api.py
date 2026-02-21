"""
NeuraLearn — Flask REST API  (spectral layout + vectorised)
============================================================
Exposes the backend algorithms (Webscraping, KnowledgeGraph, ACO, SDS)
as JSON endpoints that the React frontend can consume.

Layout uses spectral graph embedding (Laplacian eigenvectors) for natural
node positioning, falling back to topological-depth layers when the graph
is too small for meaningful spectral analysis.

Endpoints
---------
POST /api/generate       — Build knowledge graph + learning paths for a skill
POST /api/spell-check    — SDS spell-correction for a word / phrase
POST /api/sub-graph      — Generate a sub-graph for a specific subtopic
GET  /api/health         — Health check
"""
from __future__ import annotations

import logging
import math
import time
import traceback
from typing import Any

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

# ── Backend imports ─────────────────────────────────────────────────
from Webscraping import get_learning_spec
from graph import KnowledgeGraph, TopicLevel
from ACO import LearningPathACO
from SDS import spell_correct, correct_phrase, load_dictionary

# ── App setup ───────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # allow requests from the Vite dev server

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# Pre-load the dictionary on startup so /spell-check is fast
_dict = load_dictionary()
log.info("Dictionary loaded: %d words", len(_dict))


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════
_LEVEL_TO_DIFFICULTY: dict[str, str] = {
    "foundational": "beginner",
    "intermediate": "intermediate",
    "advanced":     "advanced",
    "expert":       "advanced",
}

_LEVEL_ORDER = ["foundational", "intermediate", "advanced", "expert"]


def _layout_nodes(kg: KnowledgeGraph) -> list[dict[str, Any]]:
    """
    Sugiyama-style layered layout with spectral cross-minimisation.

    1. Y-axis  — topological depth layers (vectorised BFS).
    2. X-axis  — *within* each layer, order nodes to minimise edge crossings
       via barycenter heuristic (average x of parents), seeded with spectral
       Fiedler ordering for the root layer.
    3. Spacing — enforce a hard minimum gap (NODE_W) so nodes never overlap
       regardless of how many share a layer.

    The result is a clean, readable DAG where every level is evenly spaced,
    edges flow strictly downward, and no two nodes collide.
    """
    topo = kg.learning_order()
    if not topo:
        return []

    # ── constants ───────────────────────────────────────────────────
    NODE_W   = 260          # min horizontal gap (> node width ~220 px)
    Y_GAP    = 160          # vertical gap between layers
    CENTER_X = 500          # viewport centre-x

    # ── depth vector ────────────────────────────────────────────────
    depth_vec = kg.topological_depth_vector()          # (N,) int32
    max_depth = int(depth_vec[list(kg.topics.keys())].max()) if kg.topics else 0

    # ── group topics by layer ───────────────────────────────────────
    layers: dict[int, list] = {}
    for t in topo:
        d = int(depth_vec[t.topic_id])
        layers.setdefault(d, []).append(t)

    # ── spectral seed for the root layer ────────────────────────────
    spectral_order: dict[int, float] = {}
    if kg.num_topics >= 3:
        try:
            fiedler = kg.fiedler_vector()
            fv_max = float(np.abs(fiedler).max())
            if fv_max > 1e-12:
                for tid in kg.topics:
                    spectral_order[tid] = float(fiedler[tid]) / fv_max
        except Exception:
            pass

    # ── barycenter cross-minimisation (two passes) ──────────────────
    # Assign each topic a fractional x-rank within its layer.
    # Roots are ordered by Fiedler value; deeper layers by the mean
    # x-rank of their parents (barycenter heuristic — Sugiyama §3).
    layer_rank: dict[int, float] = {}   # topic_id → x-rank (0-based float)

    for d in range(max_depth + 1):
        lt = layers.get(d, [])
        if not lt:
            continue
        if d == 0:
            # Root layer: prefer spectral order, else alphabetical
            lt.sort(key=lambda t: spectral_order.get(t.topic_id, 0.0))
        else:
            # Barycenter: average rank of already-placed parents
            def _bary(t):
                parents = [layer_rank[p] for p in t.prerequisites if p in layer_rank]
                return sum(parents) / len(parents) if parents else 0.0
            lt.sort(key=_bary)
        for i, t in enumerate(lt):
            layer_rank[t.topic_id] = float(i)

    # 2nd pass (bottom-up refinement) — reduces remaining crossings
    for d in range(max_depth, -1, -1):
        lt = layers.get(d, [])
        if not lt:
            continue
        def _child_bary(t):
            children = [layer_rank[u] for u in t.unlocks if u in layer_rank]
            return sum(children) / len(children) if children else layer_rank.get(t.topic_id, 0.0)
        lt.sort(key=_child_bary)
        for i, t in enumerate(lt):
            layer_rank[t.topic_id] = float(i)

    # ── absolute pixel positions (centred, no-overlap) ──────────────
    nodes: list[dict[str, Any]] = []
    for d in range(max_depth + 1):
        lt = layers.get(d, [])
        if not lt:
            continue
        n = len(lt)
        total_w = (n - 1) * NODE_W
        start_x = CENTER_X - total_w / 2
        for i, t in enumerate(lt):
            level_str = t.level.name.lower() if hasattr(t.level, "name") else str(t.level)
            node_type = "input" if d == 0 else ("output" if d == max_depth else "default")

            # Contextual metadata for the frontend
            prereq_names = sorted(kg.topics[p].name for p in t.prerequisites)
            unlock_names = sorted(kg.topics[u].name for u in t.unlocks)

            nodes.append({
                "id":       str(t.topic_id),
                "type":     node_type,
                "position": {"x": round(start_x + i * NODE_W), "y": d * Y_GAP},
                "data": {
                    "label":        t.name,
                    "level":        level_str,
                    "difficulty":   _LEVEL_TO_DIFFICULTY.get(level_str, "intermediate"),
                    "description":  t.description or "",
                    "mastered":     t.mastered,
                    "prerequisites": prereq_names,
                    "unlocks":      unlock_names,
                    "depth":        d,
                    "stepIndex":    topo.index(t),
                },
            })

    return nodes


def _build_edges(kg: KnowledgeGraph) -> list[dict[str, Any]]:
    """Build ReactFlow edge list from the KnowledgeGraph."""
    edges: list[dict[str, Any]] = []
    for t in kg.topics.values():
        for pid in t.prerequisites:
            edges.append({
                "id":     f"e{pid}-{t.topic_id}",
                "source": str(pid),
                "target": str(t.topic_id),
                "animated": False,
                "markerEnd": {
                    "type": "arrowclosed",
                    "width": 20,
                    "height": 20,
                    "color": "#a78bfa",
                },
            })
    return edges


def _build_learning_paths(
    kg: KnowledgeGraph,
    aco_kwargs: dict | None = None,
) -> list[dict[str, Any]]:
    """
    Build multiple learning paths with rich, actionable descriptions.

      1. Complete Path  — full topological order (every topic)
      2. Optimal Path   — ACO-optimised, smooth difficulty curve
      3. Quick Start    — foundational + intermediate only (fast wins)

    Each path carries per-step metadata so the frontend can show
    "why this order?" at every node.
    """
    paths: list[dict[str, Any]] = []
    topo = kg.learning_order()
    all_ids = [str(t.topic_id) for t in topo]

    # helper: build an ordered step list with context for a sequence of topic ids
    def _steps_with_context(topic_ids: list[int]) -> list[dict[str, str]]:
        steps = []
        for idx, tid in enumerate(topic_ids):
            t = kg.topics[tid]
            prereq_names = sorted(kg.topics[p].name for p in t.prerequisites)
            unlock_names = sorted(kg.topics[u].name for u in t.unlocks if u in {int(x) for x in [str(i) for i in topic_ids]})
            level_str = t.level.name.lower() if hasattr(t.level, "name") else str(t.level)
            steps.append({
                "topicId":   str(tid),
                "name":       t.name,
                "level":      level_str,
                "requires":   prereq_names,
                "unlocks":    unlock_names,
                "reason":     (
                    "Start here — no prerequisites needed"
                    if not prereq_names
                    else f"Ready after mastering {', '.join(prereq_names)}"
                ),
            })
        return steps

    # Path 1 — full topo order
    foundations = [t for t in topo if t.level == TopicLevel.FOUNDATIONAL]
    advanced    = [t for t in topo if t.level in (TopicLevel.ADVANCED, TopicLevel.EXPERT)]
    paths.append({
        "id":          "path-full",
        "name":        "Complete Path",
        "description": (
            f"Master all {len(topo)} topics in prerequisite order — "
            f"starting with {len(foundations)} fundamentals, building to "
            f"{len(advanced)} advanced concepts."
        ),
        "duration":    f"{max(len(topo), 1)} topics",
        "difficulty":  "advanced",
        "nodeIds":     all_ids,
    })

    # Path 2 — ACO-optimised
    try:
        # Adaptive ACO sizing: small graphs → fewer ants & iterations
        K = kg.num_topics
        kw = dict(
            m=min(max(K * 2, 10), 30),
            k_max=min(max(K * 3, 15), 40),
            time_limit=4,
        )
        if aco_kwargs:
            kw.update(aco_kwargs)
        aco = LearningPathACO(kg, **kw)
        aco_path, aco_cost = aco.optimise()
        aco_ids = [str(tid) for tid in aco_path]
        paths.append({
            "id":          "path-aco",
            "name":        "Optimal Path",
            "description": (
                f"AI-optimised order that minimises difficulty jumps and keeps "
                f"related topics together (cost {aco_cost:.1f}). This path "
                f"ensures the smoothest learning curve."
            ),
            "duration":    f"{len(aco_path)} topics",
            "difficulty":  "intermediate",
            "nodeIds":     aco_ids,
        })
    except Exception as exc:
        log.warning("ACO failed: %s", exc)

    # Path 3 — quick start (foundational + intermediate only)
    quick_topics = [
        t for t in topo
        if t.level in (TopicLevel.FOUNDATIONAL, TopicLevel.INTERMEDIATE)
    ]
    if quick_topics and len(quick_topics) < len(topo):
        paths.append({
            "id":          "path-quick",
            "name":        "Quick Start",
            "description": (
                f"Cover the {len(quick_topics)} essential topics "
                f"(foundational + intermediate) to get productive fast — "
                f"skip {len(topo) - len(quick_topics)} advanced topics for later."
            ),
            "duration":    f"{len(quick_topics)} topics",
            "difficulty":  "beginner",
            "nodeIds":     [str(t.topic_id) for t in quick_topics],
        })

    return paths


# ═══════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "dictionary_size": len(_dict)})


@app.route("/api/generate", methods=["POST"])
def generate_graph():
    """
    Build a full knowledge graph for a skill.

    Request JSON:  { "skill": "Machine Learning" }
    Response JSON: { "nodes": [...], "edges": [...], "paths": [...], "skill": "..." }
    """
    body = request.get_json(silent=True) or {}
    skill = (body.get("skill") or "").strip()
    if not skill:
        return jsonify({"error": "missing 'skill' field"}), 400

    log.info("generate  skill=%r", skill)
    t0 = time.time()

    try:
        # 1. Webscrape subtopics
        spec = get_learning_spec(skill)
        t_scrape = time.time() - t0
        if not spec:
            return jsonify({"error": f"no subtopics found for '{skill}'"}), 404

        # 2. Build knowledge graph (should be < 1 ms)
        t1 = time.time()
        kg = KnowledgeGraph.from_spec(spec)
        t_graph = time.time() - t1

        # 3. Layout + edges + paths (should be < 50 ms total)
        t2 = time.time()
        nodes = _layout_nodes(kg)
        t_layout = time.time() - t2

        t3 = time.time()
        edges = _build_edges(kg)
        t_edges = time.time() - t3

        t4 = time.time()
        paths = _build_learning_paths(kg)
        t_paths = time.time() - t4

        elapsed = time.time() - t0
        compute_ms = (t_graph + t_layout + t_edges + t_paths) * 1000
        log.info(
            "generate  skill=%r  topics=%d  elapsed=%.2fs  "
            "(scrape=%.2fs  graph=%.1fms  layout=%.1fms  edges=%.1fms  paths=%.1fms)",
            skill, kg.num_topics, elapsed,
            t_scrape, t_graph * 1000, t_layout * 1000, t_edges * 1000, t_paths * 1000,
        )

        return jsonify({
            "skill":   skill,
            "nodes":   nodes,
            "edges":   edges,
            "paths":   paths,
            "elapsed": round(elapsed, 2),
            "timing": {
                "scrape_s":    round(t_scrape, 3),
                "graph_ms":    round(t_graph * 1000, 2),
                "layout_ms":   round(t_layout * 1000, 2),
                "edges_ms":    round(t_edges * 1000, 2),
                "paths_ms":    round(t_paths * 1000, 2),
                "compute_ms":  round(compute_ms, 2),
            },
        })

    except Exception as exc:
        log.error("generate failed: %s\n%s", exc, traceback.format_exc())
        return jsonify({"error": str(exc)}), 500


@app.route("/api/sub-graph", methods=["POST"])
def sub_graph():
    """
    Generate a sub-graph for a specific subtopic.

    Request JSON:  { "topic": "Neural Networks" }
    Response JSON: { "nodes": [...], "edges": [...], "paths": [...], "skill": "..." }
    """
    body = request.get_json(silent=True) or {}
    topic = (body.get("topic") or "").strip()
    if not topic:
        return jsonify({"error": "missing 'topic' field"}), 400

    log.info("sub-graph  topic=%r", topic)
    t0 = time.time()

    try:
        spec = get_learning_spec(topic)
        t_scrape = time.time() - t0
        if not spec:
            return jsonify({"error": f"no subtopics found for '{topic}'"}), 404

        t1 = time.time()
        kg = KnowledgeGraph.from_spec(spec)
        t_graph = time.time() - t1

        t2 = time.time()
        nodes = _layout_nodes(kg)
        t_layout = time.time() - t2

        edges = _build_edges(kg)
        paths = _build_learning_paths(kg)
        compute_ms = (time.time() - t1) * 1000

        log.info(
            "sub-graph  topic=%r  topics=%d  compute=%.1fms  scrape=%.2fs",
            topic, kg.num_topics, compute_ms, t_scrape,
        )

        return jsonify({
            "skill": topic,
            "nodes": nodes,
            "edges": edges,
            "paths": paths,
            "timing": {
                "scrape_s":   round(t_scrape, 3),
                "compute_ms": round(compute_ms, 2),
            },
        })

    except Exception as exc:
        log.error("sub-graph failed: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/spell-check", methods=["POST"])
def spell_check():
    """
    Spell-correct a word or phrase using SDS.

    Request JSON:  { "text": "mathmatics" }
                or { "text": "mathmatics", "top_k": 5 }
    Response JSON: { "results": [ { "original": "...", "suggestions": [...] } ] }
    """
    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()
    top_k = body.get("top_k", 5)
    if not text:
        return jsonify({"error": "missing 'text' field"}), 400

    log.info("spell-check  text=%r", text)

    try:
        per_word = correct_phrase(text, top_k=top_k)
        tokens = text.split()
        results = []

        for token, suggestions in zip(tokens, per_word):
            results.append({
                "original":    token,
                "suggestions": [
                    {"word": w, "score": round(sc, 4)}
                    for w, sc in suggestions
                ],
                "inDictionary": len(suggestions) == 1 and suggestions[0][1] == 1.0,
            })

        return jsonify({"results": results})

    except Exception as exc:
        log.error("spell-check failed: %s", exc)
        return jsonify({"error": str(exc)}), 500


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

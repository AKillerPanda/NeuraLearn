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
    Compute node positions using spectral embedding + topological depth.

    Strategy:
      - Y axis: topological depth (vectorised via KnowledgeGraph.topological_depth_vector)
      - X axis: spectral embedding (Fiedler vector / 2nd Laplacian eigenvector)
                gives a natural left-right spread that groups related topics

    Falls back to simple column layout if the graph is too small (<3 nodes)
    for meaningful spectral analysis.
    """
    topo = kg.learning_order()
    if not topo:
        return []

    # Vectorised depth computation
    depth_vec = kg.topological_depth_vector()  # (N,) int32

    max_depth = int(depth_vec[list(kg.topics.keys())].max()) if kg.topics else 0

    # Try spectral positioning for x-axis
    use_spectral = kg.num_topics >= 3
    spectral_x: dict[int, float] = {}
    if use_spectral:
        try:
            fiedler = kg.fiedler_vector()
            # Normalise Fiedler vector to [-1, 1]
            fv_vals = np.array([fiedler[t.topic_id] for t in topo])
            fv_max = np.abs(fv_vals).max()
            if fv_max > 1e-12:
                fv_normed = fv_vals / fv_max
            else:
                fv_normed = np.zeros_like(fv_vals)
            for i, t in enumerate(topo):
                spectral_x[t.topic_id] = float(fv_normed[i])
        except Exception:
            use_spectral = False

    # Group by depth layer
    layers: dict[int, list] = {}
    for t in topo:
        d = int(depth_vec[t.topic_id])
        layers.setdefault(d, []).append(t)

    nodes: list[dict[str, Any]] = []
    x_spacing = 260
    y_spacing = 140
    center_x = 400

    for d, layer_topics in sorted(layers.items()):
        n = len(layer_topics)

        if use_spectral and spectral_x:
            # Sort layer by spectral x for consistent ordering
            layer_topics.sort(key=lambda t: spectral_x.get(t.topic_id, 0.0))
            # Use spectral values for x-positioning, scaled to viewport
            for i, t in enumerate(layer_topics):
                sx = spectral_x.get(t.topic_id, 0.0)
                x = center_x + sx * (x_spacing * max(n - 1, 1) / 2)
                level_str = t.level.name.lower() if hasattr(t.level, "name") else str(t.level)
                node_type = "input" if d == 0 else ("output" if d == max_depth else "default")
                nodes.append({
                    "id":       str(t.topic_id),
                    "type":     node_type,
                    "position": {"x": round(x), "y": d * y_spacing},
                    "data": {
                        "label":      t.name,
                        "level":      level_str,
                        "difficulty": _LEVEL_TO_DIFFICULTY.get(level_str, "intermediate"),
                        "description": t.description or "",
                        "mastered":   t.mastered,
                    },
                })
        else:
            # Fallback: simple column-centred layout
            total_width = (n - 1) * x_spacing
            start_x = center_x - total_width / 2
            for i, t in enumerate(layer_topics):
                level_str = t.level.name.lower() if hasattr(t.level, "name") else str(t.level)
                node_type = "input" if d == 0 else ("output" if d == max_depth else "default")
                nodes.append({
                    "id":       str(t.topic_id),
                    "type":     node_type,
                    "position": {"x": round(start_x + i * x_spacing), "y": d * y_spacing},
                    "data": {
                        "label":      t.name,
                        "level":      level_str,
                        "difficulty": _LEVEL_TO_DIFFICULTY.get(level_str, "intermediate"),
                        "description": t.description or "",
                        "mastered":   t.mastered,
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
            })
    return edges


def _build_learning_paths(
    kg: KnowledgeGraph,
    aco_kwargs: dict | None = None,
) -> list[dict[str, Any]]:
    """
    Build multiple learning paths:
      1. Full topological order      (Complete Path)
      2. ACO-optimised path          (Optimal Path)
      3. Quick-start: foundational + intermediate only  (Quick Start)
    """
    paths: list[dict[str, Any]] = []
    topo = kg.learning_order()
    all_ids = [str(t.topic_id) for t in topo]

    # Path 1 — full topo order
    paths.append({
        "id":          "path-full",
        "name":        "Complete Path",
        "description": "Cover every topic in prerequisite order",
        "duration":    f"{max(len(topo), 1)} topics",
        "difficulty":  "advanced",
        "nodeIds":     all_ids,
    })

    # Path 2 — ACO-optimised
    try:
        kw = dict(m=30, k_max=40, time_limit=8)
        if aco_kwargs:
            kw.update(aco_kwargs)
        aco = LearningPathACO(kg, **kw)
        aco_path, aco_cost = aco.optimise()
        aco_ids = [str(tid) for tid in aco_path]
        paths.append({
            "id":          "path-aco",
            "name":        "Optimal Path",
            "description": f"ACO-optimised learning order (cost {aco_cost:.1f})",
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
            "description": "Foundational & intermediate topics only",
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
        if not spec:
            return jsonify({"error": f"no subtopics found for '{skill}'"}), 404

        # 2. Build knowledge graph
        kg = KnowledgeGraph.from_spec(spec)

        # 3. Layout + edges + paths
        nodes = _layout_nodes(kg)
        edges = _build_edges(kg)
        paths = _build_learning_paths(kg)

        elapsed = time.time() - t0
        log.info("generate  skill=%r  topics=%d  elapsed=%.1fs", skill, kg.num_topics, elapsed)

        return jsonify({
            "skill":   skill,
            "nodes":   nodes,
            "edges":   edges,
            "paths":   paths,
            "elapsed": round(elapsed, 2),
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

    try:
        spec = get_learning_spec(topic)
        if not spec:
            return jsonify({"error": f"no subtopics found for '{topic}'"}), 404

        kg = KnowledgeGraph.from_spec(spec)
        nodes = _layout_nodes(kg)
        edges = _build_edges(kg)
        paths = _build_learning_paths(kg)

        return jsonify({
            "skill": topic,
            "nodes": nodes,
            "edges": edges,
            "paths": paths,
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

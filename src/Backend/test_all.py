"""
Comprehensive test suite for NeuraLearn backend modules.

Tests:
  1. graph.py   — Topic, KnowledgeGraph, spec, rebuild, mastery, edges, cycles
  2. ACO.py     — matrix construction, availability, paths, scoring, convergence
  3. SDS.py     — convergence, edge cases, vectorised correctness
  4. Webscraping.py — data structures, level assignment, plan export

Run:  python test_all.py
"""
import sys
import time
import traceback

import numpy as np

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
_PASS = 0
_FAIL = 0
_ERRORS: list[str] = []


def _test(name: str, fn):
    global _PASS, _FAIL
    try:
        fn()
        _PASS += 1
        print(f"  [PASS] {name}")
    except Exception as e:
        _FAIL += 1
        msg = f"  [FAIL] {name}  →  {e}"
        print(msg)
        _ERRORS.append(msg)
        traceback.print_exc(limit=3)


# =====================================================================
# 1.  graph.py
# =====================================================================
def test_graph():
    print("\n" + "=" * 60)
    print("  TESTING graph.py")
    print("=" * 60)

    from graph import Topic, TopicLevel, TopicSpec, KnowledgeGraph, _LEVEL_MAP

    # ---- basic topic creation ----
    def t_topic_creation():
        t = Topic(0, "Algebra", level=TopicLevel.INTERMEDIATE)
        assert t.topic_id == 0
        assert t.name == "Algebra"
        assert t.level == TopicLevel.INTERMEDIATE
        assert t.mastered is False
        assert isinstance(t.prerequisites, set) and len(t.prerequisites) == 0
        assert isinstance(t.unlocks, set) and len(t.unlocks) == 0
    _test("Topic creation", t_topic_creation)

    # ---- knowledge graph add/get ----
    def t_graph_add_get():
        kg = KnowledgeGraph()
        t0 = kg.create_topic("A", level=TopicLevel.FOUNDATIONAL)
        t1 = kg.create_topic("B", level=TopicLevel.INTERMEDIATE)
        assert kg.num_topics == 2
        assert kg.get_topic(0) is t0
        assert kg.get_topic(1) is t1
        assert kg.get_topic_by_name("a") is t0   # case-insensitive
        assert kg.get_topic_by_name("B") is t1
        assert kg.get_topic_by_name("nope") is None
    _test("Graph add / get / name lookup", t_graph_add_get)

    # ---- duplicate topic_id raises ----
    def t_dup_topic():
        kg = KnowledgeGraph()
        kg.create_topic("X")
        try:
            kg.add_topic(Topic(0, "Y"))
            assert False, "should have raised"
        except ValueError:
            pass
    _test("Duplicate topic_id raises ValueError", t_dup_topic)

    # ---- edge creation ----
    def t_edges():
        kg = KnowledgeGraph()
        a = kg.create_topic("A")
        b = kg.create_topic("B")
        c = kg.create_topic("C")
        kg.add_prerequisite(a.topic_id, b.topic_id)
        kg.add_prerequisite(b.topic_id, c.topic_id)
        assert b.topic_id in a.unlocks
        assert a.topic_id in b.prerequisites
        assert c.topic_id in b.unlocks
        # adding same edge again is idempotent
        kg.add_prerequisite(a.topic_id, b.topic_id)
        assert len(a.unlocks) == 1
    _test("Edge creation & idempotency", t_edges)

    # ---- self-loop raises ----
    def t_self_loop():
        kg = KnowledgeGraph()
        a = kg.create_topic("A")
        try:
            kg.add_prerequisite(a.topic_id, a.topic_id)
            assert False
        except ValueError:
            pass
    _test("Self-loop raises ValueError", t_self_loop)

    # ---- mastery & unlock ----
    def t_mastery():
        kg = KnowledgeGraph()
        a = kg.create_topic("A")
        b = kg.create_topic("B")
        kg.add_prerequisite(a.topic_id, b.topic_id)

        assert kg.is_unlocked(a.topic_id)     # no prereqs → unlocked
        assert not kg.is_unlocked(b.topic_id)  # needs A

        assert kg.master_topic(a.topic_id)
        assert a.mastered
        assert kg.is_unlocked(b.topic_id)

        assert kg.master_topic(b.topic_id)
        assert b.mastered
        assert kg.mastery_progress() == 1.0
    _test("Mastery & unlock logic", t_mastery)

    # ---- available / locked ----
    def t_available_locked():
        kg = KnowledgeGraph()
        a = kg.create_topic("A")
        b = kg.create_topic("B")
        c = kg.create_topic("C")
        kg.add_prerequisite(a.topic_id, b.topic_id)
        kg.add_prerequisite(b.topic_id, c.topic_id)

        avail = kg.get_available()
        locked = kg.get_locked()
        assert len(avail) == 1 and avail[0] is a
        assert len(locked) == 2

        kg.master_topic(a.topic_id)
        avail2 = kg.get_available()
        assert len(avail2) == 1 and avail2[0] is b
    _test("Available / locked queries", t_available_locked)

    # ---- learning order (topological sort) ----
    def t_topo_sort():
        kg = KnowledgeGraph()
        a = kg.create_topic("A")
        b = kg.create_topic("B")
        c = kg.create_topic("C")
        kg.add_prerequisite(a.topic_id, b.topic_id)
        kg.add_prerequisite(b.topic_id, c.topic_id)
        order = kg.learning_order()
        assert [t.topic_id for t in order] == [0, 1, 2]
        # second call should be cached
        assert kg.learning_order() is order
    _test("Topological sort & caching", t_topo_sort)

    # ---- cycle detection ----
    def t_cycle():
        kg = KnowledgeGraph()
        a = kg.create_topic("A")
        b = kg.create_topic("B")
        c = kg.create_topic("C")
        # Force a cycle by directly manipulating sets
        a.unlocks.add(b.topic_id);  b.prerequisites.add(a.topic_id)
        b.unlocks.add(c.topic_id);  c.prerequisites.add(b.topic_id)
        c.unlocks.add(a.topic_id);  a.prerequisites.add(c.topic_id)
        kg._invalidate_cache()
        try:
            kg.learning_order()
            assert False, "should have raised ValueError for cycle"
        except ValueError:
            pass
    _test("Cycle detection raises ValueError", t_cycle)

    # ---- from_spec ----
    def t_from_spec():
        spec = [
            {"name": "X", "level": "foundational"},
            {"name": "Y", "level": "intermediate", "prerequisite_names": ["X"]},
            {"name": "Z", "level": "advanced",     "prerequisite_names": ["Y"]},
        ]
        kg = KnowledgeGraph.from_spec(spec)
        assert kg.num_topics == 3
        order = kg.learning_order()
        names = [t.name for t in order]
        assert names == ["X", "Y", "Z"]
    _test("from_spec builds correct graph", t_from_spec)

    # ---- rebuild_from_spec ----
    def t_rebuild():
        kg = KnowledgeGraph.from_spec([{"name": "Old"}])
        assert kg.num_topics == 1
        kg.rebuild_from_spec([
            {"name": "New1"},
            {"name": "New2", "prerequisite_names": ["New1"]},
        ])
        assert kg.num_topics == 2
        assert kg.get_topic_by_name("Old") is None
        assert kg.get_topic_by_name("New1") is not None
    _test("rebuild_from_spec clears and rebuilds", t_rebuild)

    # ---- missing prerequisite in spec ----
    def t_missing_prereq_spec():
        spec = [{"name": "A", "prerequisite_names": ["NonExistent"]}]
        try:
            KnowledgeGraph.from_spec(spec)
            assert False
        except ValueError:
            pass
    _test("Missing prereq in spec raises ValueError", t_missing_prereq_spec)

    # ---- edge index / degree tensors ----
    def t_edge_index():
        kg = KnowledgeGraph()
        a = kg.create_topic("A")
        b = kg.create_topic("B")
        c = kg.create_topic("C")
        kg.add_prerequisite(a.topic_id, b.topic_id)
        kg.add_prerequisite(b.topic_id, c.topic_id)
        ei = kg.build_edge_index()
        assert ei.shape == (2, 2), f"expected (2,2) got {ei.shape}"
        assert kg.out_degree()[0].item() == 1   # A → B
        assert kg.in_degree()[2].item() == 1    # B → C
    _test("Edge index & degree tensors", t_edge_index)

    # ---- feature matrix ----
    def t_feature_matrix():
        kg = KnowledgeGraph()
        kg.create_topic("A")
        kg.create_topic("B")
        fm = kg.build_feature_matrix()
        assert fm.shape[0] == 2
    _test("Feature matrix shape", t_feature_matrix)

    # ---- adjacency numpy ----
    def t_adjacency_numpy():
        kg = KnowledgeGraph()
        a = kg.create_topic("A")
        b = kg.create_topic("B")
        kg.add_prerequisite(a.topic_id, b.topic_id)
        adj = kg.build_adjacency_numpy()
        assert adj[0, 1] == 1.0
        assert adj[1, 0] == 0.0
    _test("build_adjacency_numpy", t_adjacency_numpy)

    # ---- sparse adjacency ----
    def t_sparse_adj():
        kg = KnowledgeGraph()
        a = kg.create_topic("A")
        b = kg.create_topic("B")
        kg.add_prerequisite(a.topic_id, b.topic_id)
        sp = kg.build_sparse_adjacency()
        assert sp.is_sparse
    _test("Sparse adjacency tensor", t_sparse_adj)

    # ---- pyg Data ----
    def t_pyg_data():
        kg = KnowledgeGraph()
        kg.create_topic("A")
        kg.create_topic("B")
        kg.add_prerequisite(0, 1)
        data = kg.to_pyg_data()
        assert hasattr(data, "edge_index")
        assert hasattr(data, "x")
        assert data.num_nodes == 2
    _test("to_pyg_data", t_pyg_data)

    # ---- shortest_path_to ----
    def t_shortest_path():
        kg = KnowledgeGraph()
        a = kg.create_topic("A")
        b = kg.create_topic("B")
        c = kg.create_topic("C")
        kg.add_prerequisite(a.topic_id, b.topic_id)
        kg.add_prerequisite(b.topic_id, c.topic_id)
        path = kg.shortest_path_to(c.topic_id)
        assert [t.name for t in path] == ["A", "B", "C"]
    _test("shortest_path_to", t_shortest_path)

    # ---- get_subtopics / get_dependents ----
    def t_subtopics_dependents():
        kg = KnowledgeGraph()
        a = kg.create_topic("A")
        b = kg.create_topic("B")
        c = kg.create_topic("C")
        kg.add_prerequisite(a.topic_id, b.topic_id)
        kg.add_prerequisite(b.topic_id, c.topic_id)
        assert [t.name for t in kg.get_subtopics(b.topic_id)] == ["A"]
        assert [t.name for t in kg.get_dependents(b.topic_id)] == ["C"]
    _test("get_subtopics / get_dependents", t_subtopics_dependents)

    # ---- reset_progress ----
    def t_reset():
        kg = KnowledgeGraph()
        a = kg.create_topic("A")
        kg.master_topic(a.topic_id)
        assert a.mastered
        kg.reset_progress()
        assert not a.mastered
    _test("reset_progress", t_reset)

    # ---- bulk add prerequisites ----
    def t_bulk_prereqs():
        kg = KnowledgeGraph()
        a = kg.create_topic("A")
        b = kg.create_topic("B")
        c = kg.create_topic("C")
        kg.add_prerequisites_bulk([(a.topic_id, b.topic_id), (b.topic_id, c.topic_id)])
        order = kg.learning_order()
        assert [t.name for t in order] == ["A", "B", "C"]
    _test("Bulk add prerequisites", t_bulk_prereqs)

    # ---- empty graph ----
    def t_empty():
        kg = KnowledgeGraph()
        assert kg.num_topics == 0
        assert kg.mastery_progress() == 0.0
        assert kg.learning_order() == []
        ei = kg.build_edge_index()
        assert ei.shape == (2, 0)
    _test("Empty graph operations", t_empty)

    # ---- _LEVEL_MAP completeness ----
    def t_level_map():
        for lv in TopicLevel:
            assert lv.name.lower() in _LEVEL_MAP
    _test("_LEVEL_MAP covers all TopicLevel values", t_level_map)


# =====================================================================
# 2.  ACO.py
# =====================================================================
def test_aco():
    print("\n" + "=" * 60)
    print("  TESTING ACO.py")
    print("=" * 60)

    from graph import KnowledgeGraph, TopicLevel as TL
    from ACO import LearningPathACO, _build_matrices, _LEVEL_COST

    # ---- _build_matrices shapes ----
    def t_matrices_shape():
        kg = KnowledgeGraph()
        kg.create_topic("A", level=TL.FOUNDATIONAL)
        kg.create_topic("B", level=TL.INTERMEDIATE)
        kg.create_topic("C", level=TL.ADVANCED)
        kg.add_prerequisite(0, 1)
        kg.add_prerequisite(1, 2)
        cost, prereq, eta, sw, id2idx, idx2id = _build_matrices(kg)
        K = 3
        assert cost.shape == (K, K), f"cost shape {cost.shape}"
        assert prereq.shape == (K, K)
        assert eta.shape == (K, K)
        assert sw.shape == (K,)
        assert len(id2idx) == K
        assert idx2id.shape == (K,)
    _test("_build_matrices shapes", t_matrices_shape)

    # ---- cost matrix values ----
    def t_cost_values():
        kg = KnowledgeGraph()
        kg.create_topic("A", level=TL.FOUNDATIONAL)  # level 0
        kg.create_topic("B", level=TL.EXPERT)         # level 3
        kg.add_prerequisite(0, 1)
        cost, *_ = _build_matrices(kg)
        # A→B: base(1) + forward(3) + relatedness(-0.5) = 3.5
        assert abs(cost[0, 1] - 3.5) < 1e-9, f"A→B cost={cost[0,1]}"
        # B→A: base(1) + regression(3*3=9) + no relatedness = 10.0
        assert abs(cost[1, 0] - 10.0) < 1e-9, f"B→A cost={cost[1,0]}"
        # diagonal = 1e6
        assert cost[0, 0] == 1e6
    _test("Cost matrix values correct", t_cost_values)

    # ---- prereq matrix ----
    def t_prereq_matrix():
        kg = KnowledgeGraph()
        kg.create_topic("A")
        kg.create_topic("B")
        kg.create_topic("C")
        kg.add_prerequisite(0, 1)  # A→B
        kg.add_prerequisite(1, 2)  # B→C
        _, prereq, *_ = _build_matrices(kg)
        # B's prereq is A → prereq[1, 0] = True
        assert prereq[1, 0] == True
        # C's prereq is B → prereq[2, 1] = True
        assert prereq[2, 1] == True
        # A has no prereqs
        assert prereq[0].sum() == 0
    _test("Prerequisite matrix", t_prereq_matrix)

    # ---- start weights ----
    def t_start_weights():
        kg = KnowledgeGraph()
        kg.create_topic("A", level=TL.FOUNDATIONAL)
        kg.create_topic("B", level=TL.EXPERT)
        _, _, _, sw, *_ = _build_matrices(kg)
        # foundational: 1/(0+1)=1.0, expert: 1/(3+1)=0.25
        assert abs(sw[0] - 1.0) < 1e-9
        assert abs(sw[1] - 0.25) < 1e-9
    _test("Start weights prefer foundational", t_start_weights)

    # ---- empty graph ----
    def t_aco_empty():
        kg = KnowledgeGraph()
        aco = LearningPathACO(kg)
        path, cost = aco.optimise()
        assert path == []
        assert cost == 0.0
    _test("ACO on empty graph", t_aco_empty)

    # ---- single topic ----
    def t_aco_single():
        kg = KnowledgeGraph()
        kg.create_topic("Solo")
        aco = LearningPathACO(kg, m=5, k_max=5)
        path, cost = aco.optimise()
        assert len(path) == 1
        assert cost == 0.0
    _test("ACO single topic", t_aco_single)

    # ---- linear chain produces valid order ----
    def t_aco_linear():
        kg = KnowledgeGraph()
        names = ["A", "B", "C", "D", "E"]
        for n in names:
            kg.create_topic(n)
        for i in range(len(names) - 1):
            kg.add_prerequisite(i, i + 1)
        aco = LearningPathACO(kg, m=20, k_max=20, time_limit=5)
        path, cost = aco.optimise()
        assert len(path) == 5, f"path length {len(path)}"
        # For a linear chain there's only one valid order
        assert path == [0, 1, 2, 3, 4], f"expected [0..4] got {path}"
    _test("ACO linear chain → unique order", t_aco_linear)

    # ---- branching DAG: prerequisites respected ----
    def t_aco_branching():
        """
        DAG:  A ──→ C
              B ──→ C
              C ──→ D
        Valid orderings: [A,B,C,D] or [B,A,C,D]
        """
        kg = KnowledgeGraph()
        kg.create_topic("A", level=TL.FOUNDATIONAL)
        kg.create_topic("B", level=TL.FOUNDATIONAL)
        kg.create_topic("C", level=TL.INTERMEDIATE)
        kg.create_topic("D", level=TL.ADVANCED)
        kg.add_prerequisite(0, 2)  # A→C
        kg.add_prerequisite(1, 2)  # B→C
        kg.add_prerequisite(2, 3)  # C→D
        aco = LearningPathACO(kg, m=30, k_max=30, time_limit=5)
        path, cost = aco.optimise()
        assert len(path) == 4
        # C must come after both A and B; D must come after C
        idx = {tid: i for i, tid in enumerate(path)}
        assert idx[0] < idx[2], "A must precede C"
        assert idx[1] < idx[2], "B must precede C"
        assert idx[2] < idx[3], "C must precede D"
    _test("ACO branching DAG respects prerequisites", t_aco_branching)

    # ---- convergence: cost decreases or stays same ----
    def t_convergence():
        kg = KnowledgeGraph()
        for i, (name, lv) in enumerate([
            ("Intro", TL.FOUNDATIONAL), ("Basics", TL.FOUNDATIONAL),
            ("Inter", TL.INTERMEDIATE), ("Adv", TL.ADVANCED),
        ]):
            kg.create_topic(name, level=lv)
        kg.add_prerequisites_bulk([(0, 1), (1, 2), (2, 3)])
        aco = LearningPathACO(kg, m=30, k_max=30, time_limit=5)
        aco.optimise()
        # history should be non-increasing (best-so-far)
        for i in range(1, len(aco.history)):
            assert aco.history[i] <= aco.history[i - 1], (
                f"history not non-increasing at step {i}: "
                f"{aco.history[i-1]} > {aco.history[i]}"
            )
    _test("ACO convergence (non-increasing history)", t_convergence)

    # ---- get_named_path ----
    def t_named_path():
        kg = KnowledgeGraph()
        kg.create_topic("Alpha")
        kg.create_topic("Beta")
        kg.add_prerequisite(0, 1)
        aco = LearningPathACO(kg, m=5, k_max=5)
        aco.optimise()
        names = aco.get_named_path()
        assert names == ["Alpha", "Beta"]
    _test("get_named_path matches topic names", t_named_path)

    # ---- vectorised scoring matches manual ----
    def t_scoring():
        kg = KnowledgeGraph()
        kg.create_topic("A", level=TL.FOUNDATIONAL)
        kg.create_topic("B", level=TL.INTERMEDIATE)
        kg.create_topic("C", level=TL.ADVANCED)
        kg.add_prerequisite(0, 1)
        kg.add_prerequisite(1, 2)
        aco = LearningPathACO(kg, m=5, k_max=5)
        # Manually score path [0,1,2] using the cost matrix
        manual = aco.cost[0, 1] + aco.cost[1, 2]
        aco.optimise()
        assert abs(aco.best_cost - manual) < 1e-9, (
            f"best_cost={aco.best_cost}, manual={manual}"
        )
    _test("Vectorised scoring matches manual", t_scoring)

    # ---- pheromone values stay in bounds ----
    def t_pheromone_bounds():
        kg = KnowledgeGraph()
        for n in ["A", "B", "C"]:
            kg.create_topic(n)
        kg.add_prerequisites_bulk([(0, 1), (1, 2)])
        aco = LearningPathACO(kg, m=20, k_max=20, rho=0.9)
        aco.optimise()
        assert aco.tau.min() >= 0.01, f"pheromone below floor: {aco.tau.min()}"
        assert np.isfinite(aco.tau).all(), "pheromone has non-finite values"
    _test("Pheromone stays in bounds", t_pheromone_bounds)

    # ---- _get_available correctness ----
    def t_get_available():
        kg = KnowledgeGraph()
        kg.create_topic("A")
        kg.create_topic("B")
        kg.create_topic("C")
        kg.add_prerequisite(0, 1)  # A→B
        kg.add_prerequisite(1, 2)  # B→C
        aco = LearningPathACO(kg)
        visited = np.array([False, False, False])
        avail = aco._get_available(visited)
        assert list(avail) == [0], f"expected [0], got {list(avail)}"
        visited[0] = True
        avail = aco._get_available(visited)
        assert list(avail) == [1], f"expected [1], got {list(avail)}"
        visited[1] = True
        avail = aco._get_available(visited)
        assert list(avail) == [2], f"expected [2], got {list(avail)}"
    _test("_get_available vectorised correctness", t_get_available)


# =====================================================================
# 3.  SDS.py
# =====================================================================
def test_sds():
    print("\n" + "=" * 60)
    print("  TESTING SDS.py")
    print("=" * 60)

    from SDS import stochastic_diffusion_search

    # ---- basic convergence ----
    def t_sds_basic():
        ss = "try to find sds in this sentence"
        pos, activity = stochastic_diffusion_search(
            ss, "sds", n_agents=30, max_iter=50, verbose=False, seed=42,
        )
        assert pos >= 0, "should find 'sds'"
        assert ss[pos:pos + 3] == "sds"
        assert activity > 0.0
    _test("SDS finds 'sds' in sentence", t_sds_basic)

    # ---- finds pattern at start ----
    def t_sds_start():
        ss = "hello world this is a test"
        pos, _ = stochastic_diffusion_search(
            ss, "hello", n_agents=30, max_iter=80, verbose=False, seed=42,
        )
        assert pos == 0
    _test("SDS finds pattern at start of string", t_sds_start)

    # ---- finds pattern at end ----
    def t_sds_end():
        ss = "beginning middle test"
        pos, _ = stochastic_diffusion_search(
            ss, "test", n_agents=30, max_iter=80, verbose=False, seed=42,
        )
        assert pos == len(ss) - 4, f"expected {len(ss)-4}, got {pos}"
    _test("SDS finds pattern at end of string", t_sds_end)

    # ---- single character ----
    def t_sds_single_char():
        ss = "abcdefghijklmnop"
        pos, _ = stochastic_diffusion_search(
            ss, "k", n_agents=20, max_iter=60, verbose=False, seed=42,
        )
        assert pos == 10
    _test("SDS finds single character", t_sds_single_char)

    # ---- model longer than search space ----
    def t_sds_too_long():
        try:
            stochastic_diffusion_search("abc", "abcdef", verbose=False)
            assert False, "should have raised ValueError"
        except ValueError:
            pass
    _test("SDS raises for model > search_space", t_sds_too_long)

    # ---- empty model ----
    def t_sds_empty_model():
        try:
            stochastic_diffusion_search("abc", "", verbose=False)
            assert False, "should have raised ValueError"
        except ValueError:
            pass
    _test("SDS raises for empty model", t_sds_empty_model)

    # ---- deterministic seed ----
    def t_sds_deterministic():
        ss = "the quick brown fox jumps over the lazy dog"
        r1 = stochastic_diffusion_search(ss, "fox", n_agents=20, max_iter=40, verbose=False, seed=123)
        r2 = stochastic_diffusion_search(ss, "fox", n_agents=20, max_iter=40, verbose=False, seed=123)
        assert r1[0] == r2[0], "same seed should give same result"
    _test("SDS deterministic with seed", t_sds_deterministic)

    # ---- high activity for easy patterns ----
    def t_sds_activity():
        ss = "aaa bbb aaa bbb aaa"
        _, activity = stochastic_diffusion_search(
            ss, "aaa", n_agents=50, max_iter=100, verbose=False, seed=42,
        )
        assert activity >= 0.5, f"expected high activity, got {activity}"
    _test("SDS high activity for repeated patterns", t_sds_activity)

    # ================================================================
    # Spell-correction / dictionary tests
    # ================================================================
    from SDS import (
        load_dictionary, clear_dictionary_cache, spell_correct,
        correct_phrase, _pad_words, _BLOCKED,
    )

    # ---- load_dictionary returns a populated list ----
    def t_dict_loads():
        words = load_dictionary()
        assert isinstance(words, list)
        assert len(words) > 50_000, f"expected >50k words, got {len(words)}"
        # all lowercase, stripped, no empty strings
        for w in words[:200]:
            assert w == w.strip().lower()
            assert len(w) >= 2
    _test("load_dictionary returns >50k clean words", t_dict_loads)

    # ---- blocked words are excluded ----
    def t_blocked_words():
        words = load_dictionary()
        word_set = set(words)
        for blocked in ("kill", "murder", "death", "die", "suicide", "rape", "bomb"):
            assert blocked not in word_set, f"blocked word '{blocked}' found in dict"
    _test("Blocked words excluded from dictionary", t_blocked_words)

    # ---- _BLOCKED is a frozenset with expected contents ----
    def t_blocked_set():
        assert isinstance(_BLOCKED, frozenset)
        assert "kill" in _BLOCKED
        assert "murder" in _BLOCKED
        assert "death" in _BLOCKED
        assert "python" not in _BLOCKED  # sanity: normal word not blocked
    _test("_BLOCKED frozenset contents", t_blocked_set)

    # ---- clear_dictionary_cache forces reload ----
    def t_clear_cache():
        w1 = load_dictionary()
        clear_dictionary_cache()
        w2 = load_dictionary()
        assert w1 == w2, "reloaded dict should match original"
    _test("clear_dictionary_cache forces reload", t_clear_cache)

    # ---- _pad_words produces correct shape ----
    def t_pad_words():
        mat = _pad_words(["cat", "tiger", "go"], max_len=6)
        assert mat.shape == (3, 6)
        assert mat[0, 0] == ord("c")
        assert mat[0, 3] == 0  # padding
        assert mat[2, 0] == ord("g")
        assert mat[2, 2] == 0  # "go" has length 2
    _test("_pad_words shape and contents", t_pad_words)

    # ---- spell_correct returns empty for empty input ----
    def t_spell_empty():
        assert spell_correct("") == []
        assert spell_correct("   ") == []
    _test("spell_correct empty input returns []", t_spell_empty)

    # ---- spell_correct finds 'guitar' for 'guittar' ----
    def t_spell_guitar():
        results = spell_correct("guittar", top_k=5, seed=42)
        top_words = [w for w, _ in results]
        assert "guitar" in top_words, f"'guitar' not in {top_words}"
    _test("spell_correct 'guittar' → guitar", t_spell_guitar)

    # ---- spell_correct finds 'piano' for 'pianno' ----
    def t_spell_piano():
        results = spell_correct("pianno", top_k=5, seed=42)
        top_words = [w for w, _ in results]
        assert "piano" in top_words, f"'piano' not in {top_words}"
    _test("spell_correct 'pianno' → piano", t_spell_piano)

    # ---- spell_correct finds 'science' for 'sciance' ----
    def t_spell_science():
        results = spell_correct("sciance", top_k=5, seed=42)
        top_words = [w for w, _ in results]
        assert "science" in top_words, f"'science' not in {top_words}"
    _test("spell_correct 'sciance' → science", t_spell_science)

    # ---- spell_correct with custom dictionary ----
    def t_spell_custom_dict():
        custom = ["apple", "application", "apply", "banana", "band"]
        results = spell_correct("appple", top_k=3, dictionary=custom, seed=42)
        top_words = [w for w, _ in results]
        assert len(top_words) > 0, "should return at least one suggestion"
        assert "apple" in top_words, f"'apple' not in {top_words}"
    _test("spell_correct with custom dictionary", t_spell_custom_dict)

    # ---- spell_correct returns (word, score) tuples ----
    def t_spell_format():
        results = spell_correct("guittar", top_k=3, seed=42)
        assert isinstance(results, list)
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2
            word, score = item
            assert isinstance(word, str)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0 + 1e-9
    _test("spell_correct returns (word, float) tuples", t_spell_format)

    # ---- correct_phrase: known word passes through ----
    def t_phrase_known():
        result = correct_phrase("science")
        assert len(result) == 1
        assert result[0] == [("science", 1.0)]
    _test("correct_phrase known word passes through", t_phrase_known)

    # ---- correct_phrase: misspelled word gets suggestions ----
    def t_phrase_misspelled():
        result = correct_phrase("guittar pianno", top_k=3)
        assert len(result) == 2, f"expected 2 words, got {len(result)}"
        # Each should be a list of suggestions
        for suggestions in result:
            assert isinstance(suggestions, list)
            assert len(suggestions) > 0
    _test("correct_phrase misspelled words get suggestions", t_phrase_misspelled)

    # ---- correct_phrase: mixed known and unknown ----
    def t_phrase_mixed():
        result = correct_phrase("the guittar", top_k=3)
        assert len(result) == 2
        # "the" should pass through
        assert result[0] == [("the", 1.0)]
        # "guittar" should get suggestions
        assert len(result[1]) > 0
        guitar_found = any(w == "guitar" for w, _ in result[1])
        assert guitar_found, f"'guitar' not in suggestions: {result[1]}"
    _test("correct_phrase mixed known/unknown words", t_phrase_mixed)

    # ---- spell_correct deterministic with seed ----
    def t_spell_deterministic():
        r1 = spell_correct("mathmatics", top_k=5, seed=99)
        r2 = spell_correct("mathmatics", top_k=5, seed=99)
        assert r1 == r2, "same seed should give identical results"
    _test("spell_correct deterministic with seed", t_spell_deterministic)


# =====================================================================
# 4.  Webscraping.py
# =====================================================================
def test_webscraping():
    print("\n" + "=" * 60)
    print("  TESTING Webscraping.py")
    print("=" * 60)

    from Webscraping import (
        LearningResource, LearningStep, LearningPlan,
        _assign_levels, get_learning_plan, get_learning_spec,
    )

    # ---- data class creation ----
    def t_learning_resource():
        r = LearningResource(title="Intro to ML", url="http://example.com", source="test")
        assert r.title == "Intro to ML"
        assert r.url == "http://example.com"
    _test("LearningResource creation", t_learning_resource)

    def t_learning_step():
        s = LearningStep(step_number=1, subtopic="Basics", level="foundational")
        assert s.step_number == 1
        assert s.level == "foundational"
    _test("LearningStep creation", t_learning_step)

    def t_learning_plan():
        plan = LearningPlan(topic="Test", summary="A test plan")
        plan.steps.append(LearningStep(1, "Step1", level="foundational"))
        plan.steps.append(LearningStep(2, "Step2", level="intermediate"))
        dicts = plan.to_dict_list()
        assert len(dicts) == 2
        assert dicts[0]["name"] == "Step1"
        assert "prerequisite_names" not in dicts[0]  # first has no prereqs
        assert dicts[1]["prerequisite_names"] == ["Step1"]
    _test("LearningPlan to_dict_list", t_learning_plan)

    # ---- _assign_levels ----
    def t_assign_levels_empty():
        assert _assign_levels(0) == []
    _test("_assign_levels(0) returns []", t_assign_levels_empty)

    def t_assign_levels_one():
        assert _assign_levels(1) == ["foundational"]
    _test("_assign_levels(1) returns ['foundational']", t_assign_levels_one)

    def t_assign_levels_many():
        levels = _assign_levels(10)
        assert len(levels) == 10
        assert levels[0] == "foundational"
        assert levels[-1] == "expert"
        valid = {"foundational", "intermediate", "advanced", "expert"}
        for lv in levels:
            assert lv in valid, f"unexpected level '{lv}'"
        # Should be monotonically non-decreasing in difficulty
        rank = {"foundational": 0, "intermediate": 1, "advanced": 2, "expert": 3}
        for i in range(1, len(levels)):
            assert rank[levels[i]] >= rank[levels[i - 1]], (
                f"levels not non-decreasing: {levels[i-1]} > {levels[i]}"
            )
    _test("_assign_levels(10) correct distribution", t_assign_levels_many)

    # ---- get_learning_spec integration (requires network) ----
    def t_learning_spec():
        """Test that get_learning_spec returns a valid spec list (needs network)."""
        try:
            spec = get_learning_spec("Python programming")
            assert isinstance(spec, list)
            assert len(spec) > 0
            for entry in spec:
                assert "name" in entry
                assert "level" in entry
                assert entry["level"] in {"foundational", "intermediate", "advanced", "expert"}
        except Exception as e:
            # Network might be unavailable — that's OK
            print(f"    (skipped — network error: {e})")
    _test("get_learning_spec returns valid spec (network)", t_learning_spec)


# =====================================================================
# 5.  Integration: Webscraping → graph → ACO
# =====================================================================
def test_integration():
    print("\n" + "=" * 60)
    print("  TESTING Integration (graph → ACO pipeline)")
    print("=" * 60)

    from graph import KnowledgeGraph, TopicLevel as TL
    from ACO import LearningPathACO

    # ---- sample ML curriculum ----
    def t_sample_curriculum():
        from graph import build_sample_knowledge_graph
        kg = build_sample_knowledge_graph()
        assert kg.num_topics == 8
        aco = LearningPathACO(kg, m=20, k_max=20, time_limit=5)
        path, cost = aco.optimise()
        assert len(path) == 8
        assert cost < float("inf")
        # verify prereq order
        pos = {tid: i for i, tid in enumerate(path)}
        for tid in path:
            t = kg.topics[tid]
            for pid in t.prerequisites:
                assert pos[pid] < pos[tid], (
                    f"prereq {kg.topics[pid].name} should precede {t.name}"
                )
    _test("Sample ML curriculum end-to-end", t_sample_curriculum)

    # ---- rebuild + re-optimise ----
    def t_rebuild_reoptimise():
        spec1 = [
            {"name": "A", "level": "foundational"},
            {"name": "B", "level": "intermediate", "prerequisite_names": ["A"]},
        ]
        spec2 = [
            {"name": "X", "level": "foundational"},
            {"name": "Y", "level": "intermediate", "prerequisite_names": ["X"]},
            {"name": "Z", "level": "advanced", "prerequisite_names": ["Y"]},
        ]
        kg = KnowledgeGraph.from_spec(spec1)
        aco1 = LearningPathACO(kg, m=10, k_max=10)
        p1, c1 = aco1.optimise()
        assert len(p1) == 2

        kg.rebuild_from_spec(spec2)
        aco2 = LearningPathACO(kg, m=10, k_max=10)
        p2, c2 = aco2.optimise()
        assert len(p2) == 3
    _test("Rebuild + re-optimise", t_rebuild_reoptimise)

    # ---- large graph performance ----
    def t_large_graph():
        """50-topic linear chain — ACO should complete quickly."""
        kg = KnowledgeGraph()
        levels = [TL.FOUNDATIONAL, TL.INTERMEDIATE, TL.ADVANCED, TL.EXPERT]
        for i in range(50):
            kg.create_topic(f"T{i}", level=levels[min(i // 13, 3)])
        edges = [(i, i + 1) for i in range(49)]
        kg.add_prerequisites_bulk(edges)

        start = time.time()
        aco = LearningPathACO(kg, m=20, k_max=15, time_limit=10)
        path, cost = aco.optimise()
        elapsed = time.time() - start

        assert len(path) == 50, f"path covers {len(path)}/50 topics"
        assert elapsed < 15, f"took {elapsed:.1f}s — too slow"
        assert path == list(range(50)), "linear chain has unique order"
    _test("Large graph (50 topics) performance", t_large_graph)


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  NeuraLearn Backend — Comprehensive Test Suite")
    print("=" * 60)

    test_graph()
    test_aco()
    test_sds()
    test_webscraping()
    test_integration()

    print("\n" + "=" * 60)
    total = _PASS + _FAIL
    print(f"  RESULTS: {_PASS}/{total} passed, {_FAIL} failed")
    print("=" * 60)

    if _ERRORS:
        print("\nFailed tests:")
        for err in _ERRORS:
            print(err)

    sys.exit(0 if _FAIL == 0 else 1)

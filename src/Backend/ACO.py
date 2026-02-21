import logging
import random
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from graph import KnowledgeGraph, TopicLevel

"""
Ant Colony Optimization for Learning Path Discovery
----------------------------------------------------
Given a KnowledgeGraph (DAG of topics with prerequisite edges), find the
optimal ordering of topics to study. "Optimal" means:

  1. All prerequisite constraints are satisfied (hard constraint).
  2. Difficulty transitions are smooth  (FOUNDATIONAL → INTERMEDIATE → …).
  3. Closely related topics are studied together (locality bonus).
  4. The total "cognitive cost" of the path is minimised.

The ACO places pheromone on (topic_i → topic_j) transitions that appear in
low-cost paths, guiding future ants toward better orderings.
"""

# ---------------------------------------------------------------------------
# Numeric difficulty for smooth-transition scoring
# ---------------------------------------------------------------------------
_LEVEL_COST: dict[TopicLevel, int] = {
    TopicLevel.FOUNDATIONAL: 0,
    TopicLevel.INTERMEDIATE: 1,
    TopicLevel.ADVANCED: 2,
    TopicLevel.EXPERT: 3,
}


# ---------------------------------------------------------------------------
# Cost matrix builder
# ---------------------------------------------------------------------------
def build_cost_matrix(kg: KnowledgeGraph) -> np.ndarray:
    """
    Build an N×N cost matrix where cost[i][j] represents the cognitive
    cost of studying topic j immediately after topic i.

    Cost components:
      • difficulty_jump  — penalises big level jumps (e.g. FOUNDATIONAL → EXPERT)
      • relatedness      — bonus (negative cost) if j is a direct dependent of i
      • base_cost        — small constant so every transition has non-zero cost
    """
    n = max(kg.topics) + 1 if kg.topics else 0
    cost = np.full((n, n), 1e6)  # default: unreachable

    ids = sorted(kg.topics.keys())
    for i in ids:
        ti = kg.topics[i]
        li = _LEVEL_COST[ti.level]
        for j in ids:
            if i == j:
                continue
            tj = kg.topics[j]
            lj = _LEVEL_COST[tj.level]

            # difficulty jump penalty  (going backwards is extra costly)
            diff_jump = (lj - li)
            if diff_jump < 0:
                difficulty_cost = abs(diff_jump) * 3.0   # penalise regression
            else:
                difficulty_cost = diff_jump * 1.0         # forward is natural

            # relatedness bonus: if j is a direct dependent of i, cheaper
            relatedness = -0.5 if j in ti.unlocks else 0.0

            # base cost
            base = 1.0

            cost[i][j] = base + difficulty_cost + relatedness

    return cost


# ---------------------------------------------------------------------------
# ACO for Learning Paths
# ---------------------------------------------------------------------------
class LearningPathACO:
    """
    Ant Colony Optimization that finds the best ordering of topics in a
    KnowledgeGraph, respecting prerequisite constraints and minimising
    cognitive cost (smooth difficulty progression, related-topic locality).

    Parameters
    ----------
    kg          : KnowledgeGraph to optimise over
    m           : number of ants per iteration
    k_max       : maximum iterations
    alpha       : pheromone importance (higher → follow the colony)
    beta        : heuristic importance (higher → follow cost matrix)
    rho         : pheromone evaporation rate  (0 = no evaporation, 1 = full)
    Q           : pheromone deposit constant
    time_limit  : wall-clock seconds before early stop
    """

    def __init__(self, kg: KnowledgeGraph, **kwargs):
        self.kg = kg
        self.topic_ids: list[int] = sorted(kg.topics.keys())
        self.n = max(self.topic_ids) + 1 if self.topic_ids else 0
        self.num_topics = len(self.topic_ids)

        # build cost & heuristic matrices
        self.cost_matrix = build_cost_matrix(kg)
        self.eta = 1.0 / (self.cost_matrix + 1e-10)  # heuristic: inverse cost

        # ACO hyperparameters
        self.m = kwargs.get("m", 50)
        self.k_max = kwargs.get("k_max", 80)
        self.alpha = kwargs.get("alpha", 1.0)
        self.beta = kwargs.get("beta", 3.0)
        self.rho = kwargs.get("rho", 0.85)
        self.Q = kwargs.get("Q", 10.0)
        self.time_limit = kwargs.get("time_limit", 10)

        # pheromone matrix (initialised uniformly)
        self.tau = np.full((self.n, self.n), 1.0)

        # results tracking
        self.history: list[float] = []
        self.best_path: list[int] = []
        self.best_cost: float = float("inf")

    # ---- prerequisite-aware candidate set ---------------------------------
    def _get_available(self, visited: set[int]) -> list[int]:
        """
        Return topic_ids that are not yet visited AND whose prerequisites
        have ALL been visited (i.e. "mastered" in this ant's walk).
        """
        available = []
        for tid in self.topic_ids:
            if tid in visited:
                continue
            topic = self.kg.topics[tid]
            if topic.prerequisites.issubset(visited):
                available.append(tid)
        return available

    # ---- attractiveness ----------------------------------------------------
    def _compute_attractiveness(self) -> None:
        """Precompute attractiveness A = tau^alpha * eta^beta."""
        self.A = (self.tau ** self.alpha) * (self.eta ** self.beta)

    # ---- single ant walk ---------------------------------------------------
    def _ant_walk(self) -> tuple[list[int], float]:
        """
        One ant builds a complete learning path by choosing topics one at a
        time, respecting prerequisites. Returns (path, total_cost).
        """
        visited: set[int] = set()
        path: list[int] = []

        while len(path) < self.num_topics:
            candidates = self._get_available(visited)
            if not candidates:
                break  # stuck (shouldn't happen in a valid DAG)

            if not path:
                # first topic: use only heuristic (no previous node)
                weights = [self.eta[0][j] if j < self.n else 1.0 for j in candidates]
            else:
                prev = path[-1]
                weights = [self.A[prev][j] for j in candidates]

            # ensure all weights are positive
            min_w = min(weights)
            if min_w <= 0:
                weights = [w - min_w + 1e-10 for w in weights]

            chosen = random.choices(candidates, weights=weights, k=1)[0]
            path.append(chosen)
            visited.add(chosen)

        # score the path
        total_cost = self._score_path(path)
        return path, total_cost

    # ---- scoring -----------------------------------------------------------
    def _score_path(self, path: list[int]) -> float:
        """Total cognitive cost of a learning path."""
        if len(path) < 2:
            return 0.0
        cost = 0.0
        for i in range(len(path) - 1):
            cost += self.cost_matrix[path[i]][path[i + 1]]
        return cost

    # ---- pheromone deposit --------------------------------------------------
    def _deposit_pheromone(self, path: list[int], cost: float) -> None:
        """Deposit pheromone on edges in this path, inversely proportional to cost."""
        deposit = self.Q / max(cost, 1e-10)
        for i in range(len(path) - 1):
            self.tau[path[i]][path[i + 1]] += deposit

    # ---- main loop ----------------------------------------------------------
    def optimise(self) -> tuple[list[int], float]:
        """
        Run the ACO and return (best_path, best_cost).
        best_path is a list of topic_ids in optimal learning order.
        """
        start_time = time.time()

        for iteration in range(self.k_max):
            self._compute_attractiveness()

            # evaporate pheromone
            self.tau *= (1 - self.rho)
            # clamp so pheromone doesn't vanish completely
            self.tau = np.clip(self.tau, 0.01, None)

            for _ in range(self.m):
                path, cost = self._ant_walk()

                # deposit pheromone
                self._deposit_pheromone(path, cost)

                # track best
                if cost < self.best_cost and len(path) == self.num_topics:
                    self.best_cost = cost
                    self.best_path = path
                    logging.info(
                        "Iteration %d: better path found (cost=%.2f)",
                        iteration, cost,
                    )

            self.history.append(self.best_cost)

            # time limit
            if time.time() - start_time > self.time_limit:
                logging.info("Time limit reached after %d iterations.", iteration + 1)
                break

        return self.best_path, self.best_cost

    # ---- results ------------------------------------------------------------
    def get_named_path(self) -> list[str]:
        """Return the best path as a list of topic names."""
        return [self.kg.topics[tid].name for tid in self.best_path]

    def print_result(self) -> None:
        """Pretty-print the optimal learning path."""
        print(f"\n{'='*60}")
        print(f"  ACO Optimal Learning Path  (cost: {self.best_cost:.2f})")
        print(f"{'='*60}")
        for step, tid in enumerate(self.best_path, 1):
            t = self.kg.topics[tid]
            prereqs = ", ".join(self.kg.topics[p].name for p in t.prerequisites) or "none"
            print(f"  {step:>2}. {t.name:<40} [{t.level.name}]  prereqs: {prereqs}")
        print(f"{'='*60}\n")

    def plot_convergence(self, save_path: str | None = None) -> None:
        """Plot cost over iterations to visualise ACO convergence."""
        if not self.history:
            print("No history to plot — run optimise() first.")
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(self.history) + 1), self.history, marker="o", markersize=3)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best Path Cost")
        ax.set_title("ACO Convergence — Learning Path Optimisation")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Saved convergence plot to {save_path}")
        plt.show()


# ---------------------------------------------------------------------------
# Convenience: scrape + graph + ACO in one call
# ---------------------------------------------------------------------------
def find_optimal_learning_path(
    topic: str,
    **aco_kwargs,
) -> tuple[list[str], float, LearningPathACO]:
    """
    End-to-end: scrape a topic → build knowledge graph → run ACO → return
    the optimal learning path as a list of topic names.

    Returns (named_path, cost, aco_instance).
    """
    from Webscraping import get_learning_spec

    spec = get_learning_spec(topic)
    kg = KnowledgeGraph.from_spec(spec)
    aco = LearningPathACO(kg, **aco_kwargs)
    path, cost = aco.optimise()
    return aco.get_named_path(), cost, aco


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # --- Build a sample knowledge graph ------------------------------------
    from graph import TopicLevel as TL

    kg = KnowledgeGraph()
    overview     = kg.create_topic("Overview",                  level=TL.FOUNDATIONAL)
    history      = kg.create_topic("History",                   level=TL.FOUNDATIONAL)
    communication = kg.create_topic("In communication",         level=TL.FOUNDATIONAL)
    manuscripts  = kg.create_topic("In manuscripts",            level=TL.FOUNDATIONAL)
    science      = kg.create_topic("In science",                level=TL.INTERMEDIATE)
    expression   = kg.create_topic("As artistic expression",    level=TL.INTERMEDIATE)
    artists      = kg.create_topic("Notable artists",           level=TL.INTERMEDIATE)
    materials    = kg.create_topic("Materials",                  level=TL.INTERMEDIATE)
    technique    = kg.create_topic("Technique",                  level=TL.ADVANCED)
    tone         = kg.create_topic("Tone",                       level=TL.ADVANCED)
    form         = kg.create_topic("Form and proportion",        level=TL.ADVANCED)
    perspective  = kg.create_topic("Perspective",                level=TL.EXPERT)
    composition  = kg.create_topic("Composition",                level=TL.EXPERT)
    process      = kg.create_topic("Process",                    level=TL.EXPERT)

    # prerequisite edges
    kg.add_prerequisite(overview.topic_id, history.topic_id)
    kg.add_prerequisite(history.topic_id, communication.topic_id)
    kg.add_prerequisite(communication.topic_id, manuscripts.topic_id)
    kg.add_prerequisite(manuscripts.topic_id, science.topic_id)
    kg.add_prerequisite(science.topic_id, expression.topic_id)
    kg.add_prerequisite(expression.topic_id, artists.topic_id)
    kg.add_prerequisite(artists.topic_id, materials.topic_id)
    kg.add_prerequisite(materials.topic_id, technique.topic_id)
    kg.add_prerequisite(technique.topic_id, tone.topic_id)
    kg.add_prerequisite(tone.topic_id, form.topic_id)
    kg.add_prerequisite(form.topic_id, perspective.topic_id)
    kg.add_prerequisite(perspective.topic_id, composition.topic_id)
    kg.add_prerequisite(composition.topic_id, process.topic_id)

    print("--- Knowledge Graph ---")
    kg.print_curriculum()

    print("\n--- Running ACO ---")
    aco = LearningPathACO(kg, m=50, k_max=60, time_limit=8)
    best_path, best_cost = aco.optimise()

    aco.print_result()

    print(f"Convergence: started at {aco.history[0]:.2f}, ended at {aco.history[-1]:.2f}")
    print(f"Improvement: {((aco.history[0] - aco.history[-1]) / aco.history[0] * 100):.1f}%")

    # --- Also demo the end-to-end convenience function ---------------------
    print("\n\n--- End-to-end: find_optimal_learning_path('Piano') ---")
    named_path, cost, _ = find_optimal_learning_path("Piano", m=30, k_max=40)
    print(f"Cost: {cost:.2f}")
    for i, name in enumerate(named_path, 1):
        print(f"  {i}. {name}")

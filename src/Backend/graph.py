import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree
from typing import Optional
from collections import deque
from enum import Enum
from dataclasses import dataclass, field

"""
NeuraLearn Knowledge Graph  (optimised for repeated creation)
-------------------------------------------------------------
Designed so the graph can be torn down and rebuilt many times cheaply:

  • Topic uses __slots__ → ~40 % less memory per node, faster attr access
  • Adjacency stored as sets → O(1) duplicate-edge checks
  • Bulk add methods → single cache invalidation per batch
  • Topological-sort result is cached and auto-invalidated
  • Name index dict → O(1) lookup by name instead of O(n) scan
  • clear() wipes the graph in-place for reuse (no new object allocation)
  • rebuild_from_spec() builds from a lightweight list-of-dicts spec
"""


# ---------------------------------------------------------------------------
# Topic difficulty / type labels
# ---------------------------------------------------------------------------
class TopicLevel(Enum):
	FOUNDATIONAL = 0
	INTERMEDIATE = 1
	ADVANCED = 2
	EXPERT = 3

# Mapping from string → enum for spec-based construction
_LEVEL_MAP: dict[str, TopicLevel] = {
	"foundational": TopicLevel.FOUNDATIONAL,
	"intermediate": TopicLevel.INTERMEDIATE,
	"advanced": TopicLevel.ADVANCED,
	"expert": TopicLevel.EXPERT,
}


# ---------------------------------------------------------------------------
# Topic node  (__slots__ for speed & memory)
# ---------------------------------------------------------------------------
class Topic:
	"""A single topic or subtopic in the knowledge graph."""

	__slots__ = (
		"topic_id", "name", "description", "level",
		"features", "mastered", "prerequisites", "unlocks",
	)

	def __init__(
		self,
		topic_id: int,
		name: str,
		description: str = "",
		level: TopicLevel = TopicLevel.FOUNDATIONAL,
		features: torch.Tensor | None = None,
	) -> None:
		self.topic_id = topic_id
		self.name = name
		self.description = description
		self.level = level
		self.features = features if features is not None else torch.zeros(1)
		self.mastered: bool = False
		self.prerequisites: set[int] = set()   # topic_ids this depends on  (in-edges)
		self.unlocks: set[int] = set()         # topic_ids this unlocks     (out-edges)

	def __repr__(self) -> str:
		status = "mastered" if self.mastered else "locked"
		return f"Topic({self.topic_id}, '{self.name}', {self.level.name}, {status})"


# ---------------------------------------------------------------------------
# Lightweight spec type used by rebuild_from_spec()
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class TopicSpec:
	"""Plain-data description of a topic — no torch, no graph pointers."""
	name: str
	description: str = ""
	level: str = "foundational"                 # key into _LEVEL_MAP
	prerequisite_names: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# KnowledgeGraph  (optimised for repeated build / tear-down)
# ---------------------------------------------------------------------------
class KnowledgeGraph:
	"""
	Directed acyclic graph of Topics.

	Edges go from prerequisite (subtopic) → dependent (topic).
	A topic is *unlocked* only when ALL its prerequisites are mastered.

	Optimised for repeated creation:
	  - call clear() then re-populate, or
	  - use the class method rebuild_from_spec() with a plain list of dicts.
	"""

	__slots__ = (
		"device", "topics", "_next_id",
		"_name_index",
		"_edge_index_cache", "_degree_cache", "_topo_cache",
	)

	def __init__(
		self,
		device: torch.device | str | None = None,
	) -> None:
		self.device = torch.device(device) if device is not None else torch.device("cpu")
		self.topics: dict[int, Topic] = {}
		self._next_id: int = 0
		self._name_index: dict[str, int] = {}          # lower(name) → topic_id
		self._edge_index_cache: torch.Tensor | None = None
		self._degree_cache: dict[str, torch.Tensor] = {}
		self._topo_cache: list[Topic] | None = None     # cached topo sort

	# ---- helpers -----------------------------------------------------------
	@property
	def num_topics(self) -> int:
		return len(self.topics)

	def _invalidate_cache(self) -> None:
		self._edge_index_cache = None
		self._degree_cache.clear()
		self._topo_cache = None

	# ---- clear & reuse (avoids allocating a new KnowledgeGraph) ------------
	def clear(self) -> None:
		"""Wipe all topics and edges so the same object can be reused."""
		self.topics.clear()
		self._name_index.clear()
		self._next_id = 0
		self._invalidate_cache()

	# ---- node helpers ------------------------------------------------------
	def add_topic(self, topic: Topic) -> None:
		"""Add a Topic node to the graph."""
		if topic.topic_id in self.topics:
			raise ValueError(f"topic_id {topic.topic_id} already exists")
		self.topics[topic.topic_id] = topic
		self._name_index[topic.name.lower()] = topic.topic_id
		self._next_id = max(self._next_id, topic.topic_id + 1)
		self._invalidate_cache()

	def add_topics_bulk(self, topics: list[Topic]) -> None:
		"""Add many topics at once — only one cache invalidation."""
		for t in topics:
			if t.topic_id in self.topics:
				raise ValueError(f"topic_id {t.topic_id} already exists")
			self.topics[t.topic_id] = t
			self._name_index[t.name.lower()] = t.topic_id
			self._next_id = max(self._next_id, t.topic_id + 1)
		self._invalidate_cache()

	def create_topic(
		self,
		name: str,
		description: str = "",
		level: TopicLevel = TopicLevel.FOUNDATIONAL,
		features: torch.Tensor | None = None,
	) -> Topic:
		"""Create a new Topic with an auto-assigned id and add it."""
		t = Topic(
			topic_id=self._next_id,
			name=name,
			description=description,
			level=level,
			features=features,
		)
		self.add_topic(t)
		return t

	def get_topic(self, topic_id: int) -> Topic:
		try:
			return self.topics[topic_id]
		except KeyError:
			raise KeyError(f"topic_id {topic_id} not found") from None

	def get_topic_by_name(self, name: str) -> Topic | None:
		"""O(1) lookup by name (case-insensitive)."""
		tid = self._name_index.get(name.lower())
		return self.topics[tid] if tid is not None else None

	# ---- edge helpers (prerequisite relationships) -------------------------
	def add_prerequisite(self, subtopic_id: int, topic_id: int) -> None:
		"""
		Add edge: subtopic → topic.
		The learner must master *subtopic* before *topic* is unlocked.
		"""
		if subtopic_id not in self.topics or topic_id not in self.topics:
			raise ValueError("both subtopic_id and topic_id must exist in the graph")
		if subtopic_id == topic_id:
			raise ValueError("a topic cannot be a prerequisite of itself")
		sub = self.topics[subtopic_id]
		dep = self.topics[topic_id]
		if topic_id in sub.unlocks:          # O(1) set check
			return
		sub.unlocks.add(topic_id)
		dep.prerequisites.add(subtopic_id)
		self._invalidate_cache()

	def add_prerequisites(self, topic_id: int, prerequisite_ids: list[int]) -> None:
		"""Add many prerequisites for one topic — single cache invalidation."""
		dep = self.topics[topic_id]
		changed = False
		for pid in prerequisite_ids:
			if pid not in self.topics:
				raise ValueError(f"prerequisite topic_id {pid} not found")
			if pid == topic_id:
				raise ValueError("a topic cannot be a prerequisite of itself")
			sub = self.topics[pid]
			if topic_id not in sub.unlocks:
				sub.unlocks.add(topic_id)
				dep.prerequisites.add(pid)
				changed = True
		if changed:
			self._invalidate_cache()

	def add_prerequisites_bulk(self, edges: list[tuple[int, int]]) -> None:
		"""
		Add many prerequisite edges at once: [(subtopic_id, topic_id), ...].
		Only invalidates caches once at the end.
		"""
		changed = False
		for subtopic_id, topic_id in edges:
			sub = self.topics[subtopic_id]
			dep = self.topics[topic_id]
			if topic_id not in sub.unlocks:
				sub.unlocks.add(topic_id)
				dep.prerequisites.add(subtopic_id)
				changed = True
		if changed:
			self._invalidate_cache()

	# ---- mastery & progress -----------------------------------------------
	def is_unlocked(self, topic_id: int) -> bool:
		"""True when ALL prerequisites are mastered."""
		topic = self.topics[topic_id]
		return all(self.topics[pid].mastered for pid in topic.prerequisites)

	def master_topic(self, topic_id: int) -> bool:
		"""Mark mastered if unlocked. Returns success bool."""
		if not self.is_unlocked(topic_id):
			return False
		self.topics[topic_id].mastered = True
		return True

	def reset_progress(self) -> None:
		"""Reset mastery on all topics (keeps graph structure)."""
		for t in self.topics.values():
			t.mastered = False

	def get_mastered(self) -> list[Topic]:
		return [t for t in self.topics.values() if t.mastered]

	def get_available(self) -> list[Topic]:
		"""Topics unlocked but not yet mastered."""
		return [
			t for t in self.topics.values()
			if not t.mastered and self.is_unlocked(t.topic_id)
		]

	def get_locked(self) -> list[Topic]:
		return [
			t for t in self.topics.values()
			if not t.mastered and not self.is_unlocked(t.topic_id)
		]

	def mastery_progress(self) -> float:
		if not self.topics:
			return 0.0
		return sum(1 for t in self.topics.values() if t.mastered) / len(self.topics)

	# ---- learning path (cached topological order) --------------------------
	def learning_order(self) -> list[Topic]:
		"""
		Valid learning order via Kahn's algorithm.  Cached until the graph
		structure changes; repeated calls are O(1).
		"""
		if self._topo_cache is not None:
			return self._topo_cache

		in_deg: dict[int, int] = {tid: 0 for tid in self.topics}
		for t in self.topics.values():
			for uid in t.unlocks:
				in_deg[uid] += 1

		queue = deque(tid for tid, d in in_deg.items() if d == 0)
		order: list[Topic] = []

		while queue:
			tid = queue.popleft()
			topic = self.topics[tid]
			order.append(topic)
			for uid in topic.unlocks:
				in_deg[uid] -= 1
				if in_deg[uid] == 0:
					queue.append(uid)

		if len(order) != len(self.topics):
			raise ValueError("knowledge graph contains a cycle — no valid learning order")

		self._topo_cache = order
		return order

	def shortest_path_to(self, target_id: int) -> list[Topic]:
		"""Min topics (in order) needed to unlock *target_id*, skipping mastered."""
		needed: set[int] = set()
		queue = deque([target_id])
		while queue:
			tid = queue.popleft()
			if tid in needed:
				continue
			needed.add(tid)
			for pid in self.topics[tid].prerequisites:
				if not self.topics[pid].mastered:
					queue.append(pid)
		full_order = self.learning_order()
		return [t for t in full_order if t.topic_id in needed]

	def get_subtopics(self, topic_id: int) -> list[Topic]:
		return [self.topics[pid] for pid in self.topics[topic_id].prerequisites]

	def get_dependents(self, topic_id: int) -> list[Topic]:
		return [self.topics[uid] for uid in self.topics[topic_id].unlocks]

	# ---- graph tensors -----------------------------------------------------
	def build_edge_index(self) -> torch.Tensor:
		"""[2, E] tensor  (prerequisite → dependent)."""
		if self._edge_index_cache is not None:
			return self._edge_index_cache
		src, dst = [], []
		for t in self.topics.values():
			tid = t.topic_id
			for uid in t.unlocks:
				src.append(tid)
				dst.append(uid)
		ei = torch.tensor([src, dst], dtype=torch.long, device=self.device)
		self._edge_index_cache = ei
		return ei

	def build_feature_matrix(self) -> torch.Tensor:
		ordered = [self.topics[tid].features for tid in sorted(self.topics)]
		return torch.stack(ordered, dim=0).to(self.device)

	def out_degree(self) -> torch.Tensor:
		cached = self._degree_cache.get("out")
		if cached is not None:
			return cached
		ei = self.build_edge_index()
		n = max(self.topics) + 1 if self.topics else 0
		out = degree(ei[0], num_nodes=n, dtype=torch.long)
		self._degree_cache["out"] = out
		return out

	def in_degree(self) -> torch.Tensor:
		cached = self._degree_cache.get("in")
		if cached is not None:
			return cached
		ei = self.build_edge_index()
		n = max(self.topics) + 1 if self.topics else 0
		ind = degree(ei[1], num_nodes=n, dtype=torch.long)
		self._degree_cache["in"] = ind
		return ind

	def build_sparse_adjacency(self) -> torch.Tensor:
		ei = self.build_edge_index()
		n = max(self.topics) + 1 if self.topics else 0
		vals = torch.ones(ei.size(1), device=self.device)
		return torch.sparse_coo_tensor(ei, vals, size=(n, n)).coalesce()

	def to_pyg_data(self) -> Data:
		ei = self.build_edge_index()
		x = self.build_feature_matrix()
		n = max(self.topics) + 1 if self.topics else 0
		return Data(edge_index=ei, x=x, num_nodes=n)

	# ---- spec-based (re)build ---------------------------------------------
	@classmethod
	def from_spec(
		cls,
		specs: list[TopicSpec] | list[dict],
		device: torch.device | str | None = None,
	) -> "KnowledgeGraph":
		"""
		Build a full KnowledgeGraph from a lightweight list of specs.
		Each spec is either a TopicSpec or a dict with keys:
			name, description, level, prerequisite_names

		This is the fastest way to rebuild the graph when the user is
		unsatisfied and wants a new curriculum.
		"""
		kg = cls(device=device)
		kg._populate_from_specs(specs)
		return kg

	def rebuild_from_spec(self, specs: list[TopicSpec] | list[dict]) -> None:
		"""Clear the current graph and rebuild from specs in-place."""
		self.clear()
		self._populate_from_specs(specs)

	def _populate_from_specs(self, specs: list[TopicSpec] | list[dict]) -> None:
		"""Internal: bulk-create topics + edges from specs."""
		# normalise to TopicSpec
		normalised: list[TopicSpec] = []
		for s in specs:
			if isinstance(s, dict):
				normalised.append(TopicSpec(
					name=s["name"],
					description=s.get("description", ""),
					level=s.get("level", "foundational"),
					prerequisite_names=s.get("prerequisite_names", []),
				))
			else:
				normalised.append(s)

		# bulk-create topics (one invalidation)
		topics: list[Topic] = []
		for spec in normalised:
			t = Topic(
				topic_id=self._next_id,
				name=spec.name,
				description=spec.description,
				level=_LEVEL_MAP.get(spec.level.lower(), TopicLevel.FOUNDATIONAL),
			)
			self._next_id += 1
			topics.append(t)
		self.add_topics_bulk(topics)

		# resolve prerequisite names → ids and bulk-add edges
		edges: list[tuple[int, int]] = []
		for spec, topic in zip(normalised, topics):
			for pre_name in spec.prerequisite_names:
				pre = self.get_topic_by_name(pre_name)
				if pre is None:
					raise ValueError(
						f"prerequisite '{pre_name}' not found for topic '{spec.name}'"
					)
				edges.append((pre.topic_id, topic.topic_id))
		if edges:
			self.add_prerequisites_bulk(edges)

	# ---- pretty printing ---------------------------------------------------
	def print_curriculum(self) -> None:
		order = self.learning_order()
		print("=== NeuraLearn Curriculum ===")
		for i, t in enumerate(order, 1):
			status = "[x]" if t.mastered else ("[ ]" if self.is_unlocked(t.topic_id) else "[locked]")
			prereqs = ", ".join(self.topics[p].name for p in t.prerequisites) or "none"
			print(f"  {i}. {status} {t.name} ({t.level.name}) | prereqs: {prereqs}")
		pct = self.mastery_progress() * 100
		print(f"\nProgress: {pct:.0f}%  ({len(self.get_mastered())}/{self.num_topics} topics mastered)")


# ---------------------------------------------------------------------------
# Sample curriculum spec (plain data — cheap to store / regenerate from)
# ---------------------------------------------------------------------------
SAMPLE_ML_CURRICULUM: list[dict] = [
	{"name": "Linear Algebra",   "description": "Vectors, matrices, transformations",          "level": "foundational"},
	{"name": "Python Basics",    "description": "Syntax, data types, control flow",            "level": "foundational"},
	{"name": "Python OOP",       "description": "Classes, inheritance, polymorphism",           "level": "intermediate", "prerequisite_names": ["Python Basics"]},
	{"name": "Statistics",       "description": "Probability, distributions, hypothesis tests", "level": "intermediate", "prerequisite_names": ["Linear Algebra", "Python Basics"]},
	{"name": "Data Wrangling",   "description": "Pandas, cleaning, feature engineering",        "level": "intermediate", "prerequisite_names": ["Statistics", "Python Basics"]},
	{"name": "Machine Learning", "description": "Supervised & unsupervised learning",           "level": "advanced",     "prerequisite_names": ["Statistics", "Python OOP"]},
	{"name": "Deep Learning",    "description": "Backprop, CNNs, RNNs, Transformers",           "level": "advanced",     "prerequisite_names": ["Machine Learning"]},
	{"name": "Neural Architecture Design", "description": "Custom layers, GNNs, attention",     "level": "expert",       "prerequisite_names": ["Deep Learning"]},
]


def build_sample_knowledge_graph() -> KnowledgeGraph:
	"""Build from the sample spec — one line, instant rebuild."""
	return KnowledgeGraph.from_spec(SAMPLE_ML_CURRICULUM)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
	# ---- first build -------------------------------------------------------
	kg = build_sample_knowledge_graph()
	print("--- Full Curriculum ---")
	kg.print_curriculum()

	print("\n--- Available to learn now ---")
	for t in kg.get_available():
		print(f"  • {t.name}")

	# simulate mastering foundational topics
	print("\n>>> Mastering: Linear Algebra, Python Basics, Python OOP")
	kg.master_topic(0)
	kg.master_topic(1)
	kg.master_topic(2)

	print("\n--- Available to learn now ---")
	for t in kg.get_available():
		print(f"  • {t.name}")

	# shortest path to Deep Learning
	dl = kg.get_topic_by_name("Deep Learning")
	if dl:
		print("\n--- Shortest path to 'Deep Learning' ---")
		for step, t in enumerate(kg.shortest_path_to(dl.topic_id), 1):
			tag = "(done)" if t.mastered else "(todo)"
			print(f"  {step}. {t.name} {tag}")

	# ---- user unsatisfied? rebuild instantly --------------------------------
	print("\n\n>>> User unsatisfied — rebuilding with a different curriculum…")
	new_spec = [
		{"name": "HTML & CSS",      "level": "foundational"},
		{"name": "JavaScript",      "level": "foundational"},
		{"name": "React",           "level": "intermediate", "prerequisite_names": ["HTML & CSS", "JavaScript"]},
		{"name": "Node.js",         "level": "intermediate", "prerequisite_names": ["JavaScript"]},
		{"name": "Full-Stack App",  "level": "advanced",     "prerequisite_names": ["React", "Node.js"]},
	]
	kg.rebuild_from_spec(new_spec)       # reuses same object, no new allocation
	print("\n--- New Curriculum ---")
	kg.print_curriculum()

	print("\n--- Graph structure ---")
	print("edge_index:\n", kg.build_edge_index())
	print("out_degree:", kg.out_degree().tolist())
	print("in_degree: ", kg.in_degree().tolist())

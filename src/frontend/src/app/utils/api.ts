/**
 * NeuraLearn — API Client  (full feature surface)
 * =================================================
 * Replaces the old hardcoded graph templates with real calls
 * to the Flask backend (Webscraping → KnowledgeGraph → ACO).
 *
 * Surfaces ALL backend features:
 *  - Knowledge graph + learning resources + spectral cluster labels
 *  - Mastery tracking (prerequisite-validated)
 *  - Shortest path to any target topic
 *  - Graph analytics (spectral gap, connectivity, etc.)
 *  - ACO convergence history
 *  - SDS spell correction
 *
 * All functions return Promises.  The React components use
 * React state + useEffect to handle the async lifecycle.
 */
import type { Node, Edge, MarkerType } from "reactflow";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

export interface LearningResource {
  title: string;
  url: string;
  source: string;
  type: string;
}

export interface LearningPath {
  id: string;
  name: string;
  description: string;
  duration: string;
  difficulty: "beginner" | "intermediate" | "advanced";
  nodeIds: string[];
  convergence?: number[];
}

export interface GraphStats {
  numTopics: number;
  numEdges: number;
  algebraicConnectivity: number | null;
  spectralGap: number | null;
  connectedComponents: number | null;
  avgOutDegree?: number;
  avgInDegree?: number;
  maxOutDegree?: number;
  maxInDegree?: number;
}

export interface SkillGraphData {
  nodes: Node[];
  edges: Edge[];
  paths: LearningPath[];
  stats?: GraphStats;
}

export interface SpellSuggestion {
  word: string;
  score: number;
}

export interface SpellResult {
  original: string;
  suggestions: SpellSuggestion[];
  inDictionary: boolean;
}

export interface MasteryState {
  mastered: { id: string; name: string }[];
  available: { id: string; name: string }[];
  locked: { id: string; name: string }[];
  progress: number;
}

export interface ShortestPathStep {
  id: string;
  name: string;
  mastered: boolean;
  level: string;
}

export interface FlashCard {
  front: string;
  back: string;
  tags: string;
}

export interface FlashCardExport {
  cards: FlashCard[];
  tsv: string;
  count: number;
}

export interface StudyStats {
  totalMinutes: number;
  masteredMinutes: number;
  remainingMinutes: number;
  totalHours: number;
  remainingHours: number;
  byLevel: Record<string, number>;
  topicCount: number;
  masteredCount: number;
}

/* ------------------------------------------------------------------ */
/*  Internal helpers                                                   */
/* ------------------------------------------------------------------ */

const API_BASE = "/api";

const REQUEST_TIMEOUT_MS = 90_000; // 90 s — webscraping can be slow

async function post<T>(path: string, body: Record<string, unknown>): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const res = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error ?? `API ${res.status}`);
    }

    return res.json() as Promise<T>;
  } catch (e: unknown) {
    if (e instanceof DOMException && e.name === "AbortError") {
      throw new Error("Request timed out — the server took too long to respond.");
    }
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

/* ------------------------------------------------------------------ */
/*  Colour / style helpers for nodes                                   */
/* ------------------------------------------------------------------ */

const LEVEL_COLOURS: Record<string, { bg: string; border: string; text: string }> = {
  beginner:     { bg: "#f0fdf4", border: "#86efac", text: "#166534" },
  intermediate: { bg: "#fefce8", border: "#fde047", text: "#854d0e" },
  advanced:     { bg: "#fef2f2", border: "#fca5a5", text: "#991b1b" },
};

function styleNode(node: Node): Node {
  const difficulty = node.data?.difficulty as string | undefined;
  const palette = LEVEL_COLOURS[difficulty ?? ""] ?? LEVEL_COLOURS.intermediate;
  return {
    ...node,
    type: "custom",
    style: {
      background: palette.bg,
      border: `2px solid ${palette.border}`,
      color: palette.text,
      borderRadius: 12,
      padding: "10px 18px",
      fontWeight: 600,
      fontSize: 13,
    },
  };
}

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

/**
 * Generate a full knowledge graph for the given skill.
 * Calls: Webscraping → KnowledgeGraph → ACO on the backend.
 */
export async function generateKnowledgeGraph(
  skill: string,
): Promise<SkillGraphData> {
  const raw = await post<{
    nodes: Node[];
    edges: Edge[];
    paths: LearningPath[];
    stats?: GraphStats;
  }>("/generate", { skill });

  return {
    nodes: raw.nodes.map(styleNode),
    edges: raw.edges.map((e) => ({
      ...e,
      style: { stroke: "#a78bfa", strokeWidth: 2 },
      type: "smoothstep",
      markerEnd: {
        type: "arrowclosed" as MarkerType,
        width: 20,
        height: 20,
        color: "#a78bfa",
      },
    })),
    paths: raw.paths,
    stats: raw.stats,
  };
}

/**
 * Generate a sub-graph for a specific subtopic.
 */
export async function generateSubGraph(
  topic: string,
): Promise<SkillGraphData> {
  const raw = await post<{
    nodes: Node[];
    edges: Edge[];
    paths: LearningPath[];
    stats?: GraphStats;
  }>("/sub-graph", { topic });

  return {
    nodes: raw.nodes.map(styleNode),
    edges: raw.edges.map((e) => ({
      ...e,
      style: { stroke: "#a78bfa", strokeWidth: 2 },
      type: "smoothstep",
      markerEnd: {
        type: "arrowclosed" as MarkerType,
        width: 20,
        height: 20,
        color: "#a78bfa",
      },
    })),
    paths: raw.paths,
    stats: raw.stats,
  };
}

/**
 * Spell-check / auto-correct a text input via SDS.
 */
export async function spellCheck(
  text: string,
  topK = 5,
): Promise<SpellResult[]> {
  const res = await post<{ results: SpellResult[] }>("/spell-check", {
    text,
    top_k: topK,
  });
  return res.results;
}

/**
 * Mark a topic as mastered (server-side prerequisite validation).
 */
export async function masterTopic(
  skill: string,
  topicId: string,
): Promise<MasteryState & { success: boolean; reason?: string }> {
  return post("/master", { skill, topicId });
}

/**
 * Un-master a topic (toggle back to incomplete), with cascade.
 */
export async function unmasterTopic(
  skill: string,
  topicId: string,
): Promise<MasteryState & { success: boolean }> {
  return post("/unmaster", { skill, topicId });
}

/**
 * Get the shortest path (min topics) to reach a target topic.
 */
export async function getShortestPath(
  skill: string,
  targetId: string,
): Promise<ShortestPathStep[]> {
  const res = await post<{ path: ShortestPathStep[] }>("/shortest-path", {
    skill,
    targetId,
  });
  return res.path;
}

/**
 * Fetch current mastery progress for a stored graph.
 */
export async function getProgress(skill: string): Promise<MasteryState> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const res = await fetch(`${API_BASE}/progress/${encodeURIComponent(skill)}`, {
      signal: controller.signal,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error ?? `API ${res.status}`);
    }
    return res.json() as Promise<MasteryState>;
  } catch (e: unknown) {
    if (e instanceof DOMException && e.name === "AbortError") {
      throw new Error("Request timed out.");
    }
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Export flashcards for a stored graph (Anki-compatible).
 */
export async function exportFlashcards(skill: string): Promise<FlashCardExport> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const res = await fetch(`${API_BASE}/flashcards/${encodeURIComponent(skill)}`, {
      signal: controller.signal,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error ?? `API ${res.status}`);
    }
    return res.json() as Promise<FlashCardExport>;
  } catch (e: unknown) {
    if (e instanceof DOMException && e.name === "AbortError") {
      throw new Error("Request timed out.");
    }
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Get study time estimation & difficulty breakdown.
 */
export async function getStudyStats(skill: string): Promise<StudyStats> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const res = await fetch(`${API_BASE}/study-stats/${encodeURIComponent(skill)}`, {
      signal: controller.signal,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error ?? `API ${res.status}`);
    }
    return res.json() as Promise<StudyStats>;
  } catch (e: unknown) {
    if (e instanceof DOMException && e.name === "AbortError") {
      throw new Error("Request timed out.");
    }
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

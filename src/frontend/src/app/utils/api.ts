/**
 * NeuraLearn — API Client
 * =======================
 * Replaces the old hardcoded graph templates with real calls
 * to the Flask backend (Webscraping → KnowledgeGraph → ACO).
 *
 * All functions return Promises.  The React components use
 * React state + useEffect to handle the async lifecycle.
 */
import type { Node, Edge } from "reactflow";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

export interface LearningPath {
  id: string;
  name: string;
  description: string;
  duration: string;
  difficulty: "beginner" | "intermediate" | "advanced";
  nodeIds: string[];
}

export interface SkillGraphData {
  nodes: Node[];
  edges: Edge[];
  paths: LearningPath[];
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

/* ------------------------------------------------------------------ */
/*  Internal helpers                                                   */
/* ------------------------------------------------------------------ */

const API_BASE = "/api";

async function post<T>(path: string, body: Record<string, unknown>): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error ?? `API ${res.status}`);
  }

  return res.json() as Promise<T>;
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
  }>("/generate", { skill });

  return {
    nodes: raw.nodes.map(styleNode),
    edges: raw.edges.map((e) => ({
      ...e,
      style: { stroke: "#a78bfa", strokeWidth: 2 },
      type: "smoothstep",
    })),
    paths: raw.paths,
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
  }>("/sub-graph", { topic });

  return {
    nodes: raw.nodes.map(styleNode),
    edges: raw.edges.map((e) => ({
      ...e,
      style: { stroke: "#a78bfa", strokeWidth: 2 },
      type: "smoothstep",
    })),
    paths: raw.paths,
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

import { useState, useCallback, useEffect, useMemo } from "react";
import { useParams, useNavigate } from "react-router";
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  Node,
  Edge,
  useNodesState,
  useEdgesState,
  Panel,
} from "reactflow";
import "reactflow/dist/style.css";

import {
  generateKnowledgeGraph,
  generateSubGraph,
  masterTopic,
  unmasterTopic,
  getShortestPath,
  getProgress,
  exportFlashcards,
  getStudyStats,
} from "../utils/api";
import type { LearningPath, SkillGraphData, LearningResource, GraphStats, ShortestPathStep, StudyStats } from "../utils/api";

import { Button } from "../components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import { Progress } from "../components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";
import {
  ArrowLeft,
  ArrowRight,
  Brain,
  CheckCircle2,
  Circle,
  Clock,
  Play,
  Route,
  Sparkles,
  Target,
  TrendingUp,
  Menu,
  Loader2,
  AlertTriangle,
  RefreshCw,
  ExternalLink,
  BarChart3,
  BookOpen,
  Zap,
  Timer,
  Trophy,
  Download,
  Moon,
  Sun,
} from "lucide-react";
import { motion } from "motion/react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "../components/ui/dialog";
import { Sheet, SheetContent, SheetTrigger } from "../components/ui/sheet";
import { toast } from "sonner";
import { Toaster } from "../components/ui/sonner";
import { CustomNode } from "../components/CustomNode";
import { PomodoroTimer } from "../components/PomodoroTimer";
import { GamificationPanel } from "../components/GamificationPanel";

/* ── Constants (stable ref — defined outside component to avoid re-renders) ── */
const nodeTypes = { custom: CustomNode } as const;

/* ── Types ──────────────────────────────────────────────────────── */
type LoadState = "idle" | "loading" | "done" | "error";

/* ── Component ──────────────────────────────────────────────────── */
export function KnowledgeGraph() {
  const { skill } = useParams<{ skill: string }>();
  const navigate = useNavigate();
  const decodedSkill = skill ? decodeURIComponent(skill) : "";

  /* --- Graph state --- */
  const [graphData, setGraphData] = useState<SkillGraphData | null>(null);
  const [loadState, setLoadState] = useState<LoadState>("idle");
  const [errorMsg, setErrorMsg] = useState("");

  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedPath, setSelectedPath] = useState<LearningPath | null>(null);
  const [completedNodes, setCompletedNodes] = useState<Set<string>>(new Set());
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);

  /* --- Sub-graph dialog --- */
  const [showSubGraph, setShowSubGraph] = useState(false);
  const [subGraphData, setSubGraphData] = useState<SkillGraphData | null>(null);
  const [subGraphLoading, setSubGraphLoading] = useState(false);

  /* --- Shortest path dialog --- */
  const [shortestPath, setShortestPath] = useState<ShortestPathStep[] | null>(null);
  const [shortestPathLoading, setShortestPathLoading] = useState(false);

  /* --- New features state --- */
  const [pomodoroSessions, setPomodoroSessions] = useState(0);
  const [darkMode, setDarkMode] = useState(false);
  const [studyStats, setStudyStats] = useState<StudyStats | null>(null);

  /* --- Fetch graph on mount / skill change --- */
  const fetchGraph = useCallback(async () => {
    if (!decodedSkill) return;
    setLoadState("loading");
    setErrorMsg("");
    setSelectedPath(null);
    setCompletedNodes(new Set());

    try {
      const data = await generateKnowledgeGraph(decodedSkill);
      setGraphData(data);
      setNodes(data.nodes);
      setEdges(data.edges);
      setLoadState("done");
      toast.success("Knowledge graph generated!", {
        description: `${data.nodes.length} topics found for ${decodedSkill}`,
      });

      // Restore server-side mastery state (if graph was generated before)
      try {
        const progress = await getProgress(decodedSkill);
        if (progress.mastered.length > 0) {
          setCompletedNodes(new Set(progress.mastered.map((m) => m.id)));
        }
      } catch {
        // First generation — no stored progress yet, that's fine
      }
    } catch (err: unknown) {
      setLoadState("error");
      setErrorMsg(err instanceof Error ? err.message : "Unknown error");
      toast.error("Generation failed", { description: err instanceof Error ? err.message : undefined });
    }
  }, [decodedSkill, setNodes, setEdges]);

  useEffect(() => {
    fetchGraph();
  }, [fetchGraph]);

  /* --- Highlight selected path --- */
  useEffect(() => {
    if (!graphData) return;

    if (selectedPath) {
      const pathSet = new Set(selectedPath.nodeIds);
      setNodes(
        graphData.nodes.map((node) => ({
          ...node,
          style: {
            ...node.style,
            opacity: pathSet.has(node.id) ? 1 : 0.25,
            border: pathSet.has(node.id)
              ? "2px solid #8b5cf6"
              : (node.style?.border ?? undefined),
          },
        })),
      );
      setEdges(
        graphData.edges.map((edge) => ({
          ...edge,
          style: {
            ...edge.style,
            opacity:
              pathSet.has(edge.source) && pathSet.has(edge.target) ? 1 : 0.15,
          },
        })),
      );
    } else {
      setNodes(graphData.nodes);
      setEdges(graphData.edges);
    }
  }, [selectedPath, graphData, setNodes, setEdges]);

  /* --- Node interactions --- */
  const toggleNodeCompletion = useCallback(async (nodeId: string) => {
    if (!decodedSkill) return;

    // If already mastered locally, un-master on server with cascade
    if (completedNodes.has(nodeId)) {
      try {
        const res = await unmasterTopic(decodedSkill, nodeId);
        if (res.success) {
          setCompletedNodes(new Set(res.mastered.map((m) => m.id)));
          toast.info("Progress updated", { description: "Topic unmarked (+ dependents)" });
        }
      } catch {
        // Offline fallback
        setCompletedNodes((prev) => {
          const next = new Set(prev);
          next.delete(nodeId);
          return next;
        });
        toast.info("Progress updated", { description: "Topic marked as incomplete (offline)" });
      }
      return;
    }

    // Server-side mastery validation
    try {
      const res = await masterTopic(decodedSkill, nodeId);
      if (res.success) {
        setCompletedNodes(new Set(res.mastered.map((m) => m.id)));
        toast.success("Great progress!", { description: "Topic mastered — server verified" });
      } else {
        toast.error("Prerequisites not met", { description: res.reason ?? "Complete prerequisites first" });
      }
    } catch {
      // Network error — fallback to local toggle
      setCompletedNodes((prev) => {
        const next = new Set(prev);
        next.add(nodeId);
        return next;
      });
      toast.success("Great progress!", { description: "Topic marked as complete (offline)" });
    }
  }, [decodedSkill, completedNodes]);

  const handleNodeClick = useCallback((_event: React.MouseEvent, node: Node) => {
    setSelectedNode(node);
  }, []);

  const openSubGraph = useCallback(async () => {
    if (!selectedNode) return;
    const label = selectedNode.data.label as string;
    setShowSubGraph(true);
    setSubGraphLoading(true);
    setSubGraphData(null);

    try {
      const data = await generateSubGraph(label);
      setSubGraphData(data);
      toast.info("Sub-graph generated", { description: `Exploring: ${label}` });
    } catch (err: unknown) {
      toast.error("Sub-graph failed", { description: err instanceof Error ? err.message : undefined });
    } finally {
      setSubGraphLoading(false);
    }
  }, [selectedNode]);

  /* --- Shortest path to a target node --- */
  const findShortestPathTo = useCallback(async () => {
    if (!selectedNode || !decodedSkill) return;
    setShortestPathLoading(true);
    setShortestPath(null);
    try {
      const path = await getShortestPath(decodedSkill, selectedNode.id);
      setShortestPath(path);
      toast.info("Shortest path found", { description: `${path.length} topics to reach ${selectedNode.data.label}` });
    } catch (err: unknown) {
      toast.error("Shortest path failed", { description: err instanceof Error ? err.message : undefined });
    } finally {
      setShortestPathLoading(false);
    }
  }, [selectedNode, decodedSkill]);

  /* --- Flashcard export --- */
  const handleExportFlashcards = useCallback(async () => {
    if (!decodedSkill) return;
    try {
      const data = await exportFlashcards(decodedSkill);
      // Download as TSV file
      const blob = new Blob([data.tsv], { type: "text/tab-separated-values" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${decodedSkill.replace(/\s+/g, "_")}_flashcards.tsv`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success(`Exported ${data.count} flashcards!`, {
        description: "Import the .tsv file into Anki or any flashcard app",
      });
    } catch {
      toast.error("Export failed", { description: "Generate a graph first" });
    }
  }, [decodedSkill]);

  /* --- Fetch study stats --- */
  const fetchStudyStats = useCallback(async () => {
    if (!decodedSkill) return;
    try {
      const stats = await getStudyStats(decodedSkill);
      setStudyStats(stats);
    } catch {
      // Not critical
    }
  }, [decodedSkill]);

  useEffect(() => {
    if (loadState === "done") fetchStudyStats();
  }, [loadState, fetchStudyStats, completedNodes]);

  /* --- Derived --- */
  const difficultyColor = useCallback((d: string) => {
    switch (d) {
      case "beginner":
        return "bg-green-100 text-green-800 border-green-300";
      case "intermediate":
        return "bg-yellow-100 text-yellow-800 border-yellow-300";
      case "advanced":
        return "bg-red-100 text-red-800 border-red-300";
      default:
        return "bg-gray-100 text-gray-800 border-gray-300";
    }
  }, []);

  const levelBadgeColor = useCallback((level: string) => {
    switch (level) {
      case "foundational":
        return "bg-green-100 text-green-700 border-green-200";
      case "intermediate":
        return "bg-yellow-100 text-yellow-700 border-yellow-200";
      case "advanced":
        return "bg-orange-100 text-orange-700 border-orange-200";
      case "expert":
        return "bg-red-100 text-red-700 border-red-200";
      default:
        return "bg-gray-100 text-gray-600 border-gray-200";
    }
  }, []);

  const calculateProgress = useMemo(() => {
    if (!selectedPath) return 0;
    const done = selectedPath.nodeIds.filter((id) => completedNodes.has(id)).length;
    return (done / selectedPath.nodeIds.length) * 100;
  }, [selectedPath, completedNodes]);

  /* Build a lookup from node id -> node for the detail view */
  const nodeMap = useMemo(() => {
    const m = new Map<string, Node>();
    if (graphData) {
      for (const n of graphData.nodes) m.set(n.id, n);
    }
    return m;
  }, [graphData]);

  /* ── Sidebar content (shared between mobile sheet & desktop aside) ── */
  const sidebarContent = graphData ? (
    <Tabs defaultValue="paths" className="h-full">
      <TabsList className="w-full grid grid-cols-5 rounded-none">
        <TabsTrigger value="paths" className="gap-1 text-[11px] px-1">
          <Route className="size-3.5" />
          Paths
        </TabsTrigger>
        <TabsTrigger value="progress" className="gap-1 text-[11px] px-1">
          <Target className="size-3.5" />
          Progress
        </TabsTrigger>
        <TabsTrigger value="stats" className="gap-1 text-[11px] px-1">
          <BarChart3 className="size-3.5" />
          Stats
        </TabsTrigger>
        <TabsTrigger value="timer" className="gap-1 text-[11px] px-1">
          <Timer className="size-3.5" />
          Timer
        </TabsTrigger>
        <TabsTrigger value="rewards" className="gap-1 text-[11px] px-1">
          <Trophy className="size-3.5" />
          Rewards
        </TabsTrigger>
      </TabsList>

      {/* -- Paths tab -- */}
      <TabsContent value="paths" className="p-4 space-y-3">
        {/* Header */}
        <div className="mb-4">
          <h3 className="font-semibold mb-1 flex items-center gap-2">
            <Sparkles className="size-4 text-purple-600" />
            {selectedPath ? selectedPath.name : "Choose Your Learning Path"}
          </h3>
          <p className="text-xs text-gray-500">
            {selectedPath
              ? selectedPath.description
              : "Select a path to highlight your journey through the knowledge graph"}
          </p>
        </div>

        {/* ── Path picker (no path selected) ── */}
        {!selectedPath &&
          graphData.paths.map((path) => (
            <motion.div
              key={path.id}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <Card
                className="cursor-pointer transition-all hover:shadow-md"
                onClick={() => setSelectedPath(path)}
              >
                <CardHeader className="pb-3">
                  <div className="flex items-start justify-between">
                    <CardTitle className="text-sm">{path.name}</CardTitle>
                    <Badge
                      className={difficultyColor(path.difficulty)}
                      variant="outline"
                    >
                      {path.difficulty}
                    </Badge>
                  </div>
                  <CardDescription className="text-xs">
                    {path.description}
                  </CardDescription>
                </CardHeader>
                <CardContent className="pb-3">
                  <div className="flex items-center justify-between text-xs text-gray-600">
                    <span className="flex items-center gap-1">
                      <TrendingUp className="size-3" />
                      {path.nodeIds.length} topics
                    </span>
                    <span>{path.duration}</span>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}

        {/* ── Expanded path detail (path selected) ── */}
        {selectedPath && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-3"
          >
            {/* Stats row */}
            <div className="grid grid-cols-3 gap-2 text-center">
              <div className="rounded-lg border bg-purple-50 p-2">
                <p className="text-lg font-bold text-purple-700">
                  {selectedPath.nodeIds.length}
                </p>
                <p className="text-[10px] text-purple-600">Topics</p>
              </div>
              <div className="rounded-lg border bg-green-50 p-2">
                <p className="text-lg font-bold text-green-700">
                  {selectedPath.nodeIds.filter((id) => completedNodes.has(id)).length}
                </p>
                <p className="text-[10px] text-green-600">Completed</p>
              </div>
              <div className="rounded-lg border bg-blue-50 p-2">
                <p className="text-lg font-bold text-blue-700">
                  {Math.round(calculateProgress)}%
                </p>
                <p className="text-[10px] text-blue-600">Progress</p>
              </div>
            </div>

            {/* Progress bar */}
            <Progress value={calculateProgress} className="h-2" />

            {/* Difficulty + estimated time */}
            <div className="flex items-center gap-2 text-xs text-gray-500">
              <Badge
                className={difficultyColor(selectedPath.difficulty)}
                variant="outline"
              >
                {selectedPath.difficulty}
              </Badge>
              <span className="flex items-center gap-1">
                <Clock className="size-3" />
                {selectedPath.duration}
              </span>
            </div>

            {/* Step-by-step topic list */}
            <div className="space-y-1">
              <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
                Learning Order
              </h4>
              {selectedPath.nodeIds.map((id, idx) => {
                const node = nodeMap.get(id);
                if (!node) return null;
                const label = node.data.label as string;
                const level = (node.data.level as string) ?? "";
                const desc = (node.data.description as string) ?? "";
                const done = completedNodes.has(id);
                const isLast = idx === selectedPath.nodeIds.length - 1;

                return (
                  <div key={id} className="flex gap-2">
                    {/* Timeline spine */}
                    <div className="flex flex-col items-center">
                      <button
                        onClick={() => toggleNodeCompletion(id)}
                        className="relative z-10 flex-shrink-0"
                      >
                        {done ? (
                          <CheckCircle2 className="size-5 text-green-600" />
                        ) : (
                          <div className="size-5 rounded-full border-2 border-purple-300 bg-white flex items-center justify-center text-[9px] font-bold text-purple-500">
                            {idx + 1}
                          </div>
                        )}
                      </button>
                      {!isLast && (
                        <div className="w-px flex-1 bg-gray-200 my-0.5" />
                      )}
                    </div>

                    {/* Content */}
                    <div
                      className={`flex-1 pb-3 ${
                        done ? "opacity-60" : ""
                      }`}
                    >
                      <div className="flex items-center gap-1.5">
                        <span
                          className={`text-sm font-medium leading-tight ${
                            done ? "line-through text-gray-400" : "text-gray-800"
                          }`}
                        >
                          {label}
                        </span>
                        {level && (
                          <span
                            className={`inline-block px-1.5 py-px rounded text-[9px] font-medium border ${
                              levelBadgeColor(level)
                            }`}
                          >
                            {level}
                          </span>
                        )}
                      </div>
                      {desc && (
                        <p className="text-[11px] text-gray-500 mt-0.5 leading-snug">
                          {desc}
                        </p>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Actions */}
            <div className="flex gap-2 pt-1">
              <Button
                variant="outline"
                size="sm"
                className="flex-1 gap-1"
                onClick={() => setSelectedPath(null)}
              >
                <ArrowLeft className="size-3" />
                All Paths
              </Button>
              <Button
                size="sm"
                className="flex-1 gap-1 bg-purple-600 hover:bg-purple-700"
                onClick={() => {
                  /* Find the first incomplete node in this path and open its detail */
                  const nextId = selectedPath.nodeIds.find(
                    (nid) => !completedNodes.has(nid),
                  );
                  if (nextId) {
                    const n = nodeMap.get(nextId);
                    if (n) setSelectedNode(n);
                  } else {
                    toast.success("All done!", {
                      description: "You've completed every topic in this path.",
                    });
                  }
                }}
              >
                <Play className="size-3" />
                {selectedPath.nodeIds.every((nid) => completedNodes.has(nid))
                  ? "All Complete!"
                  : "Continue Learning"}
              </Button>
            </div>
          </motion.div>
        )}
      </TabsContent>

      {/* -- Progress tab -- */}
      <TabsContent value="progress" className="p-4 space-y-3">
        <div className="mb-4">
          <h3 className="font-semibold mb-1">Your Progress</h3>
          <p className="text-xs text-gray-500">
            Track which topics you've completed
          </p>
        </div>

        <div className="space-y-2">
          {graphData.nodes.map((node) => (
            <div
              key={node.id}
              className="flex items-center gap-2 p-2 rounded-lg hover:bg-gray-50 cursor-pointer"
              onClick={() => toggleNodeCompletion(node.id)}
            >
              {completedNodes.has(node.id) ? (
                <CheckCircle2 className="size-5 text-green-600 flex-shrink-0" />
              ) : (
                <Circle className="size-5 text-gray-300 flex-shrink-0" />
              )}
              <span
                className={`text-sm ${
                  completedNodes.has(node.id) ? "line-through text-gray-500" : ""
                }`}
              >
                {node.data.label as string}
              </span>
            </div>
          ))}
        </div>

        <div className="pt-4 border-t">
          <div className="flex justify-between text-sm mb-2">
            <span>Overall Progress</span>
            <span className="font-medium">
              {completedNodes.size} / {graphData.nodes.length}
            </span>
          </div>
          <Progress
            value={(completedNodes.size / graphData.nodes.length) * 100}
          />
        </div>
      </TabsContent>

      {/* -- Stats tab -- */}
      <TabsContent value="stats" className="p-4 space-y-3">
        <div className="mb-4">
          <h3 className="font-semibold mb-1 flex items-center gap-2">
            <BarChart3 className="size-4 text-blue-600" />
            Graph Analytics
          </h3>
          <p className="text-xs text-gray-500">
            Spectral &amp; topological metrics measuring curriculum quality
          </p>
        </div>

        {graphData.stats && (
          <div className="space-y-2">
            <div className="grid grid-cols-2 gap-2">
              <div className="rounded-lg border p-3 bg-blue-50">
                <p className="text-xs text-blue-600 font-medium">Topics</p>
                <p className="text-xl font-bold text-blue-800">{graphData.stats.numTopics}</p>
              </div>
              <div className="rounded-lg border p-3 bg-purple-50">
                <p className="text-xs text-purple-600 font-medium">Edges</p>
                <p className="text-xl font-bold text-purple-800">{graphData.stats.numEdges}</p>
              </div>
            </div>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Spectral Analysis</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Algebraic Connectivity (λ₂)</span>
                  <span className="font-mono font-medium">
                    {graphData.stats.algebraicConnectivity?.toFixed(4) ?? "N/A"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Spectral Gap (λ₂/λₘₐₓ)</span>
                  <span className="font-mono font-medium">
                    {graphData.stats.spectralGap?.toFixed(4) ?? "N/A"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Connected Components (β₀)</span>
                  <span className="font-mono font-medium">
                    {graphData.stats.connectedComponents ?? "N/A"}
                  </span>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Degree Distribution</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Avg In-Degree</span>
                  <span className="font-mono font-medium">{graphData.stats.avgInDegree ?? "—"}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Avg Out-Degree</span>
                  <span className="font-mono font-medium">{graphData.stats.avgOutDegree ?? "—"}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Max In-Degree</span>
                  <span className="font-mono font-medium">{graphData.stats.maxInDegree ?? "—"}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Max Out-Degree</span>
                  <span className="font-mono font-medium">{graphData.stats.maxOutDegree ?? "—"}</span>
                </div>
              </CardContent>
            </Card>

            {/* ACO Convergence mini-chart */}
            {(() => {
              const acoPath = graphData.paths.find((p) => p.id === "path-aco");
              const conv = acoPath?.convergence;
              if (!conv || conv.length < 2) return null;
              const maxVal = Math.max(...conv);
              const minVal = Math.min(...conv);
              const range = maxVal - minVal || 1;
              const h = 60;
              const w = 200;
              const points = conv
                .map((v, i) => `${(i / (conv.length - 1)) * w},${h - ((v - minVal) / range) * h}`)
                .join(" ");
              return (
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">ACO Convergence</CardTitle>
                    <CardDescription className="text-xs">
                      Cost: {conv[0].toFixed(1)} → {conv[conv.length - 1].toFixed(1)} over {conv.length} iterations
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-16">
                      <polyline
                        points={points}
                        fill="none"
                        stroke="#8b5cf6"
                        strokeWidth="2"
                        strokeLinejoin="round"
                      />
                    </svg>
                  </CardContent>
                </Card>
              );
            })()}
          </div>
        )}

        {/* Study Time Estimation */}
        {studyStats && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-1.5">
                <Clock className="size-3.5 text-green-600" />
                Study Time Estimate
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">Total Study Time</span>
                <span className="font-medium">{studyStats.totalHours}h</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Remaining</span>
                <span className="font-medium text-amber-600">{studyStats.remainingHours}h</span>
              </div>
              <Progress value={studyStats.totalMinutes > 0 ? (studyStats.masteredMinutes / studyStats.totalMinutes) * 100 : 0} className="h-2" />
              <div className="flex justify-between text-xs text-gray-500">
                <span>{studyStats.masteredCount} of {studyStats.topicCount} topics done</span>
                <span>{Math.round((studyStats.masteredMinutes / Math.max(studyStats.totalMinutes, 1)) * 100)}% time saved</span>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Flashcard Export */}
        <Button
          variant="outline"
          size="sm"
          className="w-full gap-2"
          onClick={handleExportFlashcards}
        >
          <Download className="size-3.5" />
          Export Flashcards (Anki)
        </Button>
      </TabsContent>

      {/* -- Timer tab -- */}
      <TabsContent value="timer" className="p-4 space-y-3">
        <div className="mb-2">
          <h3 className="font-semibold mb-1 flex items-center gap-2">
            <Timer className="size-4 text-purple-600" />
            Pomodoro Timer
          </h3>
          <p className="text-xs text-gray-500">
            Stay focused with timed study sessions. 25 min focus, then break.
          </p>
        </div>
        <PomodoroTimer
          onSessionComplete={() => setPomodoroSessions((n) => n + 1)}
        />

        {/* Quick tips */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Study Tips</CardTitle>
          </CardHeader>
          <CardContent className="text-xs text-gray-600 space-y-1.5">
            <p>• Complete one topic per Pomodoro session</p>
            <p>• Review completed topics during breaks</p>
            <p>• After 4 sessions, take a 15-min long break</p>
            <p>• Export flashcards in Stats tab for spaced repetition</p>
          </CardContent>
        </Card>

        {/* Study time from stats */}
        {studyStats && (
          <div className="rounded-lg border bg-blue-50 p-3 text-center">
            <p className="text-xs text-blue-600 font-medium">Estimated Total Study Time</p>
            <p className="text-2xl font-bold text-blue-800">{studyStats.totalHours}h</p>
            <p className="text-[10px] text-blue-500">
              ~{Math.ceil(studyStats.remainingMinutes / 25)} Pomodoro sessions remaining
            </p>
          </div>
        )}
      </TabsContent>

      {/* -- Rewards tab -- */}
      <TabsContent value="rewards" className="p-4 space-y-3">
        <div className="mb-2">
          <h3 className="font-semibold mb-1 flex items-center gap-2">
            <Trophy className="size-4 text-amber-500" />
            Achievements & XP
          </h3>
          <p className="text-xs text-gray-500">
            Level up by mastering topics and staying focused
          </p>
        </div>
        <GamificationPanel
          topicsCompleted={completedNodes.size}
          totalTopics={graphData.nodes.length}
          pomodoroSessions={pomodoroSessions}
        />
      </TabsContent>
    </Tabs>
  ) : null;

  /* ═════════════════════════════════════════════════════════════════ */
  /*  Render                                                          */
  /* ═════════════════════════════════════════════════════════════════ */
  return (
    <div className={`h-screen flex flex-col ${darkMode ? "bg-gray-900 text-gray-100" : "bg-gray-50"}`}>
      <Toaster theme={darkMode ? "dark" : "light"} />

      {/* ── Header ─────────────────────────────────────────────────── */}
      <header className={`px-6 py-4 border-b shadow-sm z-10 ${darkMode ? "bg-gray-800 border-gray-700" : "bg-white"}`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => navigate("/")}
              className="gap-2"
            >
              <ArrowLeft className="size-4" />
              <span className="hidden sm:inline">Back</span>
            </Button>
            <div className="flex items-center gap-2">
              <Brain className="size-6 text-purple-600" />
              <div>
                <h1 className="font-bold text-sm sm:text-lg">{decodedSkill}</h1>
                <p className="text-xs text-gray-500 hidden sm:block">
                  AI-Generated Learning Path
                </p>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Mobile sidebar */}
            <Sheet>
              <SheetTrigger asChild>
                <Button variant="outline" size="sm" className="lg:hidden">
                  <Menu className="size-4" />
                </Button>
              </SheetTrigger>
              <SheetContent side="left" className="w-80 p-0">
                {sidebarContent}
              </SheetContent>
            </Sheet>

            {/* Refresh */}
            <Button
              variant="ghost"
              size="sm"
              onClick={fetchGraph}
              disabled={loadState === "loading"}
              className="gap-1"
            >
              <RefreshCw
                className={`size-4 ${loadState === "loading" ? "animate-spin" : ""}`}
              />
              <span className="hidden sm:inline">Refresh</span>
            </Button>

            {/* Dark mode toggle */}
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setDarkMode(!darkMode)}
              className="gap-1"
            >
              {darkMode ? <Sun className="size-4" /> : <Moon className="size-4" />}
            </Button>

            {selectedPath && (
              <div className="hidden sm:flex items-center gap-4">
                <div className="text-right">
                  <p className="text-sm font-medium">{selectedPath.name}</p>
                  <p className="text-xs text-gray-500">
                    Progress: {Math.round(calculateProgress)}%
                  </p>
                </div>
                <Progress value={calculateProgress} className="w-32" />
              </div>
            )}
          </div>
        </div>
      </header>

      {/* ── Body ───────────────────────────────────────────────────── */}
      <div className="flex-1 flex overflow-hidden">
        {/* Desktop sidebar */}
        <aside className={`w-80 border-r overflow-y-auto hidden lg:block ${darkMode ? "bg-gray-800 border-gray-700" : "bg-white"}`}>
          {sidebarContent}
        </aside>

        {/* Main graph area */}
        <main className="flex-1 relative">
          {/* Loading overlay */}
          {loadState === "loading" && (
            <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-white/80 backdrop-blur-sm gap-4">
              <Loader2 className="size-10 text-purple-600 animate-spin" />
              <div className="text-center">
                <p className="font-semibold text-gray-800">
                  Generating knowledge graph&hellip;
                </p>
                <p className="text-sm text-gray-500 mt-1">
                  Scraping resources &amp; optimising learning paths
                </p>
              </div>
            </div>
          )}

          {/* Error state */}
          {loadState === "error" && (
            <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-white gap-4 px-6">
              <AlertTriangle className="size-12 text-amber-500" />
              <h2 className="text-xl font-bold text-gray-800">
                Couldn't generate graph
              </h2>
              <p className="text-gray-600 text-center max-w-md">{errorMsg}</p>
              <div className="flex gap-2">
                <Button variant="outline" onClick={() => navigate("/")}>
                  Go Back
                </Button>
                <Button onClick={fetchGraph}>
                  <RefreshCw className="size-4 mr-2" />
                  Retry
                </Button>
              </div>
            </div>
          )}

          {/* Graph */}
          {loadState === "done" && (
            <ReactFlow
              nodes={nodes}
              edges={edges}
              nodeTypes={nodeTypes}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onNodeClick={handleNodeClick}
              fitView
              className="bg-gray-50"
            >
              <Background />
              <Controls />
              <MiniMap />
              <Panel
                position="top-right"
                className="bg-white p-4 rounded-lg shadow-lg m-4 max-w-xs"
              >
                <div className="space-y-2">
                  <h3 className="font-semibold text-sm flex items-center gap-2">
                    <Brain className="size-4 text-purple-600" />
                    Graph Controls
                  </h3>
                  <p className="text-xs text-gray-600">
                    &bull; Click nodes to view details
                    <br />
                    &bull; Select a path to highlight
                    <br />
                    &bull; Track your progress
                  </p>
                </div>
              </Panel>
            </ReactFlow>
          )}
        </main>
      </div>

      {/* ── Node Detail Dialog ─────────────────────────────────────── */}
      <Dialog
        open={!!selectedNode}
        onOpenChange={(open) => {
          if (!open) {
            setSelectedNode(null);
            setShortestPath(null);
          }
        }}
      >
        <DialogContent className="max-w-lg max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Play className="size-5 text-purple-600" />
              {selectedNode?.data.label as string}
            </DialogTitle>
            <DialogDescription>
              Learn more about this topic and explore sub-skills
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            {selectedNode?.data.description && (
              <p className="text-sm text-gray-600">{selectedNode.data.description}</p>
            )}

            {/* Prerequisites */}
            {(selectedNode?.data.prerequisites as string[] | undefined)?.length ? (
              <div>
                <h4 className="font-medium mb-1 text-sm flex items-center gap-1.5">
                  <ArrowLeft className="size-3.5 text-amber-500" />
                  Prerequisites
                </h4>
                <div className="flex flex-wrap gap-1.5">
                  {(selectedNode!.data.prerequisites as string[]).map((p: string) => (
                    <Badge key={p} variant="outline" className="bg-amber-50 text-amber-700 border-amber-200 text-xs">
                      {p}
                    </Badge>
                  ))}
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  Complete these first to unlock this topic
                </p>
              </div>
            ) : (
              <div className="flex items-center gap-2 text-sm text-green-600 bg-green-50 rounded-lg p-2">
                <CheckCircle2 className="size-4" />
                <span className="font-medium">No prerequisites — you can start right away!</span>
              </div>
            )}

            {/* Unlocks */}
            {(selectedNode?.data.unlocks as string[] | undefined)?.length ? (
              <div>
                <h4 className="font-medium mb-1 text-sm flex items-center gap-1.5">
                  <ArrowRight className="size-3.5 text-purple-500" />
                  Mastering this unlocks
                </h4>
                <div className="flex flex-wrap gap-1.5">
                  {(selectedNode!.data.unlocks as string[]).map((u: string) => (
                    <Badge key={u} variant="outline" className="bg-purple-50 text-purple-700 border-purple-200 text-xs">
                      {u}
                    </Badge>
                  ))}
                </div>
              </div>
            ) : (
              <div className="flex items-center gap-2 text-sm text-blue-600 bg-blue-50 rounded-lg p-2">
                <Target className="size-4" />
                <span className="font-medium">Capstone topic — completing this masters the skill!</span>
              </div>
            )}

            {/* Learning Resources */}
            {((selectedNode?.data.resources as LearningResource[] | undefined) ?? []).length > 0 && (
              <div>
                <h4 className="font-medium mb-1.5 text-sm flex items-center gap-1.5">
                  <BookOpen className="size-3.5 text-blue-500" />
                  Learning Resources
                </h4>
                <div className="space-y-1.5 max-h-40 overflow-y-auto">
                  {(selectedNode!.data.resources as LearningResource[]).map((r, i) => (
                    <a
                      key={i}
                      href={r.url || "#"}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-start gap-2 p-2 rounded-lg border hover:bg-blue-50 transition-colors group text-sm"
                    >
                      <ExternalLink className="size-3.5 text-blue-400 mt-0.5 flex-shrink-0 group-hover:text-blue-600" />
                      <div className="min-w-0 flex-1">
                        <p className="text-gray-800 leading-tight truncate group-hover:text-blue-700">{r.title}</p>
                        <div className="flex gap-1.5 mt-0.5">
                          {r.source && (
                            <span className="text-[9px] px-1.5 py-0.5 rounded bg-gray-100 text-gray-500 border">
                              {r.source}
                            </span>
                          )}
                          {r.type && (
                            <span className="text-[9px] px-1.5 py-0.5 rounded bg-blue-50 text-blue-500 border border-blue-100">
                              {r.type}
                            </span>
                          )}
                        </div>
                      </div>
                    </a>
                  ))}
                </div>
              </div>
            )}

            {/* Shortest path result */}
            {shortestPath && (
              <div>
                <h4 className="font-medium mb-1.5 text-sm flex items-center gap-1.5">
                  <Zap className="size-3.5 text-amber-500" />
                  Shortest Path ({shortestPath.length} topics)
                </h4>
                <div className="space-y-1 max-h-32 overflow-y-auto">
                  {shortestPath.map((step, idx) => (
                    <div key={step.id} className="flex items-center gap-2 text-xs">
                      <span className="w-5 text-center font-medium text-gray-400">{idx + 1}</span>
                      {step.mastered ? (
                        <CheckCircle2 className="size-3.5 text-green-500 flex-shrink-0" />
                      ) : (
                        <Circle className="size-3.5 text-gray-300 flex-shrink-0" />
                      )}
                      <span className={step.mastered ? "line-through text-gray-400" : "text-gray-700"}>
                        {step.name}
                      </span>
                      <Badge variant="outline" className="text-[8px] px-1 py-0 ml-auto">
                        {step.level}
                      </Badge>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="flex gap-2 flex-wrap">
              <Button
                onClick={() => toggleNodeCompletion(selectedNode!.id)}
                variant="outline"
                className="flex-1 min-w-[120px]"
              >
                {completedNodes.has(selectedNode?.id || "") ? (
                  <>
                    <CheckCircle2 className="size-4 mr-2" />
                    Completed
                  </>
                ) : (
                  <>
                    <Circle className="size-4 mr-2" />
                    Mark Complete
                  </>
                )}
              </Button>
              <Button
                onClick={openSubGraph}
                className="flex-1 min-w-[120px] bg-purple-600 hover:bg-purple-700"
              >
                <Sparkles className="size-4 mr-2" />
                Sub-Skills
              </Button>
              <Button
                onClick={findShortestPathTo}
                variant="outline"
                className="flex-1 min-w-[120px]"
                disabled={shortestPathLoading}
              >
                {shortestPathLoading ? (
                  <Loader2 className="size-4 mr-2 animate-spin" />
                ) : (
                  <Zap className="size-4 mr-2" />
                )}
                Shortest Path
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* ── Sub-Graph Dialog ───────────────────────────────────────── */}
      <Dialog open={showSubGraph} onOpenChange={setShowSubGraph}>
        <DialogContent className="max-w-4xl h-[600px] flex flex-col">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Brain className="size-5 text-purple-600" />
              Sub-Skills: {selectedNode?.data.label as string}
            </DialogTitle>
            <DialogDescription>
              A detailed breakdown of how to master this specific topic
            </DialogDescription>
          </DialogHeader>

          <div className="flex-1 border rounded-lg overflow-hidden relative">
            {subGraphLoading ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
                <Loader2 className="size-8 text-purple-600 animate-spin" />
                <p className="text-sm text-gray-500">
                  Generating sub-graph&hellip;
                </p>
              </div>
            ) : subGraphData ? (
              <ReactFlow
                nodes={subGraphData.nodes}
                edges={subGraphData.edges}
                nodeTypes={nodeTypes}
                fitView
              >
                <Background />
                <Controls />
              </ReactFlow>
            ) : (
              <div className="flex items-center justify-center h-full text-gray-400">
                No data
              </div>
            )}
          </div>
          <p className="text-xs text-gray-500 text-center">
            This sub-graph shows a more detailed learning path for this specific
            topic
          </p>
        </DialogContent>
      </Dialog>
    </div>
  );
}

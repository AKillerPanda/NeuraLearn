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
  Connection,
  addEdge,
  Panel,
} from "reactflow";
import "reactflow/dist/style.css";

import {
  generateKnowledgeGraph,
  generateSubGraph,
} from "../utils/api";
import type { LearningPath, SkillGraphData } from "../utils/api";

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
  ChevronDown,
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
    } catch (err: any) {
      setLoadState("error");
      setErrorMsg(err?.message ?? "Unknown error");
      toast.error("Generation failed", { description: err?.message });
    }
  }, [decodedSkill, setNodes, setEdges]);

  useEffect(() => {
    fetchGraph();
  }, [fetchGraph]);

  /* --- Connections --- */
  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges],
  );

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
  const toggleNodeCompletion = useCallback((nodeId: string) => {
    setCompletedNodes((prev) => {
      const next = new Set(prev);
      if (next.has(nodeId)) {
        next.delete(nodeId);
        toast.info("Progress updated", { description: "Topic marked as incomplete" });
      } else {
        next.add(nodeId);
        toast.success("Great progress!", { description: "Topic marked as complete" });
      }
      return next;
    });
  }, []);

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
    } catch (err: any) {
      toast.error("Sub-graph failed", { description: err?.message });
    } finally {
      setSubGraphLoading(false);
    }
  }, [selectedNode]);

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
      <TabsList className="w-full grid grid-cols-2 rounded-none">
        <TabsTrigger value="paths" className="gap-2">
          <Route className="size-4" />
          Paths
        </TabsTrigger>
        <TabsTrigger value="progress" className="gap-2">
          <Target className="size-4" />
          Progress
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
    </Tabs>
  ) : null;

  /* ═════════════════════════════════════════════════════════════════ */
  /*  Render                                                          */
  /* ═════════════════════════════════════════════════════════════════ */
  return (
    <div className="h-screen flex flex-col bg-gray-50">
      <Toaster />

      {/* ── Header ─────────────────────────────────────────────────── */}
      <header className="px-6 py-4 bg-white border-b shadow-sm z-10">
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
        <aside className="w-80 bg-white border-r overflow-y-auto hidden lg:block">
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
              onConnect={onConnect}
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
        onOpenChange={(open) => !open && setSelectedNode(null)}
      >
        <DialogContent>
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

            <div>
              <h4 className="font-medium mb-2">About this topic:</h4>
              <p className="text-sm text-gray-600">
                This is a key concept in {decodedSkill}. Click below to generate
                a detailed sub-graph that breaks down this topic into smaller,
                manageable learning units.
              </p>
            </div>

            <div className="flex gap-2">
              <Button
                onClick={() => toggleNodeCompletion(selectedNode!.id)}
                variant="outline"
                className="flex-1"
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
                className="flex-1 bg-purple-600 hover:bg-purple-700"
              >
                <Sparkles className="size-4 mr-2" />
                Explore Sub-Skills
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

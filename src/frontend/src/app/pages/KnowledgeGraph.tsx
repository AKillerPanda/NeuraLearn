import { useState, useCallback, useEffect } from "react";
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
} from 'reactflow';
import 'reactflow/dist/style.css';
import { generateKnowledgeGraph } from "../utils/graphGenerator";
import type { LearningPath } from "../utils/graphGenerator";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import { Progress } from "../components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";
import { 
  ArrowLeft, 
  Brain, 
  CheckCircle2, 
  Circle, 
  Play, 
  Route,
  Sparkles,
  Target,
  TrendingUp,
  Menu
} from "lucide-react";
import { motion } from "motion/react";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "../components/ui/dialog";
import { Sheet, SheetContent, SheetTrigger } from "../components/ui/sheet";
import { toast } from "sonner";
import { Toaster } from "../components/ui/sonner";

export function KnowledgeGraph() {
  const { skill } = useParams<{ skill: string }>();
  const navigate = useNavigate();
  const decodedSkill = skill ? decodeURIComponent(skill) : "";

  const [graphData, setGraphData] = useState(() => generateKnowledgeGraph(decodedSkill));
  const [nodes, setNodes, onNodesChange] = useNodesState(graphData.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(graphData.edges);
  const [selectedPath, setSelectedPath] = useState<LearningPath | null>(null);
  const [completedNodes, setCompletedNodes] = useState<Set<string>>(new Set());
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [showSubGraph, setShowSubGraph] = useState(false);

  // Show welcome toast on mount
  useEffect(() => {
    toast.success("Knowledge graph generated!", {
      description: `We've created a learning path for ${decodedSkill}`
    });
  }, [decodedSkill]);

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  // Highlight selected path
  useEffect(() => {
    if (selectedPath) {
      const updatedNodes = graphData.nodes.map(node => ({
        ...node,
        style: {
          ...node.style,
          opacity: selectedPath.nodeIds.includes(node.id) ? 1 : 0.3,
          border: selectedPath.nodeIds.includes(node.id) ? '2px solid #8b5cf6' : undefined,
        },
      }));
      setNodes(updatedNodes);

      const updatedEdges = graphData.edges.map(edge => ({
        ...edge,
        style: {
          ...edge.style,
          opacity: selectedPath.nodeIds.includes(edge.source) && selectedPath.nodeIds.includes(edge.target) ? 1 : 0.2,
        },
      }));
      setEdges(updatedEdges);
    } else {
      setNodes(graphData.nodes);
      setEdges(graphData.edges);
    }
  }, [selectedPath, graphData, setNodes, setEdges]);

  const toggleNodeCompletion = (nodeId: string) => {
    setCompletedNodes(prev => {
      const newSet = new Set(prev);
      if (newSet.has(nodeId)) {
        newSet.delete(nodeId);
        toast.info("Progress updated", {
          description: "Topic marked as incomplete"
        });
      } else {
        newSet.add(nodeId);
        toast.success("Great progress!", {
          description: "Topic marked as complete"
        });
      }
      return newSet;
    });
  };

  const handleNodeClick = (_event: React.MouseEvent, node: Node) => {
    setSelectedNode(node);
  };

  const openSubGraph = () => {
    if (selectedNode) {
      setShowSubGraph(true);
      toast.info("Sub-graph generated", {
        description: `Exploring: ${selectedNode.data.label}`
      });
    }
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return 'bg-green-100 text-green-800 border-green-300';
      case 'intermediate': return 'bg-yellow-100 text-yellow-800 border-yellow-300';
      case 'advanced': return 'bg-red-100 text-red-800 border-red-300';
      default: return 'bg-gray-100 text-gray-800 border-gray-300';
    }
  };

  const calculateProgress = () => {
    if (!selectedPath) return 0;
    const completedInPath = selectedPath.nodeIds.filter(id => completedNodes.has(id)).length;
    return (completedInPath / selectedPath.nodeIds.length) * 100;
  };

  return (
    <div className="h-screen flex flex-col bg-gray-50">
      <Toaster />
      {/* Header */}
      <header className="px-6 py-4 bg-white border-b shadow-sm z-10">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => navigate('/')}
              className="gap-2"
            >
              <ArrowLeft className="size-4" />
              <span className="hidden sm:inline">Back</span>
            </Button>
            <div className="flex items-center gap-2">
              <Brain className="size-6 text-purple-600" />
              <div>
                <h1 className="font-bold text-sm sm:text-lg">{decodedSkill}</h1>
                <p className="text-xs text-gray-500 hidden sm:block">AI-Generated Learning Path</p>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {/* Mobile menu */}
            <Sheet>
              <SheetTrigger asChild>
                <Button variant="outline" size="sm" className="lg:hidden">
                  <Menu className="size-4" />
                </Button>
              </SheetTrigger>
              <SheetContent side="left" className="w-80 p-0">
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

                  <TabsContent value="paths" className="p-4 space-y-3">
                    <div className="mb-4">
                      <h3 className="font-semibold mb-1 flex items-center gap-2">
                        <Sparkles className="size-4 text-purple-600" />
                        Choose Your Learning Path
                      </h3>
                      <p className="text-xs text-gray-500">
                        Select a path to highlight your journey through the knowledge graph
                      </p>
                    </div>
                    
                    {graphData.paths.map((path) => (
                      <motion.div
                        key={path.id}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                      >
                        <Card
                          className={`cursor-pointer transition-all ${
                            selectedPath?.id === path.id
                              ? 'ring-2 ring-purple-500 shadow-md'
                              : 'hover:shadow-md'
                          }`}
                          onClick={() => setSelectedPath(selectedPath?.id === path.id ? null : path)}
                        >
                          <CardHeader className="pb-3">
                            <div className="flex items-start justify-between">
                              <CardTitle className="text-sm">{path.name}</CardTitle>
                              <Badge className={getDifficultyColor(path.difficulty)} variant="outline">
                                {path.difficulty}
                              </Badge>
                            </div>
                            <CardDescription className="text-xs">{path.description}</CardDescription>
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

                    {selectedPath && (
                      <Button
                        variant="outline"
                        size="sm"
                        className="w-full"
                        onClick={() => setSelectedPath(null)}
                      >
                        Clear Selection
                      </Button>
                    )}
                  </TabsContent>

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
                          <span className={`text-sm ${completedNodes.has(node.id) ? 'line-through text-gray-500' : ''}`}>
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
                      <Progress value={(completedNodes.size / graphData.nodes.length) * 100} />
                    </div>
                  </TabsContent>
                </Tabs>
              </SheetContent>
            </Sheet>
            
            {selectedPath && (
              <div className="hidden sm:flex items-center gap-4">
                <div className="text-right">
                  <p className="text-sm font-medium">{selectedPath.name}</p>
                  <p className="text-xs text-gray-500">Progress: {Math.round(calculateProgress())}%</p>
                </div>
                <Progress value={calculateProgress()} className="w-32" />
              </div>
            )}
          </div>
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        <aside className="w-80 bg-white border-r overflow-y-auto hidden lg:block">
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

            <TabsContent value="paths" className="p-4 space-y-3">
              <div className="mb-4">
                <h3 className="font-semibold mb-1 flex items-center gap-2">
                  <Sparkles className="size-4 text-purple-600" />
                  Choose Your Learning Path
                </h3>
                <p className="text-xs text-gray-500">
                  Select a path to highlight your journey through the knowledge graph
                </p>
              </div>
              
              {graphData.paths.map((path) => (
                <motion.div
                  key={path.id}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <Card
                    className={`cursor-pointer transition-all ${
                      selectedPath?.id === path.id
                        ? 'ring-2 ring-purple-500 shadow-md'
                        : 'hover:shadow-md'
                    }`}
                    onClick={() => setSelectedPath(selectedPath?.id === path.id ? null : path)}
                  >
                    <CardHeader className="pb-3">
                      <div className="flex items-start justify-between">
                        <CardTitle className="text-sm">{path.name}</CardTitle>
                        <Badge className={getDifficultyColor(path.difficulty)} variant="outline">
                          {path.difficulty}
                        </Badge>
                      </div>
                      <CardDescription className="text-xs">{path.description}</CardDescription>
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

              {selectedPath && (
                <Button
                  variant="outline"
                  size="sm"
                  className="w-full"
                  onClick={() => setSelectedPath(null)}
                >
                  Clear Selection
                </Button>
              )}
            </TabsContent>

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
                    <span className={`text-sm ${completedNodes.has(node.id) ? 'line-through text-gray-500' : ''}`}>
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
                <Progress value={(completedNodes.size / graphData.nodes.length) * 100} />
              </div>
            </TabsContent>
          </Tabs>
        </aside>

        {/* Main Graph Area */}
        <main className="flex-1 relative">
          <ReactFlow
            nodes={nodes}
            edges={edges}
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
            <Panel position="top-right" className="bg-white p-4 rounded-lg shadow-lg m-4 max-w-xs">
              <div className="space-y-2">
                <h3 className="font-semibold text-sm flex items-center gap-2">
                  <Brain className="size-4 text-purple-600" />
                  Graph Controls
                </h3>
                <p className="text-xs text-gray-600">
                  • Click nodes to view details
                  <br />
                  • Select a path to highlight
                  <br />
                  • Track your progress
                </p>
              </div>
            </Panel>
          </ReactFlow>
        </main>
      </div>

      {/* Node Detail Dialog */}
      <Dialog open={!!selectedNode} onOpenChange={(open) => !open && setSelectedNode(null)}>
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
            <div>
              <h4 className="font-medium mb-2">About this topic:</h4>
              <p className="text-sm text-gray-600">
                This is a key concept in {decodedSkill}. Click below to generate a detailed sub-graph
                that breaks down this topic into smaller, manageable learning units.
              </p>
            </div>
            
            <div className="flex gap-2">
              <Button
                onClick={() => {
                  toggleNodeCompletion(selectedNode!.id);
                }}
                variant="outline"
                className="flex-1"
              >
                {completedNodes.has(selectedNode?.id || '') ? (
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

      {/* Sub-Graph Dialog */}
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
          <div className="flex-1 border rounded-lg overflow-hidden">
            <ReactFlow
              nodes={generateKnowledgeGraph(selectedNode?.data.label as string || "").nodes}
              edges={generateKnowledgeGraph(selectedNode?.data.label as string || "").edges}
              fitView
            >
              <Background />
              <Controls />
            </ReactFlow>
          </div>
          <p className="text-xs text-gray-500 text-center">
            This sub-graph shows a more detailed learning path for this specific topic
          </p>
        </DialogContent>
      </Dialog>
    </div>
  );
}
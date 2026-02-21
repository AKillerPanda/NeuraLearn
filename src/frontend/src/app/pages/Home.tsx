import { useState } from "react";
import { useNavigate } from "react-router";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { Brain, TrendingUp, Target, Sparkles } from "lucide-react";
import { motion } from "motion/react";

export function Home() {
  const [skill, setSkill] = useState("");
  const navigate = useNavigate();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (skill.trim()) {
      navigate(`/learn/${encodeURIComponent(skill.trim())}`);
    }
  };

  const exampleSkills = [
    "Machine Learning",
    "React Development",
    "Data Structures",
    "Public Speaking",
    "Photography",
    "Spanish Language"
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 flex flex-col">
      <header className="px-6 py-4 border-b bg-white/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-6xl mx-auto flex items-center gap-2">
          <Brain className="size-8 text-blue-600" />
          <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            NeuraLearn
          </h1>
        </div>
      </header>

      <main className="flex-1 px-6 py-12 max-w-6xl mx-auto w-full">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center mb-12"
        >
          <h2 className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            Master Any Skill with AI-Powered Learning Paths
          </h2>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Enter any skill you want to learn, and we'll generate a personalized knowledge graph with multiple learning paths tailored just for you.
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <Card className="max-w-2xl mx-auto shadow-lg mb-12">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Sparkles className="size-5 text-purple-600" />
                What do you want to learn?
              </CardTitle>
              <CardDescription>
                Type any skill, topic, or subject you'd like to master
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="flex gap-2">
                <Input
                  type="text"
                  placeholder="e.g., Web Development, Calculus, Guitar..."
                  value={skill}
                  onChange={(e) => setSkill(e.target.value)}
                  className="flex-1"
                />
                <Button type="submit" size="lg" className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700">
                  Generate Path
                </Button>
              </form>

              <div className="mt-6">
                <p className="text-sm text-gray-500 mb-3">Try these examples:</p>
                <div className="flex flex-wrap gap-2">
                  {exampleSkills.map((example) => (
                    <Button
                      key={example}
                      variant="outline"
                      size="sm"
                      onClick={() => setSkill(example)}
                      className="hover:border-purple-400 hover:text-purple-600"
                    >
                      {example}
                    </Button>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="grid md:grid-cols-3 gap-6"
        >
          <Card className="hover:shadow-md transition-shadow border-blue-200">
            <CardHeader>
              <Brain className="size-10 text-blue-600 mb-2" />
              <CardTitle>AI-Powered Graphs</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-600">
                Our AI creates comprehensive knowledge graphs that break down complex skills into manageable learning nodes.
              </p>
            </CardContent>
          </Card>

          <Card className="hover:shadow-md transition-shadow border-green-200">
            <CardHeader>
              <TrendingUp className="size-10 text-green-600 mb-2" />
              <CardTitle>Multiple Paths</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-600">
                Choose from different learning paths based on your goals, timeline, and preferred learning style.
              </p>
            </CardContent>
          </Card>

          <Card className="hover:shadow-md transition-shadow border-purple-200">
            <CardHeader>
              <Target className="size-10 text-purple-600 mb-2" />
              <CardTitle>Sub-Skill Deep Dives</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-600">
                Click any node to generate a detailed sub-graph and explore specific topics in greater depth.
              </p>
            </CardContent>
          </Card>
        </motion.div>
      </main>

      <footer className="px-6 py-6 border-t bg-white/50 backdrop-blur-sm">
        <div className="max-w-6xl mx-auto text-center text-gray-500 text-sm">
          <p>© 2026 NeuraLearn - Empowering students to learn smarter, not harder</p>
        </div>
      </footer>
    </div>
  );
}
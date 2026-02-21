import { useState, useEffect, useRef, useCallback } from "react";
import { useNavigate } from "react-router";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { Brain, TrendingUp, Target, Sparkles, Loader2 } from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import { spellCheck } from "../utils/api";
import type { SpellResult } from "../utils/api";

/* ── Debounce hook ────────────────────────────────────────────────── */
function useDebounce<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(id);
  }, [value, delayMs]);
  return debounced;
}

/* ── Constants ────────────────────────────────────────────────────── */
const EXAMPLE_SKILLS = [
  "Machine Learning",
  "React Development",
  "Data Structures",
  "Public Speaking",
  "Photography",
  "Piano",
] as const;

/* ── Component ────────────────────────────────────────────────────── */
export function Home() {
  const [skill, setSkill] = useState("");
  const [spellResults, setSpellResults] = useState<SpellResult[]>([]);
  const [checking, setChecking] = useState(false);
  const navigate = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);

  // Debounced spell-check (fires 500 ms after the user stops typing)
  const debouncedSkill = useDebounce(skill, 500);

  useEffect(() => {
    const text = debouncedSkill.trim();
    if (text.length < 2) {
      setSpellResults([]);
      return;
    }

    let cancelled = false;
    setChecking(true);

    spellCheck(text, 3)
      .then((results) => {
        if (!cancelled) setSpellResults(results);
      })
      .catch(() => {
        if (!cancelled) setSpellResults([]);
      })
      .finally(() => {
        if (!cancelled) setChecking(false);
      });

    return () => { cancelled = true; };
  }, [debouncedSkill]);

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      if (skill.trim()) {
        navigate(`/learn/${encodeURIComponent(skill.trim())}`);
      }
    },
    [skill, navigate],
  );

  const applySuggestion = useCallback(
    (original: string, replacement: string) => {
      setSkill((prev) => prev.replace(original, replacement));
      inputRef.current?.focus();
    },
    [],
  );

  // Are there misspelled words?
  const misspelled = spellResults.filter((r) => !r.inDictionary);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 flex flex-col">
      {/* ── Header ─────────────────────────────────────────────────── */}
      <header className="px-6 py-4 border-b bg-white/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-6xl mx-auto flex items-center gap-2">
          <Brain className="size-8 text-blue-600" />
          <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            NeuraLearn
          </h1>
        </div>
      </header>

      {/* ── Main ───────────────────────────────────────────────────── */}
      <main className="flex-1 px-6 py-12 max-w-6xl mx-auto w-full">
        {/* Hero */}
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
            Enter any skill you want to learn, and we'll generate a personalised
            knowledge graph with optimal learning paths tailored just for you.
          </p>
        </motion.div>

        {/* Search card */}
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
                Type any skill, topic, or subject — we'll auto-correct typos
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="flex gap-2">
                <div className="relative flex-1">
                  <Input
                    ref={inputRef}
                    type="text"
                    placeholder="e.g., Web Development, Calculus, Guitar..."
                    value={skill}
                    onChange={(e) => setSkill(e.target.value)}
                    className="w-full pr-8"
                    autoFocus
                  />
                  {checking && (
                    <Loader2 className="absolute right-2 top-1/2 -translate-y-1/2 size-4 text-purple-400 animate-spin" />
                  )}
                </div>
                <Button
                  type="submit"
                  size="lg"
                  className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
                >
                  Generate Path
                </Button>
              </form>

              {/* Spell-check suggestions */}
              <AnimatePresence>
                {misspelled.length > 0 && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className="mt-3 overflow-hidden"
                  >
                    <div className="rounded-md border border-amber-200 bg-amber-50 p-3 text-sm">
                      <p className="font-medium text-amber-800 mb-2">
                        Did you mean?
                      </p>
                      {misspelled.map((r) => (
                        <div key={r.original} className="flex items-center gap-2 mb-1 last:mb-0">
                          <span className="text-amber-700 line-through">{r.original}</span>
                          <span className="text-gray-400">&rarr;</span>
                          <div className="flex flex-wrap gap-1">
                            {r.suggestions.slice(0, 3).map((s) => (
                              <button
                                key={s.word}
                                type="button"
                                onClick={() => applySuggestion(r.original, s.word)}
                                className="px-2 py-0.5 rounded bg-amber-100 hover:bg-amber-200 text-amber-900 font-medium transition-colors"
                              >
                                {s.word}
                              </button>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Example chips */}
              <div className="mt-6">
                <p className="text-sm text-gray-500 mb-3">Try these examples:</p>
                <div className="flex flex-wrap gap-2">
                  {EXAMPLE_SKILLS.map((example) => (
                    <Button
                      key={example}
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        setSkill(example);
                        navigate(`/learn/${encodeURIComponent(example)}`);
                      }}
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

        {/* Feature cards */}
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
                Real-time webscraping builds comprehensive knowledge graphs that
                break down complex skills into manageable learning nodes.
              </p>
            </CardContent>
          </Card>

          <Card className="hover:shadow-md transition-shadow border-green-200">
            <CardHeader>
              <TrendingUp className="size-10 text-green-600 mb-2" />
              <CardTitle>ACO-Optimised Paths</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-600">
                Ant Colony Optimisation discovers the most efficient learning
                order — respecting every prerequisite relationship.
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
                Click any node to generate a detailed sub-graph and explore
                specific topics in greater depth.
              </p>
            </CardContent>
          </Card>
        </motion.div>
      </main>

      {/* ── Footer ─────────────────────────────────────────────────── */}
      <footer className="px-6 py-6 border-t bg-white/50 backdrop-blur-sm">
        <div className="max-w-6xl mx-auto text-center text-gray-500 text-sm">
          <p>&copy; 2026 NeuraLearn &mdash; Empowering students to learn smarter, not harder</p>
        </div>
      </footer>
    </div>
  );
}
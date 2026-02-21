import { useState, useEffect, useRef, useCallback, memo } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Play, Pause, RotateCcw, Coffee, Brain, Timer } from "lucide-react";
import { Button } from "./ui/button";
import { toast } from "sonner";

/* ── Configuration ─────────────────────────────────────────────── */
const PRESETS = {
  focus:      25 * 60,   // 25 min
  shortBreak:  5 * 60,   //  5 min
  longBreak:  15 * 60,   // 15 min
} as const;

type Mode = "focus" | "shortBreak" | "longBreak";

const MODE_LABELS: Record<Mode, string> = {
  focus: "Focus",
  shortBreak: "Short Break",
  longBreak: "Long Break",
};

const MODE_COLORS: Record<Mode, string> = {
  focus: "text-purple-600",
  shortBreak: "text-green-600",
  longBreak: "text-blue-600",
};

const MODE_BG: Record<Mode, string> = {
  focus: "bg-purple-50 border-purple-200",
  shortBreak: "bg-green-50 border-green-200",
  longBreak: "bg-blue-50 border-blue-200",
};

const MODE_RING: Record<Mode, string> = {
  focus: "stroke-purple-500",
  shortBreak: "stroke-green-500",
  longBreak: "stroke-blue-500",
};

/* ── SVG constants (hoisted to module level) ───────────────────── */
const SVG_RADIUS = 54;
const SVG_CIRCUMFERENCE = 2 * Math.PI * SVG_RADIUS;

/* ── Component ─────────────────────────────────────────────────── */
export const PomodoroTimer = memo(function PomodoroTimer({ onSessionComplete }: { onSessionComplete?: () => void }) {
  const [mode, setMode] = useState<Mode>("focus");
  const [secondsLeft, setSecondsLeft] = useState(PRESETS.focus);
  const [running, setRunning] = useState(false);
  const [sessions, setSessions] = useState(0);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const onCompleteRef = useRef(onSessionComplete);

  // Keep ref in sync without restarting intervals
  useEffect(() => { onCompleteRef.current = onSessionComplete; }, [onSessionComplete]);

  const totalSeconds = PRESETS[mode];
  const progress = 1 - secondsLeft / totalSeconds;
  const dashOffset = SVG_CIRCUMFERENCE * (1 - progress);

  const reset = useCallback(() => {
    setRunning(false);
    setSecondsLeft(PRESETS[mode]);
  }, [mode]);

  const switchMode = useCallback((newMode: Mode) => {
    setRunning(false);
    setMode(newMode);
    setSecondsLeft(PRESETS[newMode]);
  }, []);

  // Tick — uses functional updaters & refs to avoid stale closures
  useEffect(() => {
    if (!running) {
      if (intervalRef.current) clearInterval(intervalRef.current);
      return;
    }
    intervalRef.current = setInterval(() => {
      setSecondsLeft((prev) => {
        if (prev <= 1) {
          setRunning(false);
          if (mode === "focus") {
            setSessions((s) => {
              const newCount = s + 1;
              onCompleteRef.current?.();
              toast.success(`Focus session #${newCount} complete!`, {
                description: newCount % 4 === 0 ? "Take a long break — you've earned it!" : "Time for a short break!",
              });
              const nextMode = newCount % 4 === 0 ? "longBreak" : "shortBreak";
              setMode(nextMode);
              setSecondsLeft(PRESETS[nextMode]);
              return newCount;
            });
            return prev; // will be overwritten by setSecondsLeft inside setSessions
          } else {
            toast.info("Break over!", { description: "Ready for another focus session?" });
            setMode("focus");
            return PRESETS.focus;
          }
        }
        return prev - 1;
      });
    }, 1000);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [running, mode]);

  const mm = String(Math.floor(secondsLeft / 60)).padStart(2, "0");
  const ss = String(secondsLeft % 60).padStart(2, "0");

  return (
    <div className={`rounded-xl border p-4 ${MODE_BG[mode]} transition-colors duration-300`}>
      {/* Mode tabs */}
      <div className="flex gap-1 mb-4 justify-center">
        {(["focus", "shortBreak", "longBreak"] as Mode[]).map((m) => (
          <button
            key={m}
            onClick={() => switchMode(m)}
            className={`px-3 py-1 rounded-full text-xs font-medium transition-all ${
              mode === m
                ? `${MODE_COLORS[m]} bg-white shadow-sm`
                : "text-gray-500 hover:text-gray-700"
            }`}
          >
            {m === "focus" ? <Brain className="size-3 inline mr-1" /> : <Coffee className="size-3 inline mr-1" />}
            {MODE_LABELS[m]}
          </button>
        ))}
      </div>

      {/* Circular timer */}
      <div className="flex justify-center mb-3">
        <div className="relative w-32 h-32">
          <svg className="w-full h-full -rotate-90" viewBox="0 0 120 120">
            <circle
              cx="60" cy="60" r={SVG_RADIUS}
              fill="none"
              stroke="currentColor"
              strokeWidth="6"
              className="text-gray-200"
            />
            <motion.circle
              cx="60" cy="60" r={SVG_RADIUS}
              fill="none"
              strokeWidth="6"
              strokeLinecap="round"
              className={MODE_RING[mode]}
              strokeDasharray={SVG_CIRCUMFERENCE}
              strokeDashoffset={dashOffset}
              transition={{ duration: 0.5 }}
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className={`text-2xl font-bold font-mono ${MODE_COLORS[mode]}`}>
              {mm}:{ss}
            </span>
            <span className="text-[10px] text-gray-500 uppercase tracking-wider mt-0.5">
              {MODE_LABELS[mode]}
            </span>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="flex gap-2 justify-center">
        <Button
          size="sm"
          variant="outline"
          onClick={reset}
          className="gap-1"
        >
          <RotateCcw className="size-3" />
        </Button>
        <Button
          size="sm"
          onClick={() => setRunning(!running)}
          className={`gap-1 min-w-[80px] ${
            mode === "focus"
              ? "bg-purple-600 hover:bg-purple-700"
              : mode === "shortBreak"
              ? "bg-green-600 hover:bg-green-700"
              : "bg-blue-600 hover:bg-blue-700"
          }`}
        >
          {running ? <Pause className="size-3" /> : <Play className="size-3" />}
          {running ? "Pause" : "Start"}
        </Button>
      </div>

      {/* Session counter */}
      <AnimatePresence>
        {sessions > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-3 text-center"
          >
            <div className="flex items-center justify-center gap-1.5 text-xs text-gray-600">
              <Timer className="size-3" />
              <span>{sessions} session{sessions !== 1 ? "s" : ""} today</span>
              <span className="text-gray-300">•</span>
              <span className="font-medium">{sessions * 25} min focused</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
});

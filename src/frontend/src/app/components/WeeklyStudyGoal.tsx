import { useState, useEffect, useCallback, memo } from "react";
import { Button } from "./ui/button";
import { Progress } from "./ui/progress";
import { Target, TrendingUp, Minus, Plus } from "lucide-react";

/* ── Types ────────────────────────────────────────────────────────── */
interface WeeklyGoalData {
  target: number;       // topics per week
  weekStart: string;    // ISO date of current week's Monday
  completed: number;    // topics mastered this week
}

const STORAGE_KEY = "neuralearn_weekly_goal";

/* ── Helpers ──────────────────────────────────────────────────────── */
function getMonday(): string {
  const d = new Date();
  const day = d.getDay();
  const diff = d.getDate() - day + (day === 0 ? -6 : 1);
  const monday = new Date(d.setDate(diff));
  return monday.toISOString().slice(0, 10);
}

function loadGoal(): WeeklyGoalData {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const data = JSON.parse(raw) as WeeklyGoalData;
      // Reset if it's a new week
      const currentMonday = getMonday();
      if (data.weekStart !== currentMonday) {
        return { target: data.target, weekStart: currentMonday, completed: 0 };
      }
      return data;
    }
  } catch { /* ignore */ }
  return { target: 5, weekStart: getMonday(), completed: 0 };
}

function saveGoal(data: WeeklyGoalData) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
}

/* ── Component ────────────────────────────────────────────────────── */
export const WeeklyStudyGoal = memo(function WeeklyStudyGoal() {
  const [goal, setGoal] = useState<WeeklyGoalData>(loadGoal);

  // Expose global increment for when topics are mastered
  useEffect(() => {
    (window as unknown as Record<string, unknown>).__incrementWeeklyGoal = () => {
      setGoal((prev) => {
        const updated = { ...prev, completed: prev.completed + 1 };
        saveGoal(updated);
        return updated;
      });
    };
    return () => {
      delete (window as unknown as Record<string, unknown>).__incrementWeeklyGoal;
    };
  }, []);

  const adjustTarget = useCallback((delta: number) => {
    setGoal((prev) => {
      const newTarget = Math.max(1, Math.min(50, prev.target + delta));
      const updated = { ...prev, target: newTarget };
      saveGoal(updated);
      return updated;
    });
  }, []);

  const pct = goal.target > 0 ? Math.min((goal.completed / goal.target) * 100, 100) : 0;
  const isComplete = goal.completed >= goal.target;
  const daysLeft = 7 - ((new Date().getDay() + 6) % 7); // days until next Monday

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-sm flex items-center gap-1.5">
          <Target className="size-4 text-purple-600" />
          Weekly Goal
        </h3>
        {isComplete && (
          <span className="text-[10px] font-bold text-green-600 bg-green-50 px-2 py-0.5 rounded-full">
            COMPLETE!
          </span>
        )}
      </div>

      {/* Target selector */}
      <div className="flex items-center justify-between">
        <span className="text-xs text-gray-500">Target topics/week</span>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            className="h-6 w-6 p-0"
            onClick={() => adjustTarget(-1)}
          >
            <Minus className="size-3" />
          </Button>
          <span className="w-8 text-center text-sm font-bold">{goal.target}</span>
          <Button
            variant="ghost"
            size="sm"
            className="h-6 w-6 p-0"
            onClick={() => adjustTarget(1)}
          >
            <Plus className="size-3" />
          </Button>
        </div>
      </div>

      {/* Progress */}
      <div>
        <div className="flex justify-between text-xs mb-1">
          <span className="text-gray-600">
            {goal.completed} / {goal.target} topics
          </span>
          <span className={isComplete ? "text-green-600 font-medium" : "text-gray-400"}>
            {Math.round(pct)}%
          </span>
        </div>
        <Progress value={pct} className="h-2.5" />
      </div>

      {/* Stats */}
      <div className="flex justify-between text-[11px] text-gray-500">
        <span className="flex items-center gap-1">
          <TrendingUp className="size-3" />
          {daysLeft} day{daysLeft !== 1 ? "s" : ""} left this week
        </span>
        {!isComplete && goal.completed > 0 && (
          <span>{goal.target - goal.completed} more to go</span>
        )}
      </div>
    </div>
  );
});

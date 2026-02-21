import { useMemo } from "react";
import { motion } from "motion/react";
import { Trophy, Star, Flame, Zap, Crown, Award, Rocket, Target } from "lucide-react";

/* ── XP & Level calculation ────────────────────────────────────── */
const XP_PER_TOPIC: Record<string, number> = {
  foundational: 50,
  beginner:     50,
  intermediate: 100,
  advanced:     200,
  expert:       350,
};

function xpForLevel(level: number): number {
  // XP needed to reach this level (quadratic scaling)
  return Math.floor(100 * level * (level + 1) / 2);
}

function levelFromXP(xp: number): { level: number; current: number; needed: number; progress: number } {
  let level = 1;
  while (xpForLevel(level + 1) <= xp) level++;
  const currentThreshold = xpForLevel(level);
  const nextThreshold = xpForLevel(level + 1);
  const current = xp - currentThreshold;
  const needed = nextThreshold - currentThreshold;
  return { level, current, needed, progress: current / needed };
}

/* ── Achievement definitions ───────────────────────────────────── */
interface Achievement {
  id: string;
  name: string;
  description: string;
  icon: typeof Trophy;
  color: string;
  check: (stats: GamificationStats) => boolean;
}

interface GamificationStats {
  topicsCompleted: number;
  totalTopics: number;
  pomodoroSessions: number;
  xp: number;
  streak: number;
}

const ACHIEVEMENTS: Achievement[] = [
  {
    id: "first-step",
    name: "First Step",
    description: "Complete your first topic",
    icon: Star,
    color: "text-yellow-500",
    check: (s) => s.topicsCompleted >= 1,
  },
  {
    id: "five-star",
    name: "Rising Scholar",
    description: "Complete 5 topics",
    icon: Award,
    color: "text-blue-500",
    check: (s) => s.topicsCompleted >= 5,
  },
  {
    id: "halfway",
    name: "Halfway There",
    description: "Complete 50% of all topics",
    icon: Target,
    color: "text-purple-500",
    check: (s) => s.totalTopics > 0 && s.topicsCompleted >= s.totalTopics / 2,
  },
  {
    id: "completionist",
    name: "Completionist",
    description: "Master every topic in the graph",
    icon: Crown,
    color: "text-amber-500",
    check: (s) => s.totalTopics > 0 && s.topicsCompleted >= s.totalTopics,
  },
  {
    id: "focused",
    name: "Deep Focus",
    description: "Complete 4 Pomodoro sessions",
    icon: Flame,
    color: "text-red-500",
    check: (s) => s.pomodoroSessions >= 4,
  },
  {
    id: "marathon",
    name: "Study Marathon",
    description: "Complete 8 Pomodoro sessions",
    icon: Rocket,
    color: "text-orange-500",
    check: (s) => s.pomodoroSessions >= 8,
  },
  {
    id: "streak-3",
    name: "On a Roll",
    description: "3-day study streak",
    icon: Zap,
    color: "text-emerald-500",
    check: (s) => s.streak >= 3,
  },
  {
    id: "streak-7",
    name: "Weekly Warrior",
    description: "7-day study streak",
    icon: Trophy,
    color: "text-indigo-500",
    check: (s) => s.streak >= 7,
  },
];

/* ── Component ────────────────────────────────────────────────── */
export function GamificationPanel({
  topicsCompleted,
  totalTopics,
  pomodoroSessions,
}: {
  topicsCompleted: number;
  totalTopics: number;
  pomodoroSessions: number;
}) {
  // Calculate XP from completed topics (estimate difficulty spread)
  const xp = useMemo(() => {
    // Simple: 100 XP per topic + bonus per pomodoro
    return topicsCompleted * 100 + pomodoroSessions * 25;
  }, [topicsCompleted, pomodoroSessions]);

  const levelInfo = useMemo(() => levelFromXP(xp), [xp]);

  // Streak: stored in localStorage
  const streak = useMemo(() => {
    try {
      const data = JSON.parse(localStorage.getItem("neuralearn_streak") ?? "{}");
      const today = new Date().toISOString().slice(0, 10);
      if (data.lastDate === today) return data.count ?? 1;
      // Check if yesterday
      const yesterday = new Date(Date.now() - 86400000).toISOString().slice(0, 10);
      if (data.lastDate === yesterday) {
        const newCount = (data.count ?? 0) + 1;
        localStorage.setItem("neuralearn_streak", JSON.stringify({ lastDate: today, count: newCount }));
        return newCount;
      }
      // Streak broken
      if (topicsCompleted > 0 || pomodoroSessions > 0) {
        localStorage.setItem("neuralearn_streak", JSON.stringify({ lastDate: today, count: 1 }));
        return 1;
      }
      return 0;
    } catch {
      return 0;
    }
  }, [topicsCompleted, pomodoroSessions]);

  const stats: GamificationStats = { topicsCompleted, totalTopics, pomodoroSessions, xp, streak };
  const earned = ACHIEVEMENTS.filter((a) => a.check(stats));
  const locked = ACHIEVEMENTS.filter((a) => !a.check(stats));

  return (
    <div className="space-y-4">
      {/* Level & XP bar */}
      <div className="rounded-xl border bg-gradient-to-r from-purple-50 to-blue-50 p-4">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <div className="w-10 h-10 rounded-full bg-purple-600 flex items-center justify-center">
              <span className="text-white font-bold text-sm">{levelInfo.level}</span>
            </div>
            <div>
              <p className="font-semibold text-sm text-gray-800">Level {levelInfo.level}</p>
              <p className="text-[10px] text-gray-500">{xp} XP total</p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-xs text-gray-600">{levelInfo.current} / {levelInfo.needed} XP</p>
            <p className="text-[10px] text-gray-400">to next level</p>
          </div>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2.5 overflow-hidden">
          <motion.div
            className="h-full rounded-full bg-gradient-to-r from-purple-500 to-blue-500"
            initial={{ width: 0 }}
            animate={{ width: `${levelInfo.progress * 100}%` }}
            transition={{ duration: 0.8 }}
          />
        </div>
      </div>

      {/* Quick stats row */}
      <div className="grid grid-cols-3 gap-2">
        <div className="rounded-lg border bg-amber-50 p-2 text-center">
          <Flame className="size-4 text-amber-500 mx-auto mb-0.5" />
          <p className="text-lg font-bold text-amber-700">{streak}</p>
          <p className="text-[9px] text-amber-600">Day Streak</p>
        </div>
        <div className="rounded-lg border bg-purple-50 p-2 text-center">
          <Star className="size-4 text-purple-500 mx-auto mb-0.5" />
          <p className="text-lg font-bold text-purple-700">{earned.length}</p>
          <p className="text-[9px] text-purple-600">Badges</p>
        </div>
        <div className="rounded-lg border bg-green-50 p-2 text-center">
          <Trophy className="size-4 text-green-500 mx-auto mb-0.5" />
          <p className="text-lg font-bold text-green-700">{topicsCompleted}</p>
          <p className="text-[9px] text-green-600">Mastered</p>
        </div>
      </div>

      {/* Achievements */}
      <div>
        <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
          Achievements ({earned.length}/{ACHIEVEMENTS.length})
        </h4>
        <div className="grid grid-cols-2 gap-1.5">
          {earned.map((a) => (
            <motion.div
              key={a.id}
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className="flex items-center gap-2 p-2 rounded-lg border bg-white shadow-sm"
            >
              <a.icon className={`size-5 ${a.color} flex-shrink-0`} />
              <div className="min-w-0">
                <p className="text-[11px] font-semibold truncate">{a.name}</p>
                <p className="text-[9px] text-gray-500 truncate">{a.description}</p>
              </div>
            </motion.div>
          ))}
          {locked.map((a) => (
            <div
              key={a.id}
              className="flex items-center gap-2 p-2 rounded-lg border bg-gray-50 opacity-40"
            >
              <a.icon className="size-5 text-gray-400 flex-shrink-0" />
              <div className="min-w-0">
                <p className="text-[11px] font-semibold truncate text-gray-500">{a.name}</p>
                <p className="text-[9px] text-gray-400 truncate">{a.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

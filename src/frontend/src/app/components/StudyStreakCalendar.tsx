import { useState, useEffect, useMemo, memo } from "react";

/* ── Types ────────────────────────────────────────────────────────── */
interface StreakDay {
  date: string;   // "YYYY-MM-DD"
  count: number;  // topics mastered that day
}

interface StudyStreakData {
  days: StreakDay[];
  currentStreak: number;
  longestStreak: number;
  totalDays: number;
}

const STORAGE_KEY = "neuralearn_study_streak";
const WEEKS_TO_SHOW = 12;

/* ── Helpers ──────────────────────────────────────────────────────── */
function today(): string {
  return new Date().toISOString().slice(0, 10);
}

function loadStreak(): StudyStreakData {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw) as StudyStreakData;
  } catch { /* ignore */ }
  return { days: [], currentStreak: 0, longestStreak: 0, totalDays: 0 };
}

function saveStreak(data: StudyStreakData) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
}

function computeStreaks(days: StreakDay[]): { current: number; longest: number } {
  if (days.length === 0) return { current: 0, longest: 0 };
  const sorted = [...days].sort((a, b) => a.date.localeCompare(b.date));
  let longest = 1;
  let current = 1;
  let streak = 1;

  for (let i = 1; i < sorted.length; i++) {
    const prev = new Date(sorted[i - 1].date);
    const curr = new Date(sorted[i].date);
    const diff = (curr.getTime() - prev.getTime()) / (1000 * 60 * 60 * 24);
    if (diff === 1) {
      streak++;
    } else if (diff > 1) {
      streak = 1;
    }
    longest = Math.max(longest, streak);
  }
  current = streak;

  // Check if current streak extends to today or yesterday
  const lastDate = sorted[sorted.length - 1].date;
  const todayStr = today();
  const yesterday = new Date();
  yesterday.setDate(yesterday.getDate() - 1);
  const yesterdayStr = yesterday.toISOString().slice(0, 10);

  if (lastDate !== todayStr && lastDate !== yesterdayStr) {
    current = 0;
  }

  return { current, longest };
}

/* ── Calendar grid ────────────────────────────────────────────────── */
function getCalendarGrid(): string[][] {
  const endDate = new Date();
  // Make end date be the end of the current week (Saturday)
  const dayOfWeek = endDate.getDay();
  endDate.setDate(endDate.getDate() + (6 - dayOfWeek));

  const startDate = new Date(endDate);
  startDate.setDate(startDate.getDate() - WEEKS_TO_SHOW * 7 + 1);

  const weeks: string[][] = [];
  let currentWeek: string[] = [];
  const cursor = new Date(startDate);

  while (cursor <= endDate) {
    currentWeek.push(cursor.toISOString().slice(0, 10));
    if (currentWeek.length === 7) {
      weeks.push(currentWeek);
      currentWeek = [];
    }
    cursor.setDate(cursor.getDate() + 1);
  }
  if (currentWeek.length > 0) {
    weeks.push(currentWeek);
  }
  return weeks;
}

function getIntensity(count: number): string {
  if (count === 0) return "bg-gray-100 dark:bg-gray-700";
  if (count === 1) return "bg-green-200";
  if (count === 2) return "bg-green-300";
  if (count <= 4) return "bg-green-400";
  return "bg-green-500";
}

/* ── Component ────────────────────────────────────────────────────── */
export const StudyStreakCalendar = memo(function StudyStreakCalendar({ onStudyLogged }: { onStudyLogged?: () => void }) {
  const [data, setData] = useState<StudyStreakData>(loadStreak);

  // Expose a global function for other components to call when topics are mastered
  useEffect(() => {
    (window as unknown as Record<string, unknown>).__logStudyActivity = () => {
      setData((prev) => {
        const todayStr = today();
        const existing = prev.days.find((d) => d.date === todayStr);
        let newDays: StreakDay[];
        if (existing) {
          newDays = prev.days.map((d) =>
            d.date === todayStr ? { ...d, count: d.count + 1 } : d,
          );
        } else {
          newDays = [...prev.days, { date: todayStr, count: 1 }];
        }
        const { current, longest } = computeStreaks(newDays);
        const uniqueDays = new Set(newDays.map((d) => d.date)).size;
        const updated: StudyStreakData = {
          days: newDays,
          currentStreak: current,
          longestStreak: longest,
          totalDays: uniqueDays,
        };
        saveStreak(updated);
        onStudyLogged?.();
        // Notify GamificationPanel and other same-tab listeners
        window.dispatchEvent(new CustomEvent("neuralearn-streak-update"));
        return updated;
      });
    };
    return () => {
      delete (window as unknown as Record<string, unknown>).__logStudyActivity;
    };
  }, [onStudyLogged]);

  // Use stored streaks directly instead of recomputing
  const dayMap = useMemo(() => {
    const m = new Map<string, number>();
    for (const d of data.days) m.set(d.date, d.count);
    return m;
  }, [data.days]);
  const weeks = useMemo(getCalendarGrid, []);
  const todayStr = useMemo(today, []);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-sm">Study Streak</h3>
        <div className="flex gap-3 text-xs">
          <span>
            <span className="font-bold text-green-600">{data.currentStreak}</span> day streak
          </span>
          <span className="text-gray-400">|</span>
          <span>
            Best: <span className="font-bold">{data.longestStreak}</span>
          </span>
        </div>
      </div>

      {/* GitHub-style heatmap */}
      <div className="flex gap-[3px] overflow-x-auto">
        {weeks.map((week, wi) => (
          <div key={wi} className="flex flex-col gap-[3px]">
            {week.map((dateStr) => {
              const count = dayMap.get(dateStr) ?? 0;
              const isToday = dateStr === todayStr;
              const isFuture = dateStr > todayStr;
              return (
                <div
                  key={dateStr}
                  title={`${dateStr}: ${count} topic${count !== 1 ? "s" : ""}`}
                  className={`
                    w-[14px] h-[14px] rounded-sm transition-colors
                    ${isFuture ? "bg-transparent" : getIntensity(count)}
                    ${isToday ? "ring-1 ring-purple-400" : ""}
                  `}
                />
              );
            })}
          </div>
        ))}
      </div>

      {/* Legend */}
      <div className="flex items-center gap-1 text-[10px] text-gray-500">
        <span>Less</span>
        <div className="w-[10px] h-[10px] rounded-sm bg-gray-100" />
        <div className="w-[10px] h-[10px] rounded-sm bg-green-200" />
        <div className="w-[10px] h-[10px] rounded-sm bg-green-300" />
        <div className="w-[10px] h-[10px] rounded-sm bg-green-400" />
        <div className="w-[10px] h-[10px] rounded-sm bg-green-500" />
        <span>More</span>
        <span className="ml-2">{data.totalDays} total study days</span>
      </div>
    </div>
  );
});

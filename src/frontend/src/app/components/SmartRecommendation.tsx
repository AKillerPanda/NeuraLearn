import { memo } from "react";
import type { DifficultyRecommendation } from "../utils/api";
import { Sparkles, ArrowRight, Zap } from "lucide-react";

/* ── Difficulty bar colour ────────────────────────────────────────── */
function difficultyColor(d: number): string {
  if (d < 0.25) return "bg-green-400";
  if (d < 0.5) return "bg-yellow-400";
  if (d < 0.75) return "bg-orange-400";
  return "bg-red-400";
}

function difficultyLabel(d: number): string {
  if (d < 0.25) return "Easy";
  if (d < 0.5) return "Moderate";
  if (d < 0.75) return "Challenging";
  return "Hard";
}

/* ── Component ────────────────────────────────────────────────────── */
export const SmartRecommendation = memo(function SmartRecommendation({
  recommendations,
  onSelect,
}: {
  recommendations: DifficultyRecommendation[];
  onSelect?: (topicId: string) => void;
}) {
  if (recommendations.length === 0) {
    return (
      <div className="text-center py-4">
        <Sparkles className="size-8 text-purple-300 mx-auto mb-2" />
        <p className="text-sm text-gray-500">
          Master some topics to get personalised recommendations!
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Sparkles className="size-4 text-purple-600" />
        <h3 className="font-semibold text-sm">What's Next?</h3>
      </div>
      <p className="text-[11px] text-gray-500">
        AI-powered suggestions based on your progress and topic difficulty
      </p>

      <div className="space-y-2">
        {recommendations.map((rec, i) => (
          <button
            key={rec.topicId}
            onClick={() => onSelect?.(rec.topicId)}
            className="w-full text-left rounded-lg border p-3 hover:border-purple-300 hover:bg-purple-50/50 transition-colors group"
          >
            <div className="flex items-start justify-between gap-2">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  {i === 0 && (
                    <span className="flex items-center gap-0.5 text-[9px] font-bold text-purple-600 bg-purple-100 px-1.5 py-0.5 rounded-full">
                      <Zap className="size-2.5" />
                      TOP PICK
                    </span>
                  )}
                  <span className="text-sm font-medium truncate">{rec.name}</span>
                </div>
                <p className="text-[11px] text-gray-500 mt-0.5 line-clamp-2">
                  {rec.reason}
                </p>
              </div>

              <div className="flex flex-col items-end gap-1 shrink-0">
                {/* Difficulty indicator */}
                <div className="flex items-center gap-1.5">
                  <span className="text-[10px] text-gray-500">
                    {difficultyLabel(rec.difficulty)}
                  </span>
                  <div className="w-12 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${difficultyColor(rec.difficulty)}`}
                      style={{ width: `${rec.difficulty * 100}%` }}
                    />
                  </div>
                </div>
                <ArrowRight className="size-3.5 text-gray-300 group-hover:text-purple-500 transition-colors" />
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
});

import { memo } from "react";
import { Handle, Position } from "reactflow";

interface CustomNodeData {
  label: string;
  level?: string;
  difficulty?: string;
  mastered?: boolean;
  prerequisites?: string[];
  unlocks?: string[];
  depth?: number;
  cluster?: number;
  resources?: { title: string; url: string; source: string; type: string }[];
  difficultyScore?: number; // GAT-predicted difficulty 0-1
}

const LEVEL_BADGE: Record<string, string> = {
  foundational: "bg-green-100 text-green-700 border-green-200",
  intermediate: "bg-yellow-100 text-yellow-700 border-yellow-200",
  advanced: "bg-orange-100 text-orange-700 border-orange-200",
  expert: "bg-red-100 text-red-700 border-red-200",
};

const DEPTH_GLOW: Record<number, string> = {
  0: "shadow-green-200/60",
  1: "shadow-yellow-200/60",
  2: "shadow-orange-200/60",
  3: "shadow-red-200/60",
};

/* Spectral cluster → border/ring accent colour */
const CLUSTER_COLOURS: string[] = [
  "ring-blue-400/50",
  "ring-emerald-400/50",
  "ring-amber-400/50",
  "ring-rose-400/50",
  "ring-violet-400/50",
  "ring-cyan-400/50",
];

function CustomNodeInner({ data }: { data: CustomNodeData }) {
  const levelClass = LEVEL_BADGE[data.level ?? ""] ?? "bg-gray-100 text-gray-600 border-gray-200";
  const glowClass = DEPTH_GLOW[data.depth ?? 0] ?? "shadow-purple-200/60";
  const clusterRing = CLUSTER_COLOURS[(data.cluster ?? 0) % CLUSTER_COLOURS.length];
  const prereqs = data.prerequisites ?? [];
  const unlocks = data.unlocks ?? [];
  const isRoot = prereqs.length === 0;
  const isLeaf = unlocks.length === 0;
  const resourceCount = data.resources?.length ?? 0;

  return (
    <div
      className={`
        px-4 py-3 rounded-xl bg-white border-2 transition-all
        shadow-md ${glowClass} hover:shadow-lg ring-2 ${clusterRing}
        min-w-[180px] max-w-[240px]
        ${data.mastered
          ? "border-green-400 bg-green-50"
          : "border-purple-200 hover:border-purple-400"}
      `}
    >
      <Handle type="target" position={Position.Top} className="w-3 h-3" />

      {/* Title row */}
      <div className="font-semibold text-sm leading-tight">{data.label}</div>

      {/* Level badge + context */}
      <div className="flex items-center gap-1.5 mt-1.5 flex-wrap">
        {data.level && (
          <span className={`inline-block px-1.5 py-0.5 rounded text-[9px] font-medium border ${levelClass}`}>
            {data.level}
          </span>
        )}
        {isRoot && (
          <span className="inline-block px-1.5 py-0.5 rounded text-[9px] font-medium bg-blue-50 text-blue-600 border border-blue-200">
            start here
          </span>
        )}
        {isLeaf && (
          <span className="inline-block px-1.5 py-0.5 rounded text-[9px] font-medium bg-purple-50 text-purple-600 border border-purple-200">
            final
          </span>
        )}
        {resourceCount > 0 && (
          <span className="inline-block px-1.5 py-0.5 rounded text-[9px] font-medium bg-sky-50 text-sky-600 border border-sky-200">
            {resourceCount} link{resourceCount > 1 ? "s" : ""}
          </span>
        )}
      </div>

      {/* Quick prereq hint (only on non-root nodes) */}
      {prereqs.length > 0 && (
        <p className="text-[9px] text-gray-400 mt-1 leading-tight truncate">
          needs: {prereqs.join(", ")}
        </p>
      )}

      {/* Difficulty heatmap bar */}
      {data.difficultyScore !== undefined && (
        <div className="mt-1.5 flex items-center gap-1">
          <div className="flex-1 h-1 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all"
              style={{
                width: `${data.difficultyScore * 100}%`,
                backgroundColor:
                  data.difficultyScore < 0.25 ? "#4ade80" :
                  data.difficultyScore < 0.5 ? "#facc15" :
                  data.difficultyScore < 0.75 ? "#fb923c" :
                  "#f87171",
              }}
            />
          </div>
          <span className="text-[8px] text-gray-400 font-mono">
            {(data.difficultyScore * 100).toFixed(0)}%
          </span>
        </div>
      )}

      <Handle type="source" position={Position.Bottom} className="w-3 h-3" />
    </div>
  );
}

export const CustomNode = memo(CustomNodeInner);

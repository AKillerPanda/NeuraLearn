import { memo } from "react";
import { Handle, Position } from "reactflow";

interface CustomNodeData {
  label: string;
  level?: string;
  difficulty?: string;
  mastered?: boolean;
}

const LEVEL_BADGE: Record<string, string> = {
  foundational: "bg-green-100 text-green-700",
  intermediate: "bg-yellow-100 text-yellow-700",
  advanced: "bg-orange-100 text-orange-700",
  expert: "bg-red-100 text-red-700",
};

function CustomNodeInner({ data }: { data: CustomNodeData }) {
  const levelClass = LEVEL_BADGE[data.level ?? ""] ?? "bg-gray-100 text-gray-600";

  return (
    <div
      className={`px-4 py-2 shadow-md rounded-xl bg-white border-2 transition-colors ${
        data.mastered
          ? "border-green-400 bg-green-50"
          : "border-purple-200 hover:border-purple-400"
      }`}
    >
      <Handle type="target" position={Position.Top} className="w-3 h-3" />
      <div className="font-semibold text-sm leading-tight">{data.label}</div>
      {data.level && (
        <span
          className={`inline-block mt-1 px-1.5 py-0.5 rounded text-[10px] font-medium ${levelClass}`}
        >
          {data.level}
        </span>
      )}
      <Handle type="source" position={Position.Bottom} className="w-3 h-3" />
    </div>
  );
}

export const CustomNode = memo(CustomNodeInner);


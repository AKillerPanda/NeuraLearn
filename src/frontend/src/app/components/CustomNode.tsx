import { Handle, Position } from 'reactflow';

interface CustomNodeProps {
  data: {
    label: string;
  };
}

export function CustomNode({ data }: CustomNodeProps) {
  return (
    <div className="px-4 py-2 shadow-md rounded-lg bg-white border-2 border-purple-200 hover:border-purple-400 transition-colors">
      <Handle type="target" position={Position.Top} className="w-3 h-3" />
      <div className="font-medium text-sm">{data.label}</div>
      <Handle type="source" position={Position.Bottom} className="w-3 h-3" />
    </div>
  );
}

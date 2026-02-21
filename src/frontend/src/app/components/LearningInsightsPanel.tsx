import { memo } from "react";
import type { LearningInsights } from "../utils/api";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Shield, AlertTriangle, BookOpen, Layers } from "lucide-react";

/* ── Rating badge colours ─────────────────────────────────────────── */
const COHESION_COLOURS: Record<string, string> = {
  Strong: "bg-green-100 text-green-700 border-green-200",
  Moderate: "bg-yellow-100 text-yellow-700 border-yellow-200",
  Weak: "bg-orange-100 text-orange-700 border-orange-200",
  Disconnected: "bg-red-100 text-red-700 border-red-200",
};

const BOTTLENECK_COLOURS: Record<string, string> = {
  Low: "bg-green-100 text-green-700 border-green-200",
  Moderate: "bg-yellow-100 text-yellow-700 border-yellow-200",
  High: "bg-red-100 text-red-700 border-red-200",
};

const LOAD_COLOURS: Record<string, string> = {
  Light: "bg-green-100 text-green-700 border-green-200",
  Moderate: "bg-yellow-100 text-yellow-700 border-yellow-200",
  Heavy: "bg-red-100 text-red-700 border-red-200",
};

const SHAPE_COLOURS: Record<string, string> = {
  Deep: "bg-purple-100 text-purple-700 border-purple-200",
  Balanced: "bg-blue-100 text-blue-700 border-blue-200",
  Broad: "bg-cyan-100 text-cyan-700 border-cyan-200",
};

/* ── Component ────────────────────────────────────────────────────── */
export const LearningInsightsPanel = memo(function LearningInsightsPanel({ insights }: { insights: LearningInsights }) {
  return (
    <div className="space-y-2">
      {/* Curriculum Cohesion */}
      {insights.curriculumCohesion && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-1.5">
              <Shield className="size-3.5 text-blue-600" />
              Curriculum Cohesion
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-1.5">
            <span
              className={`inline-block px-2 py-0.5 rounded text-xs font-medium border ${
                COHESION_COLOURS[insights.curriculumCohesion.rating] ?? "bg-gray-100 text-gray-600"
              }`}
            >
              {insights.curriculumCohesion.rating}
            </span>
            <p className="text-xs text-gray-600">
              {insights.curriculumCohesion.description}
            </p>
          </CardContent>
        </Card>
      )}

      {/* Bottleneck Risk */}
      {insights.bottleneckRisk && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-1.5">
              <AlertTriangle className="size-3.5 text-amber-500" />
              Bottleneck Risk
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-1.5">
            <span
              className={`inline-block px-2 py-0.5 rounded text-xs font-medium border ${
                BOTTLENECK_COLOURS[insights.bottleneckRisk.rating] ?? "bg-gray-100 text-gray-600"
              }`}
            >
              {insights.bottleneckRisk.rating}
            </span>
            <p className="text-xs text-gray-600">
              {insights.bottleneckRisk.description}
            </p>
            {insights.bottleneckRisk.chokepoints &&
              insights.bottleneckRisk.chokepoints.length > 0 && (
                <div className="flex flex-wrap gap-1 mt-1">
                  {insights.bottleneckRisk.chokepoints.map((cp) => (
                    <span
                      key={cp}
                      className="text-[10px] bg-amber-50 text-amber-700 px-1.5 py-0.5 rounded border border-amber-200"
                    >
                      {cp}
                    </span>
                  ))}
                </div>
              )}
          </CardContent>
        </Card>
      )}

      {/* Prerequisite Load */}
      {insights.prerequisiteLoad && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-1.5">
              <BookOpen className="size-3.5 text-green-600" />
              Prerequisite Load
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-1.5">
            <span
              className={`inline-block px-2 py-0.5 rounded text-xs font-medium border ${
                LOAD_COLOURS[insights.prerequisiteLoad.rating] ?? "bg-gray-100 text-gray-600"
              }`}
            >
              {insights.prerequisiteLoad.rating}
            </span>
            <p className="text-xs text-gray-600">
              {insights.prerequisiteLoad.description}
            </p>
          </CardContent>
        </Card>
      )}

      {/* Curriculum Shape */}
      {insights.curriculumShape && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-1.5">
              <Layers className="size-3.5 text-purple-600" />
              Curriculum Shape
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-1.5">
            <span
              className={`inline-block px-2 py-0.5 rounded text-xs font-medium border ${
                SHAPE_COLOURS[insights.curriculumShape.type] ?? "bg-gray-100 text-gray-600"
              }`}
            >
              {insights.curriculumShape.type} ({insights.curriculumShape.depth} levels)
            </span>
            <p className="text-xs text-gray-600">
              {insights.curriculumShape.description}
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
});

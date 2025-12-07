import type { CompareResult } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { ChevronDown } from "lucide-react";

export interface CompareViewerProps {
  result: CompareResult | null;
}

export function CompareViewer({ result }: CompareViewerProps) {
  if (!result) {
    return (
      <div className="text-sm text-gray-500 py-4 text-center">
        Click "Compare Before vs After" to evaluate the best policy against baseline.
      </div>
    );
  }

  const improvement = result.best.score - result.baseline.score;

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold">Evaluation Results</h3>

      <Collapsible>
        <CollapsibleTrigger className="flex items-center gap-2 text-sm font-medium hover:underline">
          <ChevronDown className="h-4 w-4" />
          Context (Turn {result.env.turn_index})
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-2">
          <div className="bg-gray-50 p-3 rounded text-xs whitespace-pre-wrap">
            {result.env.prompt}
          </div>
          <div className="mt-2 text-xs text-gray-600">
            <strong>Ground Truth:</strong> {result.env.ground_truth}
          </div>
        </CollapsibleContent>
      </Collapsible>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm flex items-center justify-between">
            <span>Baseline ({result.baseline.policyId})</span>
            <Badge variant="secondary">Score: {result.baseline.score.toFixed(2)}</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-gray-700">{result.baseline.response}</p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm flex items-center justify-between">
            <span>Best Policy ({result.best.policyId})</span>
            <Badge variant={improvement > 0 ? "default" : "secondary"}>
              Score: {result.best.score.toFixed(2)}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-gray-700">{result.best.response}</p>
        </CardContent>
      </Card>

      <div className="text-center py-2">
        <div className="text-lg font-bold">
          Improvement:{" "}
          <span className={improvement > 0 ? "text-green-600" : "text-gray-600"}>
            {improvement > 0 ? "+" : ""}
            {improvement.toFixed(2)}
          </span>
        </div>
      </div>
    </div>
  );
}

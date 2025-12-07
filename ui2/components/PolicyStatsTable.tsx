import type { PolicyStats } from "@/lib/types";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

export interface PolicyStatsTableProps {
  stats: PolicyStats[];
}

export function PolicyStatsTable({ stats }: PolicyStatsTableProps) {
  if (stats.length === 0) {
    return (
      <div className="text-sm text-gray-500 py-4">
        No training data yet. Run training to see policy statistics.
      </div>
    );
  }

  const bestPolicyId = stats.reduce((best, curr) =>
    curr.avgReward > best.avgReward ? curr : best
  ).policyId;

  return (
    <div className="border rounded-lg overflow-hidden">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Policy</TableHead>
            <TableHead className="text-right">Trials</TableHead>
            <TableHead className="text-right">Avg Reward</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {stats.map((stat) => (
            <TableRow
              key={stat.policyId}
              className={
                stat.policyId === bestPolicyId ? "bg-green-50 font-semibold" : ""
              }
            >
              <TableCell>{stat.policyId}</TableCell>
              <TableCell className="text-right">{stat.trials}</TableCell>
              <TableCell className="text-right">
                {stat.avgReward.toFixed(2)}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

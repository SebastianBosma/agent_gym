"use client"

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface RewardChartProps {
  steps: { stepIndex: number; bestAvgReward: number }[];
}

export function RewardChart({ steps }: RewardChartProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Best Average Reward Over Time</CardTitle>
      </CardHeader>
      <CardContent>
        {steps.length === 0 ? (
          <div className="h-[200px] flex items-center justify-center text-muted-foreground">
            No training data yet
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={steps}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="stepIndex" 
                label={{ value: 'Step', position: 'insideBottom', offset: -5 }}
              />
              <YAxis 
                domain={[0, 1]}
                label={{ value: 'Reward', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip />
              <Line 
                type="monotone" 
                dataKey="bestAvgReward" 
                stroke="#8884d8" 
                strokeWidth={2}
                dot={{ r: 3 }}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}

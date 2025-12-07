"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export interface ControlsPanelProps {
  onRunTraining: (numSteps: number, epsilon: number) => void;
  onReset: () => void;
  onCompare: () => void;
  isTraining: boolean;
}

export function ControlsPanel({ onRunTraining, onReset, onCompare, isTraining }: ControlsPanelProps) {
  const [numSteps, setNumSteps] = useState<number>(10);
  const [epsilon, setEpsilon] = useState<number>(0.2);

  const handleRunTraining = () => {
    onRunTraining(numSteps, epsilon);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">Training Controls</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <label className="text-sm font-medium mb-2 block">Number of Steps</label>
          <Select value={numSteps.toString()} onValueChange={(v) => setNumSteps(parseInt(v))}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="10">10 steps</SelectItem>
              <SelectItem value="20">20 steps</SelectItem>
              <SelectItem value="50">50 steps</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div>
          <label className="text-sm font-medium mb-2 block">
            Epsilon (ε-greedy): {epsilon.toFixed(2)}
          </label>
          <input
            type="range"
            min="0"
            max="0.4"
            step="0.05"
            value={epsilon}
            onChange={(e) => setEpsilon(parseFloat(e.target.value))}
            className="w-full"
            disabled={isTraining}
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>0.0</span>
            <span>0.4</span>
          </div>
        </div>

        <div className="space-y-2">
          <Button onClick={handleRunTraining} disabled={isTraining} className="w-full">
            {isTraining ? "Training..." : "Run Training"}
          </Button>
          <Button onClick={onReset} variant="outline" disabled={isTraining} className="w-full">
            Reset
          </Button>
          <Button onClick={onCompare} variant="secondary" disabled={isTraining} className="w-full">
            Compare Before vs After
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

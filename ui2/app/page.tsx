"use client"

import { useState } from "react";
import { ChatWindow } from "@/components/ChatWindow";
import { ControlsPanel } from "@/components/ControlsPanel";
import { PolicyStatsTable } from "@/components/PolicyStatsTable";
import { RewardChart } from "@/components/RewardChart";
import { CompareViewer } from "@/components/CompareViewer";
import { ChatMessage, PolicyStats, TrainingStep, CompareResult } from "@/lib/types";
import { POLICIES } from "@/lib/policies";

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "system",
      content: "Welcome to RL Post-Training Demo! This application demonstrates reinforcement learning post-training using offline dialogue traces (DSTC8-style). We have a small dataset of conversation turns compiled into a proxy RL environment. You can run training to search for the best system prompt (policy) for our student model (gemini-2.0-flash-lite) using a teacher model (gemini-3-pro-preview) for evaluation via a simple ε-greedy bandit algorithm."
    }
  ]);
  
  const [policyStats, setPolicyStats] = useState<PolicyStats[]>(
    POLICIES.map(policy => ({
      policyId: policy.id,
      policyName: policy.name,
      trials: 0,
      sumReward: 0,
      avgReward: 0
    }))
  );
  
  const [chartData, setChartData] = useState<{ stepIndex: number; bestAvgReward: number }[]>([]);
  const [compareResult, setCompareResult] = useState<CompareResult | null>(null);
  const [isTraining, setIsTraining] = useState(false);

  const handleRunTraining = async (numSteps: number, epsilon: number) => {
    setIsTraining(true);
    setMessages(prev => [...prev, {
      role: "system",
      content: `Starting training for ${numSteps} steps with ε=${epsilon}...`
    }]);

    try {
      const response = await fetch('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ numSteps, epsilon })
      });

      if (!response.ok) {
        throw new Error(`Training failed: ${response.statusText}`);
      }

      const data = await response.json();
      const { steps, finalStats } = data;

      // Add messages for each step with a slight delay for visual effect
      for (const step of steps) {
        await new Promise(resolve => setTimeout(resolve, 100));
        
        setMessages(prev => [
          ...prev,
          {
            role: "system",
            content: `Step ${step.stepIndex}/${numSteps} — sampled env \`${step.envId}\``
          },
          {
            role: "env",
            content: `**Context:** ${step.contextSnippet}\n\n**Ground Truth:** ${step.groundTruth}\n\n**Rubric:**\n${step.rubric}`
          },
          {
            role: "student",
            content: `**Policy: ${step.policyName}**\n\n${step.studentResponse}`
          },
          {
            role: "teacher",
            content: `Score: ${step.reward.toFixed(2)}`
          },
          {
            role: "system",
            content: `Updated avg reward for ${step.policyName}: ${step.policyStatsAfter.find(s => s.policyId === step.policyId)?.avgReward.toFixed(2) || 'N/A'}`
          }
        ]);

        setPolicyStats(step.policyStatsAfter);
        
        // Update chart data
        const bestAvg = Math.max(...step.policyStatsAfter.map(s => s.avgReward));
        setChartData(prev => [...prev, { stepIndex: step.stepIndex, bestAvgReward: bestAvg }]);
      }

      setMessages(prev => [...prev, {
        role: "system",
        content: `Training complete! Best policy: ${finalStats.reduce((best, curr) => 
          curr.avgReward > best.avgReward ? curr : best
        ).policyName} with avg reward ${Math.max(...finalStats.map(s => s.avgReward)).toFixed(2)}`
      }]);

    } catch (error) {
      setMessages(prev => [...prev, {
        role: "system",
        content: `Error during training: ${error instanceof Error ? error.message : 'Unknown error'}`
      }]);
    } finally {
      setIsTraining(false);
    }
  };

  const handleReset = () => {
    setMessages([
      {
        role: "system",
        content: "System reset. Ready to start a new training session."
      }
    ]);
    setPolicyStats(POLICIES.map(policy => ({
      policyId: policy.id,
      policyName: policy.name,
      trials: 0,
      sumReward: 0,
      avgReward: 0
    })));
    setChartData([]);
    setCompareResult(null);
  };

  const handleCompare = async () => {
    setMessages(prev => [...prev, {
      role: "system",
      content: "Running before vs after comparison on held-out example..."
    }]);

    try {
      const response = await fetch('/api/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ policyStats })
      });

      if (!response.ok) {
        throw new Error(`Comparison failed: ${response.statusText}`);
      }

      const result: CompareResult = await response.json();
      setCompareResult(result);

      setMessages(prev => [...prev, {
        role: "system",
        content: `Comparison complete! Baseline: ${result.baseline.score.toFixed(2)} vs Best: ${result.best.score.toFixed(2)} (Improvement: ${(result.best.score - result.baseline.score >= 0 ? '+' : '')}${(result.best.score - result.baseline.score).toFixed(2)})`
      }]);

    } catch (error) {
      setMessages(prev => [...prev, {
        role: "system",
        content: `Error during comparison: ${error instanceof Error ? error.message : 'Unknown error'}`
      }]);
    }
  };

  return (
    <div className="flex h-screen bg-background">
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <header className="border-b p-4">
          <h1 className="text-2xl font-bold">RL Post-Training Demo</h1>
          <p className="text-sm text-muted-foreground">
            Prompt-as-Policy Optimization via Gemini LLM Calls
          </p>
        </header>
        <ChatWindow messages={messages} />
      </div>

      {/* Side Panel */}
      <div className="w-[400px] border-l flex flex-col overflow-hidden">
        <div className="p-4 space-y-4 overflow-y-auto flex-1">
          <ControlsPanel 
            onRunTraining={handleRunTraining}
            onReset={handleReset}
            onCompare={handleCompare}
            isTraining={isTraining}
          />
          
          <PolicyStatsTable stats={policyStats} />
          
          <RewardChart steps={chartData} />
          
          {compareResult && <CompareViewer result={compareResult} />}
        </div>
      </div>
    </div>
  );
}

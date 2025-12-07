"use client"

import { useState } from "react";
import { ChatWindow } from "@/components/ChatWindow";
import { ControlsPanel } from "@/components/ControlsPanel";
import { PolicyStatsTable } from "@/components/PolicyStatsTable";
import { RewardChart } from "@/components/RewardChart";
import { CompareViewer } from "@/components/CompareViewer";
import { ChatMessage, PolicyStats, TrainingStep, CompareResult } from "@/lib/types";
import { POLICIES } from "@/lib/policies";
import { Sparkles, Brain, Target } from "lucide-react";

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "system",
      content: "👋 Welcome! This demo shows how to optimize AI assistant prompts using reinforcement learning. Run training to find the best policy using a simple ε-greedy bandit algorithm."
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
      content: `🚀 Starting ${numSteps}-step training with ε=${epsilon.toFixed(2)}...`
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
            content: `Step ${step.stepIndex}/${numSteps} — ${step.envId}`
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
            content: `📊 ${step.policyName} avg: ${step.policyStatsAfter.find(s => s.policyId === step.policyId)?.avgReward.toFixed(3) || 'N/A'}`
          }
        ]);

        setPolicyStats(step.policyStatsAfter);
        
        // Update chart data
        const bestAvg = Math.max(...step.policyStatsAfter.map(s => s.avgReward));
        setChartData(prev => [...prev, { stepIndex: step.stepIndex, bestAvgReward: bestAvg }]);
      }

      const bestPolicy = finalStats.reduce((best, curr) => 
        curr.avgReward > best.avgReward ? curr : best
      );

      setMessages(prev => [...prev, {
        role: "system",
        content: `✅ Training complete! Best: **${bestPolicy.policyName}** (${bestPolicy.avgReward.toFixed(3)})`
      }]);

    } catch (error) {
      setMessages(prev => [...prev, {
        role: "system",
        content: `❌ Error: ${error instanceof Error ? error.message : 'Unknown error'}`
      }]);
    } finally {
      setIsTraining(false);
    }
  };

  const handleReset = () => {
    setMessages([
      {
        role: "system",
        content: "🔄 System reset. Ready for a new training session."
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
      content: "🔍 Running before vs after comparison..."
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

      const improvement = result.best.score - result.baseline.score;
      const sign = improvement >= 0 ? '+' : '';
      
      setMessages(prev => [...prev, {
        role: "system",
        content: `📈 Baseline: ${result.baseline.score.toFixed(3)} → Best: ${result.best.score.toFixed(3)} (${sign}${improvement.toFixed(3)})`
      }]);

    } catch (error) {
      setMessages(prev => [...prev, {
        role: "system",
        content: `❌ Error: ${error instanceof Error ? error.message : 'Unknown error'}`
      }]);
    }
  };

  return (
    <div className="flex h-screen bg-gradient-to-br from-background to-muted/20">
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <header className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-primary/10">
              <Brain className="w-6 h-6 text-primary" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight">RL Post-Training</h1>
              <p className="text-sm text-muted-foreground">
                Optimize prompts with ε-greedy bandit learning
              </p>
            </div>
          </div>
        </header>
        
        <div className="flex-1 overflow-hidden p-4">
          <ChatWindow messages={messages} />
        </div>
      </div>

      {/* Side Panel */}
      <div className="w-[420px] border-l bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 flex flex-col overflow-hidden">
        <div className="p-6 space-y-6 overflow-y-auto flex-1">
          <ControlsPanel 
            onRunTraining={handleRunTraining}
            onReset={handleReset}
            onCompare={handleCompare}
            isTraining={isTraining}
          />
          
          <PolicyStatsTable stats={policyStats} />
          
          {chartData.length > 0 && <RewardChart steps={chartData} />}
          
          {compareResult && <CompareViewer result={compareResult} />}
        </div>
      </div>
    </div>
  );
}

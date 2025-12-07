import { NextRequest, NextResponse } from "next/server";
import { POLICIES } from "@/lib/policies";
import type { PolicyStats, TrainingStep } from "@/lib/types";

// Mock environment samples
const MOCK_ENV_SAMPLES = [
  { id: "93_00000-1", prompt: "User: I'm looking for some free attractions to check out.", ground_truth: "Anywhere in particular?", task_description: "Travel assistance" },
  { id: "93_00000-3", prompt: "User: Something of a place of interest in NYC.", ground_truth: "There's a cool Tourist Attraction called Balto Statue.", task_description: "Travel assistance" },
  { id: "94_00001-1", prompt: "User: I need to find a rental car.", ground_truth: "What city will you be picking it up in?", task_description: "Rental car service" },
  { id: "95_00002-1", prompt: "User: Can you help me find a movie to watch?", ground_truth: "Sure! What kind of movie are you interested in?", task_description: "Movie recommendations" },
  { id: "96_00003-1", prompt: "User: I want to book a table at a restaurant.", ground_truth: "What city are you looking in?", task_description: "Restaurant booking" },
  { id: "97_00004-1", prompt: "User: I need to check the weather forecast.", ground_truth: "What city would you like the forecast for?", task_description: "Weather service" },
  { id: "98_00005-1", prompt: "User: Can you help me find a hotel?", ground_truth: "What city are you looking for a hotel in?", task_description: "Hotel search" },
  { id: "99_00006-1", prompt: "User: I want to buy some event tickets.", ground_truth: "What type of event are you interested in?", task_description: "Event tickets" },
];

// Mock student responses per policy
const MOCK_RESPONSES: Record<string, string[]> = {
  policy_base: [
    "I'd be happy to help you with that. Could you tell me more about what you're looking for?",
    "Sure, I can assist you. What specific details do you need?",
    "Of course! Let me help you find what you need.",
  ],
  policy_concise: [
    "Which city?",
    "What type?",
    "When do you need it?",
  ],
  policy_empathetic: [
    "I understand you're looking for help - I'm here for you! Let me find the best options.",
    "That sounds exciting! I'd love to help you with that. What are your preferences?",
    "I can see this is important to you. Let me assist you right away.",
  ],
  policy_clarifying: [
    "Before I help, could you clarify: are you looking for something specific or open to suggestions?",
    "To give you the best answer, may I ask: what's your budget range?",
    "I want to make sure I understand - are you looking for this today or planning ahead?",
  ],
  policy_detailed: [
    "I'd be happy to help you with that. There are several options available depending on your preferences, budget, and timeline. Let me walk you through the main categories...",
    "Great question! Here's a comprehensive overview of what's available, including popular choices, pricing tiers, and recommendations based on common preferences...",
    "Absolutely! Let me provide you with detailed information including options, comparisons, and my top recommendations based on various criteria...",
  ],
};

// Mock rubrics
const MOCK_RUBRICS = [
  "1. Relevance: Response addresses the user's query\n2. Helpfulness: Provides actionable next steps\n3. Tone: Professional and friendly\n4. Clarity: Easy to understand",
  "1. Accuracy: Information is correct\n2. Completeness: Covers key aspects\n3. Conciseness: Not overly verbose\n4. Engagement: Encourages continued interaction",
  "1. Understanding: Shows comprehension of user needs\n2. Guidance: Offers clear direction\n3. Professionalism: Maintains appropriate tone\n4. Efficiency: Moves conversation forward",
];

function initializePolicyStats(): PolicyStats[] {
  return POLICIES.map((p) => ({
    policyId: p.id,
    policyName: p.name,
    trials: 0,
    sumReward: 0,
    avgReward: 0,
  }));
}

function selectPolicy(stats: PolicyStats[], epsilon: number): string {
  // ε-greedy selection
  if (Math.random() < epsilon) {
    // Explore: random policy
    return POLICIES[Math.floor(Math.random() * POLICIES.length)].id;
  } else {
    // Exploit: best policy (or random if no trials yet)
    const withTrials = stats.filter((s) => s.trials > 0);
    if (withTrials.length === 0) {
      return POLICIES[Math.floor(Math.random() * POLICIES.length)].id;
    }
    return withTrials.reduce((best, curr) =>
      curr.avgReward > best.avgReward ? curr : best
    ).policyId;
  }
}

function generateMockReward(policyId: string): number {
  // Different policies have different "true" reward distributions
  const baseMeans: Record<string, number> = {
    policy_base: 0.65,
    policy_concise: 0.55,
    policy_empathetic: 0.75,
    policy_clarifying: 0.70,
    policy_detailed: 0.60,
  };
  const mean = baseMeans[policyId] || 0.6;
  // Add noise: reward = mean + noise, clamped to [0, 1]
  const noise = (Math.random() - 0.5) * 0.3;
  return Math.max(0, Math.min(1, mean + noise));
}

export async function POST(req: NextRequest) {
  try {
    const { numSteps, epsilon, initialStats } = await req.json();

    let policyStats: PolicyStats[] = initialStats || initializePolicyStats();
    const steps: TrainingStep[] = [];

    for (let i = 0; i < numSteps; i++) {
      // Simulate async delay (50-150ms per step)
      await new Promise((resolve) => setTimeout(resolve, 50 + Math.random() * 100));

      // Sample random environment
      const envSample = MOCK_ENV_SAMPLES[Math.floor(Math.random() * MOCK_ENV_SAMPLES.length)];

      // Select policy using ε-greedy
      const selectedPolicyId = selectPolicy(policyStats, epsilon);
      const policy = POLICIES.find((p) => p.id === selectedPolicyId)!;

      // Get mock response
      const responses = MOCK_RESPONSES[selectedPolicyId] || MOCK_RESPONSES.policy_base;
      const studentResponse = responses[Math.floor(Math.random() * responses.length)];

      // Get mock rubric
      const rubric = MOCK_RUBRICS[Math.floor(Math.random() * MOCK_RUBRICS.length)];

      // Generate mock reward
      const reward = generateMockReward(selectedPolicyId);

      // Update stats
      policyStats = policyStats.map((s) => {
        if (s.policyId === selectedPolicyId) {
          const newTrials = s.trials + 1;
          const newSumReward = s.sumReward + reward;
          return {
            ...s,
            trials: newTrials,
            sumReward: newSumReward,
            avgReward: newSumReward / newTrials,
          };
        }
        return s;
      });

      const step: TrainingStep = {
        stepIndex: i + 1,
        envId: envSample.id,
        policyId: selectedPolicyId,
        policyName: policy.name,
        contextSnippet: envSample.prompt.slice(0, 100) + (envSample.prompt.length > 100 ? "..." : ""),
        groundTruth: envSample.ground_truth,
        rubric,
        studentResponse,
        reward,
        policyStatsAfter: policyStats,
      };

      steps.push(step);
    }

    return NextResponse.json({
      steps,
      finalStats: policyStats,
    });
  } catch (error) {
    console.error("Training error:", error);
    return NextResponse.json(
      { error: "Training failed", details: String(error) },
      { status: 500 }
    );
  }
}

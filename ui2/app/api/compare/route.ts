import { NextRequest, NextResponse } from "next/server";
import { POLICIES } from "@/lib/policies";
import type { PolicyStats, CompareResult } from "@/lib/types";

// Mock environment samples for comparison
const MOCK_EVAL_SAMPLES = [
  {
    id: "eval_001",
    dialogue_id: "eval_001",
    turn_index: 1,
    prompt: "User: I'm planning a trip and need help finding flights.",
    ground_truth: "I'd be happy to help you find flights. What's your departure city and destination?",
    rubric: "1. Helpfulness: Offers assistance\n2. Clarity: Asks for necessary details\n3. Tone: Friendly and professional\n4. Efficiency: Gets key information",
    task_description: "Flight booking assistance"
  },
  {
    id: "eval_002",
    dialogue_id: "eval_002",
    turn_index: 1,
    prompt: "User: Can you recommend a good restaurant for a date night?",
    ground_truth: "I'd love to help! What city are you in, and do you have any cuisine preferences?",
    rubric: "1. Engagement: Shows enthusiasm\n2. Relevance: Asks pertinent questions\n3. Personalization: Considers user context\n4. Helpfulness: Guides toward solution",
    task_description: "Restaurant recommendations"
  },
  {
    id: "eval_003",
    dialogue_id: "eval_003",
    turn_index: 1,
    prompt: "User: I need to schedule a doctor's appointment.",
    ground_truth: "I can help you with that. What type of doctor are you looking for, and what's your location?",
    rubric: "1. Responsiveness: Acknowledges request\n2. Clarity: Asks clear questions\n3. Professionalism: Appropriate tone\n4. Actionability: Moves toward booking",
    task_description: "Medical appointment scheduling"
  },
];

// Mock responses per policy type
const MOCK_POLICY_RESPONSES: Record<string, (prompt: string) => string> = {
  policy_base: (prompt) => "I'd be happy to help you with that. Could you provide more details about what you're looking for?",
  policy_concise: (prompt) => "Sure. What details do you need?",
  policy_empathetic: (prompt) => "I understand this is important to you! I'm here to help. Let me know your preferences and I'll find the best options for you.",
  policy_clarifying: (prompt) => "Before I assist, may I ask: do you have any specific requirements or preferences I should know about?",
  policy_detailed: (prompt) => "I'd be delighted to assist you with this. Let me walk you through the process step by step. First, I'll need some information about your specific needs, preferences, and any constraints you might have. This will help me provide you with the most relevant and personalized recommendations.",
};

function generateMockScore(policyId: string, isBaseline: boolean): number {
  // Baseline always gets a moderate score
  if (isBaseline) {
    return 0.65 + (Math.random() - 0.5) * 0.1; // 0.60-0.70
  }
  
  // Best policy scores based on their "true" performance
  const baseMeans: Record<string, number> = {
    policy_base: 0.65,
    policy_concise: 0.55,
    policy_empathetic: 0.78,
    policy_clarifying: 0.72,
    policy_detailed: 0.62,
  };
  
  const mean = baseMeans[policyId] || 0.65;
  const noise = (Math.random() - 0.5) * 0.08;
  return Math.max(0, Math.min(1, mean + noise));
}

export async function POST(req: NextRequest) {
  try {
    const { policyStats } = await req.json();

    // Simulate API delay
    await new Promise((resolve) => setTimeout(resolve, 800 + Math.random() * 400));

    // Pick a random eval sample
    const envSample = MOCK_EVAL_SAMPLES[Math.floor(Math.random() * MOCK_EVAL_SAMPLES.length)];

    // Find baseline and best policy
    const baselinePolicy = POLICIES.find((p) => p.id === "policy_base")!;
    
    const bestStats: PolicyStats = policyStats.reduce(
      (best: PolicyStats, current: PolicyStats) =>
        current.avgReward > best.avgReward ? current : best
    );
    const bestPolicy = POLICIES.find((p) => p.id === bestStats.policyId)!;

    // Generate mock responses
    const baselineResponseFn = MOCK_POLICY_RESPONSES[baselinePolicy.id] || MOCK_POLICY_RESPONSES.policy_base;
    const bestResponseFn = MOCK_POLICY_RESPONSES[bestPolicy.id] || MOCK_POLICY_RESPONSES.policy_base;

    const baselineResponse = baselineResponseFn(envSample.prompt);
    const bestResponse = bestResponseFn(envSample.prompt);

    // Generate mock scores
    const baselineScore = generateMockScore(baselinePolicy.id, true);
    const bestScore = generateMockScore(bestPolicy.id, false);

    const result: CompareResult = {
      env: envSample,
      baseline: {
        policyId: baselinePolicy.id,
        policyName: baselinePolicy.name,
        response: baselineResponse,
        score: baselineScore,
      },
      best: {
        policyId: bestPolicy.id,
        policyName: bestPolicy.name,
        response: bestResponse,
        score: bestScore,
      },
    };

    return NextResponse.json(result);
  } catch (error) {
    console.error("Comparison error:", error);
    return NextResponse.json(
      { error: "Comparison failed", details: String(error) },
      { status: 500 }
    );
  }
}

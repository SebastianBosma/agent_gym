import type { EnvSample, Policy, PolicyStats, TrainingStep } from "./types";
import { ensureRubric, scoreResponse } from "./teacher";
import { generateStudentResponse } from "./student";

/**
 * Run one training step using epsilon-greedy bandit.
 */
export async function runTrainingStep({
  stepIndex,
  envSamples,
  policies,
  policyStats,
  epsilon,
}: {
  stepIndex: number;
  envSamples: EnvSample[];
  policies: Policy[];
  policyStats: PolicyStats[];
  epsilon: number;
}): Promise<TrainingStep> {
  // Sample a random environment example
  const envSample = envSamples[Math.floor(Math.random() * envSamples.length)];

  // Epsilon-greedy policy selection
  let selectedPolicy: Policy;
  if (Math.random() < epsilon) {
    // Explore: random policy
    selectedPolicy = policies[Math.floor(Math.random() * policies.length)];
  } else {
    // Exploit: best policy so far
    const bestStats = policyStats.reduce((best, current) =>
      current.avgReward > best.avgReward ? current : best
    );
    selectedPolicy = policies.find((p) => p.id === bestStats.policyId)!;
  }

  // Ensure rubric exists
  const rubric = await ensureRubric(envSample);

  // Generate student response
  const studentResponse = await generateStudentResponse({
    policy: selectedPolicy,
    envSample,
  });

  // Score the response
  const reward = await scoreResponse({
    envSample,
    candidate: studentResponse,
    rubric,
  });

  // Update policy stats
  const updatedStats = policyStats.map((stats) => {
    if (stats.policyId === selectedPolicy.id) {
      const newTrials = stats.trials + 1;
      const newSumReward = stats.sumReward + reward;
      return {
        ...stats,
        trials: newTrials,
        sumReward: newSumReward,
        avgReward: newSumReward / newTrials,
      };
    }
    return stats;
  });

  // Create context snippet (first 100 chars)
  const contextSnippet =
    envSample.prompt.length > 100
      ? envSample.prompt.substring(0, 100) + "..."
      : envSample.prompt;

  return {
    stepIndex,
    envId: envSample.id,
    policyId: selectedPolicy.id,
    policyName: selectedPolicy.name,
    contextSnippet,
    groundTruth: envSample.ground_truth,
    rubric,
    studentResponse,
    reward,
    policyStatsAfter: updatedStats,
  };
}

/**
 * Initialize policy stats for all policies.
 */
export function initializePolicyStats(policies: Policy[]): PolicyStats[] {
  return policies.map((policy) => ({
    policyId: policy.id,
    trials: 0,
    sumReward: 0,
    avgReward: 0,
  }));
}

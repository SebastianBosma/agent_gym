/**
 * Type definitions for the RL post-training demo
 */

export type EnvSample = {
  id: string;
  dialogue_id: string;
  turn_index: number;
  prompt: string;
  ground_truth: string;
  rubric: string | null;
  task_description: string;
};

export type Policy = {
  id: string;
  name: string;
  systemPrompt: string;
};

export type PolicyStats = {
  policyId: string;
  trials: number;
  sumReward: number;
  avgReward: number;
};

export type TrainingStep = {
  stepIndex: number;
  envId: string;
  policyId: string;
  policyName: string;
  contextSnippet: string;
  groundTruth: string;
  rubric: string;
  studentResponse: string;
  reward: number;
  policyStatsAfter: PolicyStats[];
};

export type ChatMessage = {
  id: string;
  role: 'system' | 'env' | 'student' | 'teacher';
  content: string;
  metadata?: {
    policyName?: string;
    reward?: number;
    envId?: string;
  };
};

export type CompareResult = {
  env: EnvSample;
  baseline: {
    policyId: string;
    policyName: string;
    response: string;
    score: number;
  };
  best: {
    policyId: string;
    policyName: string;
    response: string;
    score: number;
  };
};

/**
 * Policy definitions - different system prompts for the student model
 */

import { Policy } from './types';

export const POLICIES: Policy[] = [
  {
    id: 'policy_base',
    name: 'Base Assistant',
    systemPrompt: 'You are a helpful assistant. Provide clear and accurate responses to user queries.',
  },
  {
    id: 'policy_concise',
    name: 'Concise Assistant',
    systemPrompt: 'You are a helpful assistant. Be concise and precise in your responses. Keep answers brief and to the point while remaining accurate.',
  },
  {
    id: 'policy_empathetic',
    name: 'Empathetic Assistant',
    systemPrompt: 'You are a helpful and empathetic assistant. Acknowledge the user\'s feelings and needs, then provide a thoughtful, supportive response.',
  },
  {
    id: 'policy_clarifying',
    name: 'Clarifying Assistant',
    systemPrompt: 'You are a helpful assistant. When the user\'s request is ambiguous or could benefit from clarification, ask one concise clarifying question before responding. Otherwise, provide a clear and helpful answer.',
  },
  {
    id: 'policy_detailed',
    name: 'Detailed Assistant',
    systemPrompt: 'You are a helpful assistant. Provide detailed, comprehensive responses that thoroughly address the user\'s query with relevant context and examples when appropriate.',
  },
];

export function getPolicyById(policyId: string): Policy | undefined {
  return POLICIES.find((p) => p.id === policyId);
}

export function getInitialPolicyStats(): Record<string, { trials: number; sumReward: number }> {
  return Object.fromEntries(
    POLICIES.map((p) => [p.id, { trials: 0, sumReward: 0 }])
  );
}

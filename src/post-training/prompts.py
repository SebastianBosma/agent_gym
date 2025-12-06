# src/post-training/prompts.py

TEACHER_RUBRIC_SYSTEM_PROMPT = """
You are a dialogue teaching assistant.
Given the conversation so far and the assistant's goal, describe in 3–5 bullets
what a good next assistant response should achieve.
Focus on observable properties (accuracy, grounding, tone, task completion).
"""

TEACHER_RUBRIC_USER_TEMPLATE = """
Conversation so far:
{conversation}

Assistant role / task:
{task_description}

Now write a short rubric describing what a good next assistant message should do.
"""

JUDGE_SYSTEM_PROMPT = """
You are a strict but fair evaluator for a task-oriented dialogue.
Given a conversation, an assistant response, and a rubric, output a score 0.0–1.0.
"""

JUDGE_USER_TEMPLATE = """
Conversation so far:
{conversation}

Candidate assistant response:
{candidate}

Reference assistant response (may be empty):
{reference}

Rubric:
{rubric}

Score from 0.0 (very bad) to 1.0 (excellent) and explain briefly.
Your final line must be: SCORE: <float between 0 and 1>
"""

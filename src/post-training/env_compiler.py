# src/post-training/env_compiler.py

from dataclasses import dataclass
from typing import Optional, List
from sgd_trace_parser import Turn
import prompts

@dataclass
class EnvSample:
    id: str
    dialogue_id: str
    turn_index: int
    state: str            # prompt: conversation so far
    ground_truth: str     # ground-truth system response
    rubric: Optional[str] # optional, can be None if we generate on the fly
    task_description: Optional[str] = None  # e.g., "help user book a restaurant"


def format_conversation_prefix(turns: List[Turn], upto_turn_index: int) -> str:
    """
    Build a text prompt like:
    User: ...
    Assistant: ...
    ...
    User: <last user turn>
    """
    lines = []
    for t in turns[: upto_turn_index + 1]:
        role = "User" if t.role == "user" else "Assistant"
        lines.append(f"{role}: {t.text}")
    return "\n".join(lines)


def infer_task_description(turns: List[Turn]) -> str:
    """
    For hackathon: simple heuristic. Later can ask Gemini to summarize.
    """
    # e.g., use first user utterance + first service name
    first_user = next((t for t in turns if t.role == "user"), None)
    service = next((t.service for t in turns if t.service is not None), None)
    base = first_user.text if first_user else "Help the user complete their task."
    if service:
        return f"The assistant helps the user with service '{service}'.\nUser goal: {base}"
    return f"User goal: {base}"


def compile_dialogue_to_env_samples(
    turns: List[Turn],
    generate_rubric_fn=None,
) -> List[EnvSample]:
    """
    For each system turn, create a single-step bandit sample:
      state = conversation up to the *previous* turn (usually a user turn)
      ground_truth = this system turn's text
      rubric = optional, via Gemini teacher
    """
    samples: List[EnvSample] = []
    task_desc = infer_task_description(turns)

    for i, t in enumerate(turns):
        if t.role != "system":
            continue

        # state = conversation up to *previous* turn (index i-1)
        if i == 0:
            continue  # skip edge case
        state = format_conversation_prefix(turns, upto_turn_index=i - 1)

        rubric: Optional[str] = None
        if generate_rubric_fn is not None:
            rubric_prompt = prompts.TEACHER_RUBRIC_USER_TEMPLATE.format(
                conversation=state,
                task_description=task_desc,
            )
            rubric = generate_rubric_fn(
                system_prompt=prompts.TEACHER_RUBRIC_SYSTEM_PROMPT,
                user_prompt=rubric_prompt,
            )

        sample_id = f"{t.dialogue_id}-{t.turn_index}"

        samples.append(
            EnvSample(
                id=sample_id,
                dialogue_id=t.dialogue_id,
                turn_index=t.turn_index,
                state=state,
                ground_truth=t.text,
                rubric=rubric,
                task_description=task_desc,
            )
        )

    return samples

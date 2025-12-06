from __future__ import annotations

from typing import Any, List


def _to_text(x: Any) -> str:
    """
    TRL can sometimes pass conversational-format structures.
    We aggressively normalize to plain text.
    """
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    # Conversational prompt: list[{"role":..., "content":...}, ...]
    if isinstance(x, list):
        parts = []
        for item in x:
            if isinstance(item, dict) and "content" in item:
                parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()
    if isinstance(x, dict) and "content" in x:
        return str(x["content"])
    return str(x)


def make_gemini_reward_func(judge):
    """
    Returns a TRL-compatible reward function.

    TRL will call this with batched columns from the dataset + completions.
    Common signatures:
      reward(completions, **kwargs)
      reward(prompts, completions, ground_truth, rubric, **kwargs)
    """
    def reward_func(prompts=None, completions=None, ground_truth=None, rubric=None, **kwargs) -> List[float]:
        # Normalize inputs to lists
        prompts_l = prompts if isinstance(prompts, list) else [prompts] * len(completions)
        gt_l = ground_truth if isinstance(ground_truth, list) else [ground_truth] * len(completions)

        if rubric is None:
            rubric_l = ["- Be helpful\n- Be accurate\n- Ask for missing required info\n- Keep it concise"] * len(completions)
        else:
            rubric_l = rubric if isinstance(rubric, list) else [rubric] * len(completions)

        comps_l = completions if isinstance(completions, list) else [completions]

        prompts_txt = [_to_text(p) for p in prompts_l]
        comps_txt = [_to_text(c) for c in comps_l]
        gt_txt = [_to_text(g) for g in gt_l]
        rubric_txt = [_to_text(r) for r in rubric_l]

        return judge.score_batch(prompts_txt, comps_txt, rubric_txt, gt_txt)

    return reward_func

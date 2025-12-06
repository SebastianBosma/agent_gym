from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from google import genai


_SCORE_RE = re.compile(r"(-?\d+(?:\.\d+)?)")


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _extract_score(text: str) -> Optional[float]:
    """
    Robust parsing:
    - Prefer JSON {"score": 0.7}
    - Else pick first float-looking token.
    """
    if not text:
        return None
    t = text.strip()

    # Strip code fences if the model insists.
    if t.startswith("```"):
        t = t.strip("`")
        # try to remove leading language tag
        t = re.sub(r"^\s*(json|JSON)\s*", "", t)

    # Try JSON
    try:
        obj = json.loads(t)
        if isinstance(obj, dict) and "score" in obj:
            return float(obj["score"])
    except Exception:
        pass

    m = _SCORE_RE.search(t)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


class RewardCache:
    """
    Tiny SQLite cache so your Gemini bill doesn't explode during GRPO.
    Keyed by sha1(prompt + completion + rubric + ground_truth + model).
    """
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        with self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS rewards (k TEXT PRIMARY KEY, v REAL)"
            )

    def get(self, k: str) -> Optional[float]:
        with self._lock:
            cur = self._conn.execute("SELECT v FROM rewards WHERE k=?", (k,))
            row = cur.fetchone()
            return None if row is None else float(row[0])

    def set(self, k: str, v: float) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "INSERT OR REPLACE INTO rewards (k, v) VALUES (?, ?)",
                    (k, float(v)),
                )


@dataclass
class GeminiJudge:
    model: str
    cache: RewardCache
    max_retries: int = 4
    base_sleep_s: float = 0.8

    def __post_init__(self) -> None:
        # The client picks up GEMINI_API_KEY from env if set.
        # (You can also pass api_key=... if you prefer.)
        self.client = genai.Client()

    def score_one(
        self,
        prompt: str,
        completion: str,
        rubric: str,
        ground_truth: str,
    ) -> float:
        # Cache key includes judge model.
        key_material = f"{self.model}\nPROMPT:{prompt}\nRUBRIC:{rubric}\nGT:{ground_truth}\nC:{completion}"
        k = _sha1(key_material)
        cached = self.cache.get(k)
        if cached is not None:
            return cached

        contents = (
            "You are a strict evaluator for a customer support assistant.\n\n"
            "Return ONLY valid JSON like: {\"score\": 0.73}\n"
            "Score range: 0.0 (bad) to 1.0 (excellent).\n"
            "Use this rubric as the main criteria; ground-truth is provided as a reference.\n\n"
            f"=== RUBRIC ===\n{rubric}\n\n"
            f"=== CONVERSATION PROMPT ===\n{prompt}\n\n"
            f"=== CANDIDATE ASSISTANT RESPONSE ===\n{completion}\n\n"
            f"=== REFERENCE (GROUND TRUTH) RESPONSE ===\n{ground_truth}\n\n"
            "Now output JSON."
        )

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                )
                text = (resp.text or "").strip()
                score = _extract_score(text)
                if score is None:
                    # If parsing failed, treat as low score.
                    score = 0.0
                # Clamp.
                score = max(0.0, min(1.0, float(score)))
                self.cache.set(k, score)
                return score
            except Exception as e:
                last_err = e
                sleep_s = self.base_sleep_s * (2 ** attempt)
                time.sleep(sleep_s)

        # Hard fail: cache 0 so you don't repeatedly error.
        self.cache.set(k, 0.0)
        return 0.0

    def score_batch(
        self,
        prompts: Iterable[str],
        completions: Iterable[str],
        rubrics: Iterable[str],
        ground_truths: Iterable[str],
    ) -> List[float]:
        out: List[float] = []
        for p, c, r, gt in zip(prompts, completions, rubrics, ground_truths):
            out.append(self.score_one(p, c, r, gt))
        return out

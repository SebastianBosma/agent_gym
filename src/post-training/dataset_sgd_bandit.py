# src/post-training/dataset_sgd_bandit.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from jsonl_utils import read_jsonl

@dataclass
class BanditExample:
    id: str
    prompt: str
    ground_truth: str
    rubric: Optional[str]

class SGDBanditDataset:
    def __init__(self, path: str):
        rows = read_jsonl(path)
        self.examples: List[BanditExample] = [
            BanditExample(
                id=r["id"],
                prompt=r["prompt"],
                ground_truth=r.get("ground_truth", ""),
                rubric=r.get("rubric"),
            )
            for r in rows
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> BanditExample:
        return self.examples[idx]

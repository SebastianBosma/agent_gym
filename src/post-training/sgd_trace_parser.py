# src/post-training/sgd_trace_parser.py

from dataclasses import dataclass
from typing import Literal, List, Dict, Any, Optional
import json
from pathlib import Path

Role = Literal["user", "system"]

@dataclass
class Turn:
    dialogue_id: str
    turn_index: int
    role: Role
    text: str
    service: Optional[str] = None
    frames: Optional[List[Dict[str, Any]]] = None

def load_sgd_dialogue_file(path: Path) -> List[List[Turn]]:
    """
    Load a single DSTC8 SGD JSON file (list of dialogues) and convert
    to a list[dialogue], where each dialogue is list[Turn].
    """
    data = json.loads(path.read_text())
    dialogues: List[List[Turn]] = []

    for d in data:
        try:
            dialogue_id = d["dialogue_id"]
        except:
            print(d)
            dialogue_id = d["dialogue_id"]
        turns: List[Turn] = []

        for i, t in enumerate(d["turns"]):
            role = "user" if t["speaker"] == "USER" else "system"
            text = t["utterance"]
            # frames describe API / slot info; we keep them in case we want tasks later
            frames = t.get("frames", [])
            service = frames[0]["service"] if frames else None

            turns.append(
                Turn(
                    dialogue_id=dialogue_id,
                    turn_index=i,
                    role=role,
                    text=text,
                    service=service,
                    frames=frames,
                )
            )
        dialogues.append(turns)

    return dialogues


def load_sgd_corpus(root: Path) -> List[List[Turn]]:
    """
    root/
      train/
      dev/
      test/
    each containing *.json
    """
    all_dialogues: List[List[Turn]] = []
    for split in ["train", "dev"]:
        split_dir = root / split
        for f in split_dir.glob("*.json"):
            if "schema.json" not in str(f):
                all_dialogues.extend(load_sgd_dialogue_file(f))
    return all_dialogues

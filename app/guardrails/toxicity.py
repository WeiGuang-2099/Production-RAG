"""Conservative wordlist toxicity flagging (illustrative; a production system
would call a moderation API instead)."""
from __future__ import annotations

import re

TOXIC_TERMS = frozenset({"idiot", "moron", "scumbag"})

_PATTERN = re.compile(r"\b(" + "|".join(sorted(TOXIC_TERMS)) + r")\b", re.I)


def detect_toxicity(text: str) -> list[str]:
    if not text:
        return []
    return sorted({m.lower() for m in _PATTERN.findall(text)})

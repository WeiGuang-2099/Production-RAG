"""Heuristic prompt-injection detection (label, pattern)."""
from __future__ import annotations

import re

_RULES = [
    ("ignore_previous", re.compile(r"ignore (all |the )?(previous|prior|above) (instruction|prompt)s?", re.I)),
    ("disregard_above", re.compile(r"disregard (the |all )?(previous|prior|above)", re.I)),
    ("system_prompt", re.compile(r"system prompt", re.I)),
    ("you_are_now", re.compile(r"you are now", re.I)),
    ("reveal_prompt", re.compile(r"reveal (your )?(system )?(prompt|instructions)", re.I)),
    ("override_rules", re.compile(r"override (your |the )?(instruction|rule|setting)s?", re.I)),
    ("pretend", re.compile(r"pretend (to be|you are)", re.I)),
    ("jailbreak", re.compile(r"jailbreak", re.I)),
    ("dan", re.compile(r"\bDAN\b")),
    ("new_instructions", re.compile(r"new instructions\s*:", re.I)),
]


def detect_injection(text: str) -> list[str]:
    if not text:
        return []
    return [label for label, pattern in _RULES if pattern.search(text)]

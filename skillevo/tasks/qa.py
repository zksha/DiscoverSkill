"""
Simple open-domain QA demo task.

Replace this module with ALFWorld / WebShop / tool-using agent tasks
when moving beyond the demo phase.
"""

from __future__ import annotations
import random
import re
from typing import List

from .base import BaseTask, TaskInstance


_QA_PAIRS = [
    {"q": "What is the capital of France?", "a": "paris"},
    {"q": "Which planet is known as the Red Planet?", "a": "mars"},
    {"q": "What is the square root of 144?", "a": "12"},
    {"q": "Who wrote 'Pride and Prejudice'?", "a": "jane austen"},
    {"q": "What is the chemical symbol for gold?", "a": "au"},
    {"q": "How many sides does a hexagon have?", "a": "6"},
    {"q": "What is the largest ocean on Earth?", "a": "pacific"},
    {"q": "In which year did World War II end?", "a": "1945"},
    {"q": "What is the speed of light in m/s (approximate)?", "a": "3e8"},
    {"q": "What language has the most native speakers?", "a": "mandarin"},
    {"q": "What is the powerhouse of the cell?", "a": "mitochondria"},
    {"q": "How many bones are in the adult human body?", "a": "206"},
    {"q": "What element has atomic number 1?", "a": "hydrogen"},
    {"q": "Who painted the Mona Lisa?", "a": "da vinci"},
    {"q": "What is the smallest prime number?", "a": "2"},
    {"q": "What is the boiling point of water in Celsius?", "a": "100"},
    {"q": "Which country has the largest land area?", "a": "russia"},
    {"q": "What is the currency of Japan?", "a": "yen"},
    {"q": "How many chromosomes do humans have?", "a": "46"},
    {"q": "What is the sum of angles in a triangle in degrees?", "a": "180"},
]


class SimpleQATask(BaseTask):
    def __init__(self, pairs: List[dict] | None = None, seed: int | None = None):
        self._pairs = pairs or _QA_PAIRS
        self._rng = random.Random(seed)
        self._counter = 0

    def sample(self) -> TaskInstance:
        pair = self._rng.choice(self._pairs)
        self._counter += 1
        return TaskInstance(
            id=f"qa_{self._counter:04d}",
            prompt=pair["q"],
            metadata={"answer": pair["a"]},
        )

    def evaluate(self, instance: TaskInstance, response: str) -> float:
        expected = instance.metadata["answer"].lower().strip()
        resp_lower = response.lower()
        # liberal matching: expected answer appears anywhere in response
        # also handle numeric aliases (3e8 ≈ 3×10^8)
        if expected in resp_lower:
            return 1.0
        if re.search(re.escape(expected), resp_lower):
            return 1.0
        return 0.0

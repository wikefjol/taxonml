# truncation.py
import random
from typing import List, Optional

class TruncateFrontStrategy:
    """Keep the first N tokens."""
    def __init__(self, optimal_length: int):
        if not isinstance(optimal_length, int) or optimal_length <= 0:
            raise ValueError("optimal_length must be int > 0")
        self.optimal_length = optimal_length
    def __repr__(self):
        return f"{self.__class__.__name__}(optimal_length={self.optimal_length})"
    def execute(self, seq: List[List[str]]) -> List[List[str]]:
        return seq[:self.optimal_length]

class TruncateEndStrategy:
    """Keep the last N tokens."""
    def __init__(self, optimal_length: int):
        if not isinstance(optimal_length, int) or optimal_length <= 0:
            raise ValueError("optimal_length must be int > 0")
        self.optimal_length = optimal_length
    def __repr__(self):
        return f"{self.__class__.__name__}(optimal_length={self.optimal_length})"
    def execute(self, seq: List[List[str]]) -> List[List[str]]:
        return seq[-self.optimal_length:]

class SlidingWindowTruncationStrategy:
    """Keep a random window of length N (or whole seq if shorter)."""
    def __init__(self, optimal_length: int, rng: Optional[random.Random] = None):
        if not isinstance(optimal_length, int) or optimal_length <= 0:
            raise ValueError("optimal_length must be int > 0")
        self.optimal_length = optimal_length
        self.rng = rng or random.Random()
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"optimal_length={self.optimal_length})"
        )
    def execute(self, seq: List[List[str]]) -> List[List[str]]:
        if len(seq) <= self.optimal_length:
            return seq[:]
        max_start = len(seq) - self.optimal_length
        start = self.rng.randint(0, max_start)
        return seq[start:start + self.optimal_length]

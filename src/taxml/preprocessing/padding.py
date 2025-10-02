# padding.py
from typing import List

class PaddingEndStrategy:
    """
    Pad with ['PAD'] sublists at the end until reaching optimal_length.
    Does NOT trim; truncation module handles trimming.
    """
    def __init__(self, optimal_length: int):
        if not isinstance(optimal_length, int) or optimal_length <= 0:
            raise ValueError("optimal_length must be int > 0")
        self.optimal_length = optimal_length
    
    def __repr__(self):
        return f"{self.__class__.__name__}(optimal_length={self.optimal_length})"
    
    def execute(self, seq: List[List[str]]) -> List[List[str]]:
        padded = seq[:]
        while len(padded) < self.optimal_length:
            padded.append(['PAD'])
        return padded

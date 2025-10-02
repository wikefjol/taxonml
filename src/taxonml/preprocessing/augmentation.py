# augmentation.py
import random
from typing import List, Sequence, Optional

class SequenceModifier:
    """Modifies a sequence at a specific position."""
    def __init__(self, alphabet: Sequence[str]):
        if not alphabet:
            raise ValueError("alphabet must be non-empty")
        self.alphabet = list(alphabet)

    def _insert(self, seq: List[str], idx: int, rng: random.Random) -> None:
        insert_idx = rng.choice([idx, idx + 1])
        if insert_idx <= len(seq):
            seq.insert(insert_idx, rng.choice(self.alphabet))

    def _replace(self, seq: List[str], idx: int, rng: random.Random) -> None:
        current = seq[idx]
        choices = [a for a in self.alphabet if a != current] or self.alphabet
        seq[idx] = rng.choice(choices)

    def _delete(self, seq: List[str], idx: int, rng: random.Random) -> None:
        if len(seq) > 1:
            seq.pop(idx)

    def _swap(self, seq: List[str], idx: int, rng: random.Random) -> None:
        swap_pos = idx + rng.choice([-1, 1])
        if 0 <= swap_pos < len(seq):
            seq[idx], seq[swap_pos] = seq[swap_pos], seq[idx]

class BaseStrategy:
    """Standard augmentation strategy."""
    def __init__(
        self,
        modifier: SequenceModifier,
        alphabet: Sequence[str],
        modification_probability: float = 0.05,
        rng: Optional[random.Random] = None,
    ):
        if not 0.0 <= modification_probability <= 1.0:
            raise ValueError("modification_probability must be in [0,1]")
        if not alphabet:
            raise ValueError("alphabet must be non-empty")

        self.alphabet = list(alphabet)
        self.modifier = modifier
        self.modification_probability = modification_probability
        self.rng = rng or random.Random()

        self.operation_map = {
            'insert': lambda s, i: self.modifier._insert(s, i, self.rng),
            'replace': lambda s, i: self.modifier._replace(s, i, self.rng),
            'delete': lambda s, i: self.modifier._delete(s, i, self.rng),
            'swap':   lambda s, i: self.modifier._swap(s, i, self.rng),
        }
        self.operations = list(self.operation_map.keys())
        self.weights = [0.25, 0.25, 0.25, 0.25]
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"modification_probability={self.modification_probability}, "
            f"alphabet_size={len(self.alphabet)})"
        )
    
    def execute(self, seq: List[str]) -> List[str]:
        augmented_seq = seq[:]
        i = 0
        while i < len(augmented_seq):
            if self.rng.random() < self.modification_probability:
                operation = self.rng.choices(self.operations, weights=self.weights, k=1)[0]
                self.operation_map[operation](augmented_seq, i)
            i += 1
        return augmented_seq

class RandomStrategy(BaseStrategy):
    """Modify at every position."""
    def __init__(self, alphabet: Sequence[str], modifier: SequenceModifier, rng: Optional[random.Random] = None):
        super().__init__(modifier, alphabet, modification_probability=1.0, rng=rng)        
    def __repr__(self):
        return f"{self.__class__.__name__}(alphabet_size={len(self.alphabet)})"

class IdentityStrategy(BaseStrategy):
    """Do-nothing augmentation."""
    def __init__(self):
        # Provide a no-op modifier/alphabet but never used
        self._rng = random.Random()
    def __repr__(self):
        return f"{self.__class__.__name__}()"
    def execute(self, seq: List[str]) -> List[str]:
        return seq[:]

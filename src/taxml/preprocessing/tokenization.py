# tokenization.py
import random
from typing import List, Sequence, Optional

class KmerStrategy:
    def __init__(self, k: int, alphabet: Sequence[str] = ('A','C','G','T'), rng: Optional[random.Random] = None):
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"[PROCESSING.TOKENIZATION] 'KmerStrategy.__init__': invalid k={k}; must be int > 0")
        if not alphabet:
            raise ValueError("alphabet must be non-empty")
        self.k = k
        self.alphabet = list(alphabet)
        self.rng = rng or random.Random()
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"k={self.k}, "
            f"alphabet_size={len(self.alphabet)})"
        )
    
    def _make_divisible_by_k(self, input_seq: List[str]) -> List[str]:
        """Pads seq if its length is not divisible by k, using random characters from the padding alphabet."""
        seq = input_seq[:]
        while len(seq) % self.k != 0:
            seq.append(self.rng.choice(self.alphabet))
        return seq

    def execute(self, input_seq: List[str]) -> List[List[str]]:
        """Tokenizes the seq into k-mers, adding padding if necessary."""
        seq = input_seq[:]
        remainder = len(seq) % self.k
        if remainder != 0:
            seq = self._make_divisible_by_k(seq)
        kmer_seq = [''.join(seq[i:i + self.k]) for i in range(0, len(seq), self.k)]
        return [[kmer] for kmer in kmer_seq]

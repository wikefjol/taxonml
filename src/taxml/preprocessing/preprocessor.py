# preprocessor.py
from typing import Protocol, List, Optional
from taxml.preprocessing.vocab import Vocabulary

# Precise protocols for each stage
class AugmentStrategy(Protocol):
    def execute(self, sequence: List[str]) -> List[str]: ...

class TokenizeStrategy(Protocol):
    def execute(self, sequence: List[str]) -> List[List[str]]: ...

class PadStrategy(Protocol):
    def execute(self, tokens: List[List[str]]) -> List[List[str]]: ...

class TruncateStrategy(Protocol):
    def execute(self, tokens: List[List[str]]) -> List[List[str]]: ...

class Preprocessor:
    def __init__(
        self,
        augmentation_strategy: AugmentStrategy,
        tokenization_strategy: TokenizeStrategy,
        padding_strategy: PadStrategy,
        truncation_strategy: TruncateStrategy,
        vocab: Optional[Vocabulary] = None
    ):
        self.augmentation_strategy = augmentation_strategy
        self.tokenization_strategy = tokenization_strategy
        self.padding_strategy = padding_strategy
        self.truncation_strategy = truncation_strategy
        self.vocab = vocab
    def __repr__(self):
        return (
            f"Preprocessor(\n"
            f"  augmentation={repr(self.augmentation_strategy)},\n"
            f"  tokenization={repr(self.tokenization_strategy)},\n"
            f"  padding={repr(self.padding_strategy)},\n"
            f"  truncation={repr(self.truncation_strategy)},\n"
            f"  vocab=Vocabulary(size={len(self.vocab) if self.vocab else 'None'})\n"
            f")"
        )

    def process(self, sequence: str) -> List[int]:
        if self.vocab is None:
            raise ValueError("Preprocessor.vocab is required before mapping.")

        chars: List[str] = list(sequence)
        augmented: List[str] = self.augmentation_strategy.execute(chars)
        tokenized: List[List[str]] = self.tokenization_strategy.execute(augmented)
        padded: List[List[str]] = self.padding_strategy.execute(tokenized)
        truncated: List[List[str]] = self.truncation_strategy.execute(padded)
        mapped: List[int] = self.vocab.map_sentence(truncated)  # flat list of IDs
        return mapped

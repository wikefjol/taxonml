# taxonml/preprocessing/build.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any
from . import vocab as vocab_mod
from . import augmentation, tokenization, padding, truncation
from .preprocessor import Preprocessor

@dataclass(frozen=True)
class PreprocSpec:
    k: int
    core_tokens: int
    alphabet: tuple[str, ...]
    aug_p: float

def derive_shapes(cfg: Dict[str, Any]) -> Tuple[int, int]:
    k = int(cfg["tokenizer"]["k"])
    max_bases = int(cfg["tokenizer"]["max_length"])
    assert max_bases % k == 0, "max_length must be divisible by k"
    core_tokens = max_bases // k
    return k, core_tokens

def build_vocab(cfg: Dict[str, Any]) -> vocab_mod.Vocabulary:
    k = int(cfg["tokenizer"]["k"])
    alphabet = tuple(cfg["tokenizer"]["alphabet"])
    vocab = vocab_mod.Vocabulary()
    vocab_mod.KmerVocabConstructor(k=k, alphabet=list(alphabet)).build_vocab([], v)
    return v

def make_preprocessors(cfg: Dict[str, Any], vocab: vocab_mod.Vocabulary) -> Tuple[Preprocessor, Preprocessor, PreprocSpec]:
    k, core = derive_shapes(cfg)
    alphabet = tuple(cfg["tokenizer"]["alphabet"])
    aug_p = float(cfg["tokenizer"].get("augmentation_probability", 0.01))

    # strategies
    tok = tokenization.KmerStrategy(k=k, padding_alphabet=list(alphabet))
    pad = padding.PaddingEndStrategy(optimal_length=core)
    trunc = truncation.SlidingWindowTruncationStrategy(optimal_length=core)

    mod = augmentation.SequenceModifier(alphabet=list(alphabet))
    aug_train = augmentation.BaseStrategy(modifier=mod, alphabet=list(alphabet), modification_probability=aug_p)
    aug_eval  = augmentation.IdentityStrategy()

    preproc_train = Preprocessor(aug_train, tok, pad, trunc, vocab)
    preproc_val   = Preprocessor(aug_eval,  tok, pad, trunc, vocab)

    spec = PreprocSpec(k=k, core_tokens=core, alphabet=alphabet, aug_p=aug_p)
    return preproc_train, preproc_val, spec

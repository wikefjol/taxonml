# vocab.py
import json
from typing import Dict, List
import collections
from tqdm import tqdm
from abc import ABC, abstractmethod
from itertools import product
import random
def _truncate_mapping(mapping: dict, head: int = 10, tail: int = 5) -> dict:
    """Truncate a dict in the middle if too many items."""
    n = len(mapping)
    if n <= head + tail:
        return mapping
    keys = list(mapping.keys())
    head_keys = keys[:head]
    tail_keys = keys[-tail:]
    new_map = {k: mapping[k] for k in head_keys}
    new_map["..."] = f"... {n - (head + tail)} more ..."
    for k in tail_keys:
        new_map[k] = mapping[k]
    return new_map

class VocabConstructor(ABC):
    """
    Abstract base class for vocabulary constructors.
    """
    @abstractmethod
    def build_vocab(self, data: List[str], vocab: 'Vocabulary') -> None:
        """
        Build the vocabulary from the provided data.
        
        Args:
            data (List[str]): List of raw sequences.
            vocab (Vocabulary): Vocabulary instance to populate.
        """
        pass

class Vocabulary:
    """
    Vocabulary class to handle token-ID mappings.
    """
    def __init__(self):
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.alphabet = None

        # Fixed special tokens
        self.special_tokens = {
            'PAD': 0,    # Padding token
            'UNK': 1,    # Unknown token
            'CLS': 2,  # Classification token
            'SEP': 3,  # Separator token
            'MASK': 4  # Mask token
        }

        # Initialize special tokens
        for token, idx in self.special_tokens.items():
            self.add_token(token, idx)
    
    def __len__(self):
        return len(self.token_to_id)

    def __str__(self):
        truncated = _truncate_mapping(self.token_to_id, head=10, tail=5)
        return (
            "Vocabulary(\n"
            f"  size={len(self)},\n"
            f"  alphabet={self.alphabet},\n"
            f"  special_tokens={self.special_tokens},\n"
            f"  token_to_id=\n{json.dumps(truncated, indent=4)}\n"
            ")"
        )
    
    def dump(self, path: str | None = None):
        """
        Dump the entire vocabulary in JSON format (optionally to a file).
        """
        payload = {
            "size": len(self),
            "alphabet": self.alphabet,
            "special_tokens": self.special_tokens,
            "token_to_id": self.token_to_id,
        }
        if path:
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
        return json.dumps(payload, indent=2)

    
    def set_alphabet(self, alphabet):
        self.alphabet = alphabet

    def get_alphabet(self):
        return(self.alphabet)
    
    def get_special_tokens(self) -> list[str]:
        return list(self.special_tokens.values())
    
    def pad_id(self): 
        return self.get_id("PAD")
    def cls_id(self): 
        return self.get_id("CLS")
    def sep_id(self): 
        return self.get_id("SEP")
    def mask_id(self): 
        return self.get_id("MASK")
    def unk_id(self): 
        return self.get_id("UNK")
    
    def add_token(self, token: str, idx: int = None):
        """
        Add a token to the vocabulary. Optionally, specify its ID.
        """
        if token not in self.token_to_id:
            if idx is None:
                idx = len(self.token_to_id) # Assign the next available ID
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    def get_id(self, token: str) -> int:
        return self.token_to_id.get(token, self.special_tokens['UNK'])  # Default to UNK ID
    
    def get_token(self, idx: int) -> str:
        return self.id_to_token.get(idx, 'UNK')  # Default to UNK token
    
    def get_non_special_tokens(self) -> List[str]:
        """
        Returns a list of tokens that are not special tokens.

        Returns:
            List[str]: A list of non-special tokens.
        """
        special_tokens_set = set(self.special_tokens.keys())
        return [token for token in self.token_to_id if token not in special_tokens_set]

    def special_token_map(self): 
        return dict(self.special_tokens)

    def get_random_non_special_token(self) -> str:
        """
        Returns a random token that is not a special token.

        Returns:
            str: A random non-special token.
        """
        non_special_tokens = self.get_non_special_tokens()

        if not non_special_tokens:
            raise ValueError("No non-special tokens available in the vocabulary.")

        return random.choice(non_special_tokens)

    def map_sentence(self, processed_sentence: List[List[str]]) -> List[int]:
        """
        Maps a processed sentence from tokens to their corresponding IDs using the vocabulary.

        Args:
            processed_sentence (List[List[str]]): The preprocessed sentence as a list of token lists.

        Returns:
            List[int]: The sentence with tokens replaced by their corresponding IDs as a flat list.
        """
        return [self.get_id(token) for token_list in processed_sentence for token in token_list]
    
    def decode_sentence(
        self,
        ids,
        *,
        skip_specials: bool = False,
        trim_pad: bool = False,
        stop_at_sep: bool = False,
    ) -> list[str]:
        """
        Decode a sequence of token IDs back to token strings.

        Args:
            ids: Iterable/sequence of ints (or a 1D torch.Tensor / numpy array) with token IDs.
            skip_specials: If True, drop all special tokens (PAD/UNK/CLS/SEP/MASK) from the output.
            trim_pad: If True, strip trailing PAD tokens from the right before decoding.
            stop_at_sep: If True, stop decoding when the first SEP token is encountered (excluded).

        Returns:
            List[str]: Decoded token strings (e.g., k-mers and/or special-token names).
        """
        # Normalize to a Python list of ints
        try:
            ids = list(map(int, ids))
        except Exception:
            ids = [int(x) for x in ids]  # fallback

        # Optionally trim trailing PADs
        if trim_pad:
            pad = self.pad_id()
            end = len(ids)
            while end > 0 and ids[end - 1] == pad:
                end -= 1
            ids = ids[:end]

        special_id_set = set(self.get_special_tokens())
        sep_id = self.sep_id()

        out_tokens: list[str] = []
        for tid in ids:
            if stop_at_sep and tid == sep_id:
                break
            if skip_specials and tid in special_id_set:
                continue
            out_tokens.append(self.get_token(tid))
        return out_tokens

    def save(self, filepath: str):
        """
        Save the vocabulary to a JSON file.
        
        Args:
            filepath (str): Path to the JSON file.
        """
        with open(filepath, 'w') as f:
            json.dump(self.token_to_id, f, indent=4)
    
    def load(self, filepath: str):
        """
        Load the vocabulary from a JSON file.
        
        Args:
            filepath (str): Path to the JSON file.
        """
        with open(filepath, 'r') as f:
            self.token_to_id = json.load(f)
        self.id_to_token = {int(idx): token for token, idx in self.token_to_id.items()}
    
    def build_from_constructor(
        self, constructor: 'VocabConstructor', data: List[str] | None = None
    ) -> None:
        """
        Build the vocabulary using a specified constructor.
        
        Args:
            constructor (VocabConstructor): An instance of a vocabulary constructor.
            data (List[str] | None): Raw sequences if required by the constructor.
                                     May be omitted if the constructor ignores data.
        """
        # Defensive: constructors must handle None if they don’t require data
        constructor.build_vocab(data if data is not None else [], self)

class KmerVocabConstructor(VocabConstructor):
    """
    Vocabulary constructor for k-mer tokenization.
    Generates all possible k-mers based on the provided k and alphabet.
    """
    def __init__(self, k: int, alphabet: List[str]):
        """
        Initialize the k-mer constructor.
        
        Args:
            k (int): Length of each k-mer.
            alphabet (List[str]): List of characters to construct k-mers.
        """
        self.k = k
        self.alphabet = alphabet
        
    def build_vocab(self, data: List[str], vocab: 'Vocabulary') -> None:
        """
        Build the vocabulary by generating all possible k-mers from the alphabet.
        Ignores the input data as the vocabulary is exhaustive.
        
        Args:
            data (List[str]): List of raw sequences. (Ignored)
            vocab (Vocabulary): Vocabulary instance to populate.
        """

        # Generate all possible k-mers using Cartesian product
        vocab.set_alphabet(self.alphabet)
        all_kmers = [''.join(p) for p in product(self.alphabet, repeat=self.k)]
        
        # Add all k-mers to the Vocabulary
        for kmer in sorted(all_kmers):
            vocab.add_token(kmer)


class fKmerVocabConstructor(VocabConstructor):
    """
    Constructs a k-mer vocabulary based on token frequency in the provided data.
    
    Parameters:
      k (int): Length of each k-mer.
      alphabet (List[str]): Alphabet to use.
      max_size (int, optional): Maximum number of tokens to keep. If None, include all tokens.
      overlapping (bool): If True, use overlapping k-mers; else, use non-overlapping tokens.
    """
    def __init__(self, k: int, alphabet: list, max_size: int = None, overlapping: bool = True):
        self.k = k
        self.alphabet = alphabet
        self.max_size = max_size
        self.overlapping = overlapping
        self.token_counts = None  # will hold frequency counts

    def build_vocab(self, data: list[str], vocab: 'Vocabulary') -> None:
        # Set the alphabet in the Vocabulary
        vocab.set_alphabet(self.alphabet)
        token_counts = collections.Counter()
        
        for seq in data:
            seq = seq.strip()
            if len(seq) < self.k:
                continue
            if self.overlapping:
                # Overlapping k-mers: all starting positions
                for i in range(len(seq) - self.k + 1):
                    token = seq[i:i+self.k]
                    token_counts[token] += 1
            else:
                # Non-overlapping: step by k, so tokens don't overlap
                for i in range(0, len(seq) - self.k + 1, self.k):
                    token = seq[i:i+self.k]
                    token_counts[token] += 1

        # Save the counts for later use in histogram plotting.
        self.token_counts = token_counts

        # Select tokens by frequency (if max_size is provided)
        if self.max_size is not None:
            most_common_tokens = token_counts.most_common(self.max_size)
        else:
            most_common_tokens = token_counts.items()

        # Add tokens to the vocabulary (skip tokens already present, e.g. special tokens)
        for token, _ in most_common_tokens:
            if token not in vocab.token_to_id:
                vocab.add_token(token)
# samplers.py
from __future__ import annotations
from typing import Iterable
import numpy as np, torch
from torch.utils.data import Sampler, RandomSampler, WeightedRandomSampler

class StraightSampler(RandomSampler):
    """Plain random over dataset indices (replacement=False by default)."""
    pass

class SpeciesBalancedSampler(WeightedRandomSampler):
    """
    1/freq(species) sampling, replacement=True.
    Expect dataset to expose `df['species']` or a vector of species indices.
    """
    @staticmethod
    def from_dataset(ds, species_col: str = "species") -> "SpeciesBalancedSampler":
        if hasattr(ds, "df"):  # pandas case
            species = ds.df[species_col].astype(str).values
            # map to indices to count properly
            uniq, inv = np.unique(species, return_inverse=True)
            counts = np.bincount(inv).astype(np.float64)
            w = 1.0 / counts[inv]
        else:  # tensor of species_idx provided by dataset
            sp = np.array([int(ds[i]["species_idx"]) for i in range(len(ds))])
            counts = np.bincount(sp).astype(np.float64)
            w = 1.0 / counts[sp]
        weights = torch.tensor(w, dtype=torch.double)
        return SpeciesBalancedSampler(weights=weights, num_samples=len(weights), replacement=True)

class SpeciesPowerSampler(WeightedRandomSampler):
    """
    1 / (class_size ** alpha) sampling, replacement=True.
    - alpha=1.0 -> inverse-frequency (your old sampler)
    - alpha=0.0 -> uniform over samples
    Optional clipping to avoid extreme oversampling.
    """
    @staticmethod
    def from_dataset(
        ds,
        species_col: str = "species",
        alpha: float = 1.0,
        eps: float = 0.0,
        min_count: int = 0,# does this work?
        clip_max_ratio: float | None = None,  # e.g. 20.0 caps per-sample weight at 20× mean
        normalize: bool = True,
    ) -> "SpeciesPowerSampler":
        # ----- extract per-sample class ids -----
        if hasattr(ds, "df"):
            labels = ds.df[species_col].astype(str).values
            uniq, inv = np.unique(labels, return_inverse=True)
            counts = np.bincount(inv).astype(np.float64)
            cls_w = 1.0 / np.power(np.maximum(counts, min_count) + eps, alpha)
            w = cls_w[inv]
        else:
            # expect integer species indices from items
            sp = np.array([int(ds[i]["species_idx"]) for i in range(len(ds))], dtype=np.int64)
            counts = np.bincount(sp, minlength=int(sp.max()) + 1).astype(np.float64)
            cls_w = 1.0 / np.power(np.maximum(counts, min_count) + eps, alpha)
            w = cls_w[sp]

        if normalize:
            m = w.mean()
            if m > 0: w = w / m
        if clip_max_ratio is not None:
            cap = w.mean() * clip_max_ratio
            w = np.minimum(w, cap)

        weights = torch.tensor(w, dtype=torch.double)
        return SpeciesPowerSampler(weights=weights, num_samples=len(weights), replacement=True)
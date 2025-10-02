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

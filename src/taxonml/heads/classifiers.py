from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch
from torch import nn, Tensor
from taxonml.labels.space import LabelSpace

logger = logging.getLogger(__name__)


class HierarchicalHead(nn.Module):
    """
    Cascading hierarchical classifier that also works as a single-level head.

    Design:
      - Let levels = ["phylum","class",...,"species"] or any *contiguous slice*
        of CANON_LEVELS (validated at init).
      - For the first level r=0, input is the pooled encoder vector h_pool.
      - For subsequent levels r>0, the input is concat([h_pool, z_{r-1}]) where
        z_{r-1} are the *logits* from the previous level.
      - Always returns a list `logits_list`, len == len(levels).
      - No implicit softmax; trainers decide the loss/activation (e.g., CE, entmax).

    This matches your thesis cascade:
        z^(1) = head^(1)(h_pool)
        z^(r) = head^(r)([h_pool ⊕ z^(r-1)]) for r>1
    """

    def __init__(
        self,
        in_features: int,
        *,
        levels: List[str],
        class_sizes: Dict[str, int],
        bottleneck: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.levels: List[str] = LabelSpace.validate_levels(levels)
        self.in_features = in_features
        self.class_sizes = {lvl: int(class_sizes[lvl]) for lvl in self.levels}

        # Build a per-rank MLP head
        self._heads = nn.ModuleList()
        self._head_in_dims: List[int] = []   # for debugging/describe()

        for i, lvl in enumerate(self.levels):
            out_c = self.class_sizes[lvl]
            in_dim = in_features if i == 0 else (in_features + self.class_sizes[self.levels[i - 1]])
            self._head_in_dims.append(in_dim)

            mlp = nn.Sequential(
                nn.Linear(in_dim, bottleneck),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(bottleneck, out_c),
            )
            self._heads.append(mlp)

        logger.info(
            "HierarchicalHead built: " +
            ", ".join(f"{lvl}[in={din}->out={self.class_sizes[lvl]}]" for lvl, din in zip(self.levels, self._head_in_dims))
        )

    @property
    def head_input_dims(self) -> List[int]:
        """Input dimensionality per head (for logging/debug)."""
        return list(self._head_in_dims)

    def forward(self, pooled: Tensor, *, return_inputs: bool = False) -> List[Tensor] | Tuple[List[Tensor], List[Tensor]]:
        """
        Args:
            pooled: [B, H] pooled encoder embedding.
            return_inputs: if True, also return the input tensors fed to each head
                           (useful for deep debugging).

        Returns:
            logits_list: list of length K, each tensor is [B, C_lvl].
            If `return_inputs` is True, returns `(logits_list, inputs_list)` where each
            element in `inputs_list` is the actual input to that head (shape [B, in_dim]).
        
        Notes: 
            B = batch size
            H = hidden size / embedding dimension
            K = number of ranks/levels
            C_lvl = class count at that level
            in_dim = input dimension
        """
        inputs_dbg: List[Tensor] = []
        logits_list: List[Tensor] = []

        cur_in = pooled
        for i, head in enumerate(self._heads):
            if i > 0:
                # concat pooled with previous logits
                prev_logits = logits_list[-1]
                cur_in = torch.cat([pooled, prev_logits], dim=1)
            if return_inputs:
                inputs_dbg.append(cur_in)
            logits = head(cur_in)
            logits_list.append(logits)

        if return_inputs:
            return logits_list, inputs_dbg
        return logits_list

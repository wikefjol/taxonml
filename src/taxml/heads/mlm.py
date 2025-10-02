from __future__ import annotations

from typing import Optional
import torch
from torch import nn, Tensor


class MLMHead(nn.Module):
    """
    Masked Language Modeling (MLM) head for BERT-style pretraining.

    This module projects per-token hidden states to vocabulary logits.
    It is implemented as a small feed-forward network applied independently
    to each token position.

    Inputs
    -------
    x : Tensor
        Hidden states of shape [batch_size, seq_len, hidden_dim].

    Outputs
    -------
    logits : Tensor
        Vocabulary logits of shape [batch_size, seq_len, vocab_size].

    Key points
    ----------
    - Uses a 2-layer MLP with ReLU and dropout by default.
    - Supports optional weight tying with a token embedding matrix
      (common in BERT-like models). When tied, the output bias is removed.

    Example
    -------
    >>> vocab_size, hidden_dim = 70, 512
    >>> embeddings = nn.Embedding(vocab_size, hidden_dim)
    >>> head = MLMHead(in_features=hidden_dim, vocab_size=vocab_size, dropout=0.1)
    >>> head.tie_weights(embeddings)  # optional, will remove output bias
    >>> x = torch.randn(8, 128, hidden_dim)  # [batch, seq, hidden]
    >>> logits = head(x)                     # [8, 128, vocab_size]
    """

    def __init__(
        self,
        in_features: int,
        vocab_size: int,
        dropout: float = 0.1,
        hidden_dim: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        in_features : int
            Dimension of the incoming per-token hidden states.
        vocab_size : int
            Size of the output vocabulary.
        dropout : float, optional
            Dropout probability applied between the two linear layers.
        hidden_dim : int, optional
            Internal projection size. If None, defaults to `in_features`.
            Keeping this flexible allows experimenting with a smaller or larger
            bottleneck without changing the encoder dimension.
        """
        super().__init__()
        hidden_dim = hidden_dim or in_features

        # Expose the final projection explicitly for clarity and weight tying.
        self.decoder = nn.Linear(hidden_dim, vocab_size, bias=True)

        self.proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            self.decoder,
        )

        self._tied = False  # tracks whether tie_weights() has been applied

    def forward(self, x: Tensor) -> Tensor:
        """
        Project hidden states to vocabulary logits.

        Notes
        -----
        `nn.Linear` operates on the last dimension, so shapes are preserved:
        [B, L, H] → [B, L, V].
        """
        return self.proj(x)

    @torch.no_grad()
    def tie_weights(self, embedding: nn.Embedding) -> None:
        """
        Tie the decoder weights to a token embedding matrix.

        This shares parameters between input embeddings and output projection,
        reducing total parameters and often improving training stability.

        Behavior
        --------
        - Replaces `self.decoder.weight` with `embedding.weight`.
        - Removes the decoder bias (sets it to `None`) since a tied output layer
          typically omits bias.

        Raises
        ------
        ValueError
            If shapes are incompatible (vocab size or embedding dimension mismatch).
        """
        dec = self.decoder
        if dec.weight.shape != embedding.weight.shape:
            raise ValueError(
                "Cannot tie weights: decoder weight "
                f"{tuple(dec.weight.shape)} vs embedding weight "
                f"{tuple(embedding.weight.shape)}"
            )

        # Share weights; remove bias to match standard tied setups.
        dec.weight = embedding.weight
        dec.bias = None
        self._tied = True

    @property
    def tied(self) -> bool:
        """Whether the decoder weights are currently tied to an embedding."""
        return self._tied

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Literal, Any

import torch
from torch import nn, Tensor

from taxml.encoders.bert import set_frozen, describe_encoder
from taxml.heads.mlm import MLMHead
from taxml.heads.classifiers import HierarchicalHead
from taxml.labels.space import LabelSpace

logger = logging.getLogger(__name__)


class TaxonomyModel(nn.Module):
    """
    Unified model for pretraining (MLM) and hierarchical/single-rank classification.

    Dependency injection: pass an already-built encoder (e.g., HuggingFace BertModel).
    Modes:
      - "pretrain":   encoder + MLMHead → {"mlm_logits": [B,T,V]}
      - "classify":   encoder + HierarchicalHead → {"levels", "logits_by_level", ...}
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        mode: Literal["pretrain", "classify"],
        # classification config (mode="classify")
        levels: Optional[List[str]] = None,
        class_sizes: Optional[Dict[str, int]] = None,
        hierarchical_dropout: float = 0.3,
        bottleneck: int = 256,
        # pretraining config (mode="pretrain")
        vocab_size: Optional[int] = None,
        mlm_hidden: Optional[int] = None,
        mlm_dropout: float = 0.1,
        tie_emb: bool = True,
    ) -> None:
        super().__init__()
        self._mode: Literal["pretrain", "classify"] = mode

        # ---- Encoder (injected) ---------------------------------------------
        self.encoder = encoder  # expected to expose .config.hidden_size and .pooler_output in forward

        hidden_size: int = getattr(self.encoder.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Injected encoder must have .config.hidden_size")

        # ---- Heads -----------------------------------------------------------
        self.mlm_head: Optional[MLMHead] = None
        self.classifier: Optional[HierarchicalHead] = None
        self._levels: List[str] = []

        if self._mode == "pretrain":
            if vocab_size is None:
                raise ValueError("vocab_size is required for mode='pretrain'")
            if mlm_hidden is None:
                mlm_hidden = hidden_size
            self.mlm_head = MLMHead(in_features=hidden_size, vocab_size=vocab_size,
                                    dropout=mlm_dropout, hidden_dim=mlm_hidden)

            # Optionally tie decoder weights to input token embeddings
            if tie_emb:
                token_embeddings: Optional[nn.Embedding] = getattr(
                    getattr(self.encoder, "embeddings", None), "word_embeddings", None
                )
                if isinstance(token_embeddings, nn.Embedding):
                    try:
                        self.mlm_head.tie_weights(token_embeddings)
                        logger.info("MLM head weights tied to encoder token embeddings.")
                    except ValueError as e:
                        logger.warning(f"Could not tie MLM weights: {e}")

        elif self._mode == "classify":
            if not levels or not class_sizes:
                raise ValueError("For mode='classify', both `levels` and `class_sizes` must be provided.")
            self._levels = LabelSpace.validate_levels(levels)
            self._class_sizes = dict(class_sizes)  # <-- keep for repr
            self.classifier = HierarchicalHead(
                in_features=hidden_size,
                levels=self._levels,
                class_sizes=class_sizes,
                bottleneck=bottleneck,
                dropout=hierarchical_dropout,
            )
        else:  # pragma: no cover
            raise ValueError(f"Unknown mode: {self._mode}")

        logger.info(self.describe())


    def __repr__(self) -> str:
        # --- tiny local formatters (no external deps) --------------------------
        def _indent_block(text: str, n: int = 2) -> str:
            pad = " " * n
            return "\n".join(pad + line if line else line for line in text.splitlines())

        def _fmt_list(lst, max_front: int = 6, max_back: int = 3) -> str:
            lst = list(lst)
            if len(lst) <= max_front + max_back:
                return "[" + ", ".join(map(str, lst)) + f"] ({len(lst)})"
            head = ", ".join(map(str, lst[:max_front]))
            tail = ", ".join(map(str, lst[-max_back:]))
            return "[" + head + ", …, " + tail + f"] ({len(lst)})"

        def _fmt_dict_middle(d: dict, max_items: int = 10, tail_items: int = 5) -> str:
            items = list(d.items())
            if len(items) <= max_items + tail_items:
                body = ",\n".join(f'  "{k}": {v}' for k, v in items)
            else:
                head = items[:max_items]
                tail = items[-tail_items:]
                head_s = ",\n".join(f'  "{k}": {v}' for k, v in head)
                tail_s = ",\n".join(f'  "{k}": {v}' for k, v in tail)
                body = head_s + ",\n  \"…\": \"… " + str(len(items) - (max_items + tail_items)) + " more …\",\n" + tail_s
            return "{\n" + body + "\n}"

        # --- counts -------------------------------------------------------------
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        # --- encoder block ------------------------------------------------------
        enc = self.encoder
        cfg = getattr(enc, "config", None)
        enc_lines = []
        if cfg is not None:
            enc_lines.append(f"arch={enc.__class__.__name__}")
            enc_lines.append(
                "dims="
                f"hidden={getattr(cfg,'hidden_size', 'NA')}, "
                f"layers={getattr(cfg,'num_hidden_layers','NA')}, "
                f"heads={getattr(cfg,'num_attention_heads','NA')}, "
                f"intermediate={getattr(cfg,'intermediate_size','NA')}"
            )
            enc_lines.append(
                "embeddings="
                f"vocab={getattr(cfg,'vocab_size','NA')}, "
                f"max_position_embeddings={getattr(cfg,'max_position_embeddings','NA')}"
            )
            enc_lines.append(
                "dropout="
                f"hidden={getattr(cfg,'hidden_dropout_prob','NA')}, "
                f"attn={getattr(cfg,'attention_probs_dropout_prob','NA')}"
            )
            enc_block = "Encoder(\n" + _indent_block("\n".join(enc_lines), 2) + "\n)"
        else:
            # Fallback: use torch nn.Module repr (indented)
            enc_block = "Encoder(\n" + _indent_block(repr(enc), 2) + "\n)"

        # --- head block ---------------------------------------------------------
        if self._mode == "pretrain":
            # MLM head details
            tied = False
            vocab_sz = None
            mlm_hidden = None
            mlm_drop = None
            if self.mlm_head is not None:
                tied = bool(getattr(self.mlm_head, "tied", False))
                vocab_sz = getattr(self.mlm_head, "vocab_size", None)
                mlm_hidden = getattr(self.mlm_head, "hidden_dim", None)
                mlm_drop = getattr(self.mlm_head, "dropout_p", None) or getattr(self.mlm_head, "dropout", None)
            head_lines = [
                "type=MLMHead" + (" (tied)" if tied else ""),
                f"vocab_size={vocab_sz}",
                f"hidden={mlm_hidden}, dropout={mlm_drop}",
            ]
            head_block = "Head(\n" + _indent_block("\n".join(head_lines), 2) + "\n)"
        else:
            # Hierarchical head details
            levels = getattr(self, "_levels", [])
            class_sizes = getattr(self, "_class_sizes", {})
            head_dims = getattr(self.classifier, "head_input_dims", [])
            bottleneck = getattr(self.classifier, "bottleneck", None)
            hdrop = getattr(self.classifier, "dropout_p", None) or getattr(self.classifier, "dropout", None)

            head_lines = [
                "type=HierarchicalHead",
                f"levels={_fmt_list(levels)}",
                "class_sizes=" + _fmt_dict_middle(class_sizes, max_items=6, tail_items=3),
                f"bottleneck={bottleneck}, dropout={hdrop}",
                f"head_input_dims={_fmt_list(head_dims)}",
            ]
            head_block = "Head(\n" + _indent_block("\n".join(head_lines), 2) + "\n)"

        # --- assemble -----------------------------------------------------------
        lines = [
            f"TaxonomyModel(mode={self._mode})",
            "Parameters(",
            _indent_block(
                f"total={total_params:,}, trainable={trainable_params:,}, frozen={frozen_params:,}"
            , 2),
            ")",
            enc_block,
            head_block,
        ]
        return "\n".join(lines)

    # ---------------------------- Convenience constructors -------------------

    @classmethod
    def for_pretrain(
        cls,
        *,
        encoder: nn.Module,
        vocab_size: int,
        mlm_hidden: Optional[int] = None,
        mlm_dropout: float = 0.1,
        tie_emb: bool = True,
    ) -> "TaxonomyModel":
        return cls(
            encoder=encoder,
            mode="pretrain",
            vocab_size=vocab_size,
            mlm_hidden=mlm_hidden,
            mlm_dropout=mlm_dropout,
            tie_emb=tie_emb,
        )

    @classmethod
    def for_classify(
        cls,
        *,
        encoder: nn.Module,
        levels: List[str],
        class_sizes: Dict[str, int],
        hierarchical_dropout: float = 0.3,
        bottleneck: int = 256,
    ) -> "TaxonomyModel":
        return cls(
            encoder=encoder,
            mode="classify",
            levels=levels,
            class_sizes=class_sizes,
            hierarchical_dropout=hierarchical_dropout,
            bottleneck=bottleneck,
        )

    # ---------------------------- Public API ---------------------------------

    @property
    def mode(self) -> Literal["pretrain", "classify"]:
        return self._mode

    def set_mode(self, mode: Literal["pretrain", "classify"]) -> None:
        if mode == self._mode:
            return
        if mode == "pretrain" and self.mlm_head is None:
            raise RuntimeError("MLM head not initialized for pretraining mode.")
        if mode == "classify" and self.classifier is None:
            raise RuntimeError("Classification head not initialized for classify mode.")
        self._mode = mode

    def freeze_encoder(self, frozen: bool = True) -> None:
        set_frozen(self.encoder, frozen)

    # ---------------------------- Forward ------------------------------------

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        *,
        return_debug: bool = False,
    ) -> Dict[str, Any]:
        """
        Returns:
          mode="pretrain":
            {"mlm_logits": [B,T,V]}
          mode="classify":
            {
              "levels": List[str],
              "logits_by_level": { level: [B, C_level], ... },
              # if return_debug:
              "pooled": [B, H],
              "head_input_dims": List[int],
            }
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        seq = outputs.last_hidden_state  # [B, T, H]

        if self._mode == "pretrain":
            assert self.mlm_head is not None
            mlm_logits = self.mlm_head(seq)  # [B, T, V]
            return {"mlm_logits": mlm_logits}

        # classify
        assert self.classifier is not None
        pooled = outputs.pooler_output if getattr(outputs, "pooler_output", None) is not None else seq[:, 0]
        logits_list = self.classifier(pooled)  # List[Tensor[B, C_r]]

        result: Dict[str, Any] = {
            "levels": list(self._levels),
            "logits_by_level": {lvl: logits for lvl, logits in zip(self._levels, logits_list)},
        }
        if return_debug:
            result["pooled"] = pooled
            result["head_input_dims"] = self.classifier.head_input_dims
        return result

    # ---------------------------- Introspection ------------------------------

    def describe(self) -> str:
        parts = [describe_encoder(self.encoder)]
        if self._mode == "pretrain":
            parts.append("Head: MLMHead (tied)" if getattr(self.mlm_head, "tied", False) else "Head: MLMHead")
        else:
            assert self.classifier is not None
            lvl = ", ".join(self.classifier.levels)
            dims = ", ".join(str(d) for d in self.classifier.head_input_dims)
            parts.append(
                f"Head: HierarchicalHead(levels=[{lvl}])\n"
                f"  head_input_dims=[{dims}] (pooled ⊕ prev-logits for r>0)"
            )
        return "\n".join(parts)

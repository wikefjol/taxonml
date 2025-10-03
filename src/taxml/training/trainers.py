# Standard library
import logging
logger = logging.getLogger(__name__)

import math
import os
import time
import shutil
import json
from typing import Any, Dict, List, Optional, Set, Tuple, Sequence, NamedTuple
from collections.abc import Mapping

# Third-party
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")  # headless / no-GUI backend
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, MaxNLocator

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from entmax import Entmax15Loss

# Local project
from taxml.core.constants import IGNORE_INDEX
from taxml.metrics.rank import compute_rank_metrics

TensorByLevel = Dict[str, torch.Tensor]
MaskCache = Dict[str, Tuple[torch.Tensor, torch.Tensor]]



class ForwardStep(NamedTuple):
    """Forward+loss result for one batch."""
    loss_total: torch.Tensor                         # scalar
    loss_by_level: dict[str, torch.Tensor]           # {level: scalar}
    logits_by_level: dict[str, torch.Tensor]         # {level: [B,C_full]}
    labels_by_level: dict[str, torch.Tensor]         # {level: [B]}
    batch_size: int

class OutputContractError(ValueError): pass

def extract_logits_by_level(
    out: Mapping[str, Any], levels: Sequence[str]
) -> Dict[str, torch.Tensor]:
    """
    Validate and extract per-level logits from a model forward output.

    Contract:
    - `out` must be a Mapping (dict-like).
    - It must contain either:
        * "logits_by_level": Mapping[level -> Tensor[B, C]]
        * or "logits": same structure (accepted as an alias).
    - Keys of the logits mapping must exactly match `levels`.
    - Each tensor must be 2D with shape [batch_size, num_classes].

    Args:
        out: Forward output from the model (dict-like).
        levels: Expected set/list of level identifiers (str).

    Returns:
        Dict[str, torch.Tensor]: Mapping from level name to [B, C] logits.

    Raises:
        OutputContractError: if the output does not meet the contract.
    """
    # Must be a dict-like structure
    if not isinstance(out, Mapping):
        raise OutputContractError("forward must return a Mapping[str, Any]")

    # Support 'logits_by_level' (preferred) or 'logits' (alias)
    logits = out.get("logits_by_level", out.get("logits"))
    if not isinstance(logits, Mapping):
        raise OutputContractError("expected 'logits_by_level': Mapping[level, Tensor]")

    # Levels must match exactly
    want, got = set(levels), set(logits)
    if got != want:
        missing, extra = sorted(want - got), sorted(got - want)
        raise OutputContractError(f"levels mismatch: missing={missing}, extra={extra}")

    # Each value must be a 2D tensor [B, C]
    for k, v in logits.items():
        if not torch.is_tensor(v) or v.dim() != 2:
            shape = tuple(v.shape) if torch.is_tensor(v) else type(v)
            raise OutputContractError(f"{k}: logits must be [B,C], got {shape}")

    return logits  # Dict[str, Tensor[B, C]]

def extract_mlm_logits(out: Any) -> torch.Tensor:
    """
    Validate and extract masked language modeling (MLM) logits.

    Contract:
    - `out` may be:
        * A Mapping with "mlm_logits" (preferred) or "logits" (alias).
        * A raw Tensor of logits.
    - The resulting tensor must be 3D with shape [batch_size, seq_len, vocab_size].

    Args:
        out: Forward output from the model (dict-like or Tensor).

    Returns:
        torch.Tensor: MLM logits with shape [B, T, V].

    Raises:
        OutputContractError: if the output does not meet the contract.
    """
    # Support dict-like outputs or direct Tensor
    if isinstance(out, Mapping):
        logits = out.get("mlm_logits", out.get("logits"))
    else:
        logits = out

    # Must be a 3D tensor [B, T, V]
    if not (torch.is_tensor(logits) and logits.dim() == 3):
        raise OutputContractError(
            "MLM expects logits Tensor [B,T,V] or Mapping with 'mlm_logits'/'logits'"
        )

    return logits

def _save_checkpoint(path: str, model, optimizer, scheduler, epoch: int, extra: Dict, **kwargs):
    """
    Save a checkpoint with minimal training state.

    Always stores: model/scheduler/optimizer state, epoch, extra (user metrics).
    Optionally stores: scaler state, global_step, best_val_loss if provided via kwargs.

    Parameters
    ----------
    path : str
        Destination file path.
    model, optimizer, scheduler
        Torch modules/objects to serialize.
    epoch : int
        1-based epoch count when the checkpoint is written.
    extra : Dict
        Free-form metrics or metadata (e.g., val_loss).
    kwargs :
        Optional keys:
          - scaler: GradScaler or None
          - global_step: int
          - best_val: float
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "extra": {
            **extra,
            **({"global_step": int(kwargs["global_step"])} if "global_step" in kwargs else {}),
            **({"best_val_loss": float(kwargs["best_val"])} if "best_val" in kwargs else {}),
        },
        "scaler": (kwargs["scaler"].state_dict() if kwargs.get("scaler", None) is not None else None),
    }
    torch.save(payload, path)

def token_accuracy(logits: torch.Tensor,
                   labels: torch.Tensor,
                   ignore_index: int) -> float:
    """
    Compute token-level accuracy for MLM.
    Args:
      logits: [B, T, V]
      labels: [B, T] with ignore_index where loss/acc shouldn't count
    Returns:
      float in [0,1]
    """
    if not (torch.is_tensor(logits) and torch.is_tensor(labels)):
        return 0.0
    with torch.no_grad():
        # [B,T]
        pred = logits.argmax(dim=-1)
        mask = (labels != ignore_index)
        valid = mask.sum().item()
        if valid == 0:
            return 0.0
        correct = (pred.eq(labels) & mask).sum().item()
        return correct / valid


def _load_checkpoint(path: str, model, optimizer, scheduler, scaler: Optional[GradScaler]):
    """
    Load model/optimizer/scheduler(/scaler) states and return (start_epoch, global_step, best_val_loss).
    start_epoch is the epoch stored in the file; the caller typically continues from start_epoch.
    """
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    extra = ckpt.get("extra", {})
    start_epoch = int(ckpt.get("epoch", 0))
    global_step = int(extra.get("global_step", 0))
    best_val = float(extra.get("best_val_loss", math.inf))
    return start_epoch, global_step, best_val
LEVEL_ALIASES = {
    "phylum": "p", "class": "c", "order": "o",
    "family": "f", "genus": "g", "species": "s"
}

def _fmt_by_level(prefix: str, values_by_level: dict, levels: list[str], max_levels: int | None = None) -> str:
    """
    Build 'acc:p=0.843,c=0.771,...' or 'loss:tot=1.23,p=0.89,...'
    Order follows `levels`. Unknown levels get first letter.
    """
    if not values_by_level:
        return ""
    parts = []
    for lvl in levels:
        if lvl in values_by_level:
            k = LEVEL_ALIASES.get(lvl, lvl[:1])
            v = values_by_level[lvl]
            parts.append(f"{k}={v:.3f}")
    if max_levels and len(parts) > max_levels:
        parts = parts[:max_levels] + ["…"]
    return f"{prefix}:" + ",".join(parts)

def _build_postfix(*chunks: str, lr: float | None = None) -> str:
    parts = [c for c in chunks if c]
    if lr is not None: parts.append(f"lr={lr:.2e}")
    return "  ".join(parts)




class ClassificationTrainer:
    """
    Supports hierarchical (+single-rank as len(levels)==1).
    Loss: entmax15.
    Applies fold masks in loss only.
    """
    def __init__(
            self,
            model,
            train_loader,
            val_loader,
            levels: List[str],
            masks_train: Dict[str, List[int]],
            masks_val: Optional[Dict[str, List[int]]],
            optimizer,
            scheduler=None,
            amp=True,
            log_every=100,
            checkpoints_dir: Optional[str] = None,
            rank_metrics: Optional[Sequence[str]] = ("accuracy",),
            select_best_by: str = "species",
        ):
        """
        Args:
            model:          torch.nn.Module with hierarchical classification heads.
            train_loader:   torch DataLoader for training data.
            val_loader:     torch DataLoader for validation data.
            levels:         List of str; the classification hierarchy levels (e.g. ["phylum","class","species"]).
            masks_train:    Dict[level -> 0/1 list] fold-specific active classes for training split.
            masks_val:      Dict[level -> 0/1 list] fold-specific active classes for validation split, or None.
            optimizer:      torch optimizer for training.
            scheduler:      optional LR scheduler (stepped either per-batch or per-epoch depending on config).
            amp:            use automatic mixed precision (requires CUDA).
            log_every:      logging frequency (batches).
            checkpoints_dir: path for saving checkpoints and metrics logs.
            select_best_by: which level to use for "best model" selection.
        """

        # --- core components ---
        self.model = model
        self.device = next(self.model.parameters()).device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.levels = levels
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.amp = amp

        # --- AMP setup ---
        if amp and self.device.type != "cuda":
            raise RuntimeError(
                f"AMP is enabled but model is on {self.device}. "
                "Call model.to('cuda') before constructing the trainer or set amp=False."
            )
        self.scaler = GradScaler(enabled=amp)

        # --- bookkeeping ---
        self.log_every = int(log_every)
        self.ckpt = checkpoints_dir
        self.select_best_by = select_best_by  # e.g. "species"
        self.best_metric = -1.0
        self.best_path = None
        self.global_step = 0
        self.rank_metrics = tuple(rank_metrics)

        # --- criterion ---
        self.criterion = Entmax15Loss(
            reduction="elementwise_mean",
            ignore_index=IGNORE_INDEX
        ).to(self.device)

        # --- fold-specific class masking ---
        # Converts 0/1 lists into (active_idx, remap) tensors for each level
        self.mask_cache_train = self._build_mask_cache(masks_train)
        self.mask_cache_val   = self._build_mask_cache(masks_val) if masks_val else None

        # --- checkpoint + logging ---
        if self.ckpt:
            os.makedirs(self.ckpt, exist_ok=True)
            self.metrics_path = os.path.join(self.ckpt, "metrics.jsonl")   # epoch-level train/val
            self.batchlog_path = os.path.join(self.ckpt, "batches.jsonl")  # per-batch (appended once per epoch)
        else:
            self.metrics_path = None
            self.batchlog_path = None
        self.static_metadata = None

    def _append_jsonl(self, path: Optional[str], records: List[dict]) -> None:
        """Append a list of dict records to a JSONL file (no-op if path/records empty)."""
        if not path or not records:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            for rec in records:
                f.write(json.dumps(rec, default=float) + "\n")


    def _ensure_history(self):
        if hasattr(self, "_history"):
            return
        # one figure per phase; panels = metrics; lines = levels
        self._history = {
            "epoch": [],
            "train": {m: {lvl: [] for lvl in self.levels} for m in (*self.rank_metrics, "loss")},
            "val":   {m: {lvl: [] for lvl in self.levels} for m in (*self.rank_metrics, "loss")},
        }

    def _update_history(self, epoch: int, train_pkg: dict, val_pkg: dict) -> None:
        self._ensure_history()
        self._history["epoch"].append(int(epoch))

        for phase, pkg in (("train", train_pkg), ("val", val_pkg)):
            mb = pkg["epoch"]["metrics_by_level"]   # {level: {metric: value}}
            # rank metrics per level
            for lvl in self.levels:
                for m in self.rank_metrics:
                    v = float(mb.get(lvl, {}).get(m, float("nan")))
                    self._history[phase][m][lvl].append(v)
            # total loss (store once per level; duplicate into each level key to keep plotting simple)
            loss_v = float(pkg["epoch"]["loss"])
            for lvl in self.levels:
                self._history[phase]["loss"][lvl].append(loss_v)

    def _plot_batch_overview(self) -> None:
        """
        Batch-level plots (train only):
        • Global panel: total loss + learning rate
        • Per-rank losses (log y)
        • Per-rank accuracy (0–1)
        Produces batch_metrics_overview.png in checkpoint folder.
        """
        if not self.batchlog_path or not os.path.exists(self.batchlog_path):
            return

        import json
        import numpy as np
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FixedLocator, FixedFormatter

        # --- load JSONL ---
        recs = []
        with open(self.batchlog_path, "r") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if isinstance(d, dict) and d.get("phase") == "train":
                        recs.append(d)
                except Exception:
                    pass
        if not recs:
            return
        df = pd.DataFrame(recs)
        if df.empty:
            return

        # --- flatten nested dicts safely ---
        # loss_by_level → loss_<lvl>
        if "loss_by_level" in df.columns:
            raw = [d for d in df["loss_by_level"].dropna().tolist() if isinstance(d, dict)]
            loss_lvls = sorted({k for d in raw for k in d.keys() if isinstance(k, str)})
            for lvl in loss_lvls:
                df[f"loss_{lvl}"] = df["loss_by_level"].apply(
                    lambda d: (d or {}).get(lvl) if isinstance(d, dict) else None
                )

        # metrics_by_level (e.g., accuracy) → acc_<lvl>
        if "metrics_by_level" in df.columns:
            raw = [d for d in df["metrics_by_level"].dropna().tolist() if isinstance(d, dict)]
            acc_lvls = sorted({k for d in raw for k in d.keys() if isinstance(k, str)})
            for lvl in acc_lvls:
                df[f"acc_{lvl}"] = df["metrics_by_level"].apply(
                    lambda d: (d or {}).get(lvl, {}).get("accuracy") if isinstance(d, dict) else None
                )

        # --- smooth numerics for readability ---
        df_s = df.sort_values("step").copy()
        for c in df_s.columns:
            if c in ("epoch", "phase", "step"):
                continue
            if pd.api.types.is_numeric_dtype(df_s[c]):
                df_s[c] = df_s[c].rolling(5, min_periods=1).mean()

        # --- detect columns to plot ---
        loss_rank_cols = [c for c in df_s.columns if c.startswith("loss_") and c not in ("loss_by_level","loss_total")]
        acc_rank_cols  = [c for c in df_s.columns if c.startswith("acc_")]
        n_rows = 1 + (1 if loss_rank_cols else 0) + (1 if acc_rank_cols else 0)

        fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3.0*n_rows), sharex=True)
        if n_rows == 1:
            axes = [axes]

        # --- panel 1: global (loss_total + lr) ---
        ax = axes[0]
        if "loss_total" in df_s:
            ax.plot(df_s["step"], df_s["loss_total"], label="total_loss")
            ax.set_ylabel("total loss")
        ax.grid(True, ls="--", lw=.5, alpha=.7)
        ax2 = ax.twinx()
        if "lr" in df_s:
            ax2.plot(df_s["step"], df_s["lr"], alpha=.4, label="lr")
            ax2.set_ylabel("learning rate")
        ax.set_title("Batch metrics (train) — global")
        if ax.get_legend_handles_labels()[1]:
            ax.legend(loc="upper right")
        if ax2.get_legend_handles_labels()[1]:
            ax2.legend(loc="upper left")

        # --- panel 2: per-rank losses (log scale) ---
        r = 1
        if loss_rank_cols:
            axl = axes[r]; r += 1
            for c in sorted(loss_rank_cols):
                axl.plot(df_s["step"], df_s[c], label=c.replace("loss_", ""))
            axl.set_ylabel("loss")
            axl.set_title("Per-rank loss (train)")
            axl.set_yscale("log")
            axl.grid(True, ls="--", lw=.5, alpha=.7)
            if axl.get_legend_handles_labels()[1]:
                axl.legend(ncol=max(1, len(loss_rank_cols)//6), fontsize=8)

        # --- panel 3: per-rank accuracy ---
        if acc_rank_cols:
            axa = axes[r]; r += 1
            for c in sorted(acc_rank_cols):
                axa.plot(df_s["step"], df_s[c], label=c.replace("acc_", ""))
            axa.set_ylabel("accuracy"); axa.set_ylim(0, 1)
            axa.set_title("Per-rank accuracy (train)")
            axa.grid(True, ls="--", lw=.5, alpha=.7)
            if axa.get_legend_handles_labels()[1]:
                axa.legend(ncol=max(1, len(acc_rank_cols)//6), fontsize=8)

        # --- x-axis: global steps, with epoch ticks ---
        if "epoch" in df_s.columns:
            starts = df_s.groupby("epoch")["step"].min().dropna().astype(float).sort_index()
            if len(starts) >= 1:
                major_pos = starts.values
                major_lab = [str(int(e)) for e in starts.index.tolist()]
                for ax_ in axes:
                    ax_.xaxis.set_major_locator(FixedLocator(major_pos))
                    ax_.xaxis.set_major_formatter(FixedFormatter(major_lab))

                # optional: faint epoch separators
                for ax_ in axes:
                    for s in major_pos:
                        ax_.axvline(s, alpha=0.12, linewidth=1)

        axes[-1].set_xlabel("Global step   (major ticks = epoch starts)")
        fig.tight_layout()
        out_png = os.path.join(self.ckpt, "batch_metrics_overview.png")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)

    def _plot_epoch_overview(self) -> None:
        if not self.metrics_path or not os.path.exists(self.metrics_path):
            return


        # --- load epoch lines ---
        recs = []
        with open(self.metrics_path, "r") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if isinstance(d, dict) and d.get("type") == "epoch":
                        recs.append(d)
                except Exception:
                    pass
        if not recs:
            return
        df = pd.DataFrame(recs)

        # --- helpers to widen dict fields safely ---
        def _extract_metric(df_phase, metric_name):
            rows = []
            for _, r in df_phase.iterrows():
                mb = r.get("metrics_by_level", None)
                if isinstance(mb, dict):
                    row = {"epoch": int(r["epoch"])}
                    for lvl, md in mb.items():
                        if isinstance(lvl, str) and isinstance(md, dict) and metric_name in md:
                            row[f"{metric_name}_{lvl}"] = md[metric_name]
                    rows.append(row)
            out = pd.DataFrame(rows)
            if out.empty:
                return out
            return out.groupby("epoch").last().reset_index()

        def _extract_loss_by_level(df_phase):
            rows = []
            for _, r in df_phase.iterrows():
                lb = r.get("loss_by_level", None)
                if isinstance(lb, dict):
                    row = {"epoch": int(r["epoch"])}
                    for lvl, v in lb.items():
                        if isinstance(lvl, str):
                            row[f"loss_{lvl}"] = v
                    rows.append(row)
            out = pd.DataFrame(rows)
            if out.empty:
                return out
            return out.groupby("epoch").last().reset_index()

        # --- discover available metric names (accuracy, etc.) ---
        metric_names = set()
        if "metrics_by_level" in df.columns:
            for mb in df["metrics_by_level"].dropna().tolist():
                if isinstance(mb, dict):
                    for md in mb.values():
                        if isinstance(md, dict):
                            for k in md.keys():
                                if isinstance(k, str):
                                    metric_names.add(k)
        metric_names = sorted(metric_names) or ["accuracy"]

        df_tr = df[df["phase"] == "train"].copy()
        df_va = df[df["phase"] == "val"].copy()

        # check if per-rank loss_by_level exists
        has_rank_loss = any(isinstance(x, dict) and x for x in df.get("loss_by_level", []))

        # --- figure layout ---
        n_rows = 1 + len(metric_names) + (1 if has_rank_loss else 0)
        fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3.2*n_rows), sharex=True)
        if n_rows == 1:
            axes = [axes]

        # --- global loss (train dashed, val solid) ---
        ax0 = axes[0]
        if "loss" in df_tr.columns and not df_tr.empty:
            ax0.plot(df_tr["epoch"], df_tr["loss"], label="train_loss", linestyle="--")
        if "loss" in df_va.columns and not df_va.empty:
            ax0.plot(df_va["epoch"], df_va["loss"], label="val_loss", linestyle="-")
        ax0.set_ylabel("loss")
        ax0.set_title("Epoch metrics — global")
        ax0.grid(True, ls="--", lw=.5, alpha=.7)
        if ax0.get_legend_handles_labels()[1]:
            ax0.legend()

        # --- consistent colors per rank across all panels ---
        # Build the set of ranks we might plot from metrics and (optionally) loss_by_level
        ranks = set()
        for side in (df_tr, df_va):
            for mb in side.get("metrics_by_level", []):
                if isinstance(mb, dict):
                    for lvl in mb.keys():
                        if isinstance(lvl, str):
                            ranks.add(lvl)
        if has_rank_loss:
            for side in (df_tr, df_va):
                for lb in side.get("loss_by_level", []):
                    if isinstance(lb, dict):
                        for lvl in lb.keys():
                            if isinstance(lvl, str):
                                ranks.add(lvl)
        ranks = sorted(ranks)
        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", None) or [f"C{i}" for i in range(10)]
        color_map = {lvl: colors[i % len(colors)] for i, lvl in enumerate(ranks)}

        # --- per-metric panels (train dashed, val solid) ---
        for i, mname in enumerate(metric_names, start=1):
            ax = axes[i]
            tr_w = _extract_metric(df_tr, mname)
            va_w = _extract_metric(df_va, mname)

            # union of rank columns
            cols = sorted({c for c in getattr(tr_w, "columns", []) if c != "epoch"} |
                        {c for c in getattr(va_w, "columns", []) if c != "epoch"})

            for c in cols:
                rank = c.replace(f"{mname}_", "")
                if not isinstance(rank, str):
                    continue
                if not tr_w.empty and c in tr_w:
                    ax.plot(tr_w["epoch"], tr_w[c], label=f"train {rank}", linestyle="--", color=color_map.get(rank))
                if not va_w.empty and c in va_w:
                    ax.plot(va_w["epoch"], va_w[c], label=f"val {rank}",   linestyle="-",  color=color_map.get(rank))

            if mname == "accuracy":
                ax.set_ylim(0, 1)

            ax.set_ylabel(mname)
            ax.set_title(f"Per-rank {mname} (train dashed, val solid)")
            ax.grid(True, ls="--", lw=.5, alpha=.7)
            if ax.get_legend_handles_labels()[1]:
                ax.legend(ncol=2, fontsize=8)

        # --- optional: per-rank loss panel if loss_by_level was logged ---
        if has_rank_loss:
            axl = axes[-1] if axes[-1].get_ylabel() == "" else axes[len(metric_names)+1]
            tr_l = _extract_loss_by_level(df_tr)
            va_l = _extract_loss_by_level(df_va)
            cols = sorted({c for c in getattr(tr_l, "columns", []) if c != "epoch"} |
                        {c for c in getattr(va_l, "columns", []) if c != "epoch"})

            for c in cols:
                rank = c.replace("loss_", "")
                if not tr_l.empty and c in tr_l:
                    axl.plot(tr_l["epoch"], tr_l[c], label=f"train {rank}", linestyle="--", color=color_map.get(rank))
                if not va_l.empty and c in va_l:
                    axl.plot(va_l["epoch"], va_l[c], label=f"val {rank}",   linestyle="-",  color=color_map.get(rank))

            axl.set_ylabel("loss")
            axl.set_title("Per-rank loss (train dashed, val solid)")
            axl.grid(True, ls="--", lw=.5, alpha=.7)
            if axl.get_legend_handles_labels()[1]:
                axl.legend(ncol=2, fontsize=8)

        # --- integer x-ticks only ---
        for ax in axes:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        axes[-1].set_xlabel("Epoch")
        fig.tight_layout()
        out_png = os.path.join(self.ckpt, "epoch_metrics_overview.png")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)



    def _build_mask_cache(self, masks: dict[str, list[int]]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """
        Build a cache of active class indices and global→local remap tensors for each taxonomy level.

        Purpose:
        • Aligns global class space (all possible labels for a level) with the fold-specific
            subset of classes that are actually present in this split.
        • Avoids varying head dimensions across folds: the classifier head always outputs
            logits for the full label space, but masking makes training fold-specific.
        • Each level gets:
            (active_idx, remap)
                - active_idx: indices of active classes [C_active] in the global space
                - remap: LongTensor of size [C_full], mapping each global class ID → [0..C_active-1]
                        or IGNORE_INDEX for inactive classes.

        Args:
            masks: dict mapping {level → list of 0/1 flags of length = num_classes_full}
                where 1 marks presence in this fold's split.

        Returns:
            cache: dict mapping {level → (active_idx, remap)}
        """
        cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for lvl in self.levels:
            # Convert Python list to a boolean tensor [C_full]
            mask = torch.tensor(masks[lvl], dtype=torch.bool, device=self.device)

            # Gather active class indices, e.g. [3,7,12]
            active_idx = mask.nonzero(as_tuple=False).squeeze(1)

            # Build a remap array from full space → compact local space
            remap = torch.full((mask.numel(),), IGNORE_INDEX,
                            dtype=torch.long, device=self.device)
            if active_idx.numel() > 0:
                remap[active_idx] = torch.arange(active_idx.numel(), device=self.device)

            cache[lvl] = (active_idx, remap)
        return cache

    def _apply_label_fold_mask(
        self,
        lvl: str,
        logits_global: torch.Tensor,      # [B, C_full]
        labels_global: torch.Tensor,    # [B] (global class ids)
        mask_cache: MaskCache | None = None,
        *,
        strict: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Restrict a level's logits/labels to the fold-active classes and remap labels to
        a compact local id space.

        This helper does three things if a mask is defined for the level:
        1) Slices logits to the active class columns (C_full → C_active).
        2) Remaps GLOBAL class ids → LOCAL contiguous ids using the provided `remap`.
        3) Handles labels that are not part of the active set (marked as IGNORE_INDEX):
            - strict=True  → raise with a helpful diagnostic
            - strict=False → drop those samples from the loss (may return empty tensors)

        If no mask exists for the level, the inputs are returned unchanged.

        Args:
        lvl: Taxonomy level name (e.g., "species").
        logits_global: Logits over the full class space, shape [B, C_full].
        labels_global: Ground-truth labels as GLOBAL class ids, shape [B].
        mask_cache: Mapping {level → (active_idx, remap)}, where:
            - active_idx: LongTensor of shape [C_active] with indices into the full class space.
            - remap:      LongTensor of shape [C_full] mapping GLOBAL id → LOCAL id,
                            or IGNORE_INDEX for inactive classes.
        strict: When True, any label outside the active set triggers an error; when False,
                such samples are filtered out before loss computation.

        Returns:
        (logits_local, labels_local)
            - If masked: shapes are [B', C_active] and [B'] (B' ≤ B if strict=False).
            - If unmasked: returns (logits_global, labels_global).

        Raises:
        ValueError: If the level has a mask with an empty active set.
        RuntimeError: If strict=True and any label is not covered by the mask.

        Notes:
        - IGNORE_INDEX comes from the project constants and marks labels to exclude from loss.
        - Use strict=True during development to fail fast on mask/dataset mismatches.
        """
        
        # No-op. Standard for validation 
        if mask_cache is None:
            return logits_global, labels_global
        entry = mask_cache.get(lvl)
        if entry is None:
            # No masking for this level: use the full class space.
            return logits_global, labels_global
        active_idx, remap = entry
        if active_idx.numel() == 0:
            # Mask exists but no classes are active: this is a configuration error.
            raise ValueError(
                f"Level '{lvl}' has an empty active set (C_full={remap.numel()}). "
                "Remove this level from `levels` if you intend to skip it."
            )

        # 1) Slice logits to the active class columns → [B, C_active]
        logits_local = logits_global.index_select(1, active_idx)

        # 2) Remap GLOBAL → LOCAL label ids; inactive classes become IGNORE_INDEX → [B]
        labels_local = remap[labels_global]

        # 3) Handle labels outside the active set
        invalid = (labels_local == IGNORE_INDEX)
        if strict:
            if invalid.any():
                bad = int(invalid.sum().item())
                bad_ids = labels_global[invalid][:8].tolist()
                raise RuntimeError(
                    f"Level '{lvl}': {bad}/{labels_global.numel()} labels not covered by mask. "
                    f"Examples of bad global ids: {bad_ids}. "
                    "Ensure fold masks match the dataset and label space."
                )
            return logits_local, labels_local

        # Non-strict path: drop invalid samples (can result in empty tensors)
        valid = ~invalid
        if not valid.any():
            # Caller may decide to skip this level's loss if empty.
            return logits_local[:0], labels_local[:0]

        return logits_local[valid], labels_local[valid]
    
    def _forward_step(self, *, batch: dict) -> tuple[dict[str, torch.Tensor], int]:
        """
        Run the model forward pass for one batch.

        Moves inputs to the correct device, executes the model
        and validates that outputs follow the contract for hierarchical logits.

        Args:
            batch: A minibatch dict containing "input_ids" and "attention_mask".

        Returns:
            logits_by_level: Dict[level → Tensor[B, C_full]] with full-class logits.
            batch_size: Number of samples in the batch (B).
        """
        dev = self.device
        input_ids = batch["input_ids"].to(dev, non_blocking=True)
        batch_size = int(input_ids.size(0))
        attention_mask = batch["attention_mask"].to(dev, non_blocking=True) 
    
        # forward call
        out = self.model(
            input_ids = input_ids,
            attention_mask = attention_mask
            )
        # verifiy contract and extract logits
        logits_by_level = extract_logits_by_level(
            out = out,
            levels = self.levels
            )

        return logits_by_level, batch_size


    def _loss_step(
        self,
        *,
        logits_by_level: dict[str, torch.Tensor],
        labels_by_level: dict[str, torch.Tensor],
        mask_cache: MaskCache | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute masked losses for each taxonomy level.

        Applies fold-specific masks (slice logits and remap labels) before computing
        loss per level. Returns both the per-level mapping and the summed total.

        Args:
            logits_by_level: Dict[level → Tensor[B, C_full]] of model logits.
            labels_by_level: Dict[level → Tensor[B]] of ground truth labels.
            mask_cache: Dict[level → (active_idx, remap)] fold-specific class maps.

        Returns:
            loss_total: Scalar total loss (sum of all levels).
            loss_by_level: Dict[level → scalar Tensor] of per-level losses.
        """
        loss_by_level: dict[str, torch.Tensor] = {}
        for lvl in self.levels:
            lg, yg = logits_by_level[lvl], labels_by_level[lvl]
            ll, yl = self._apply_label_fold_mask(lvl, lg, yg, mask_cache, strict=True)
            loss_by_level[lvl] = self.criterion(ll, yl)

        if not loss_by_level:
            raise RuntimeError("Empty loss set; check levels, masks, and labels.")

        loss_total = torch.stack(list(loss_by_level.values())).sum()
        return loss_total, loss_by_level


    def _forward_loss_step(
        self,
        *,
        batch: dict,
        mask_cache: MaskCache | None = None,
        use_amp: bool = True,
    ) -> ForwardStep:
        """
        Full forward pass + loss computation for one batch.

        Runs the model, validates outputs, applies fold masks, and computes losses.
        Keeps logits in the full class space; masking is used only for loss.

        Args:
            batch: A minibatch dict with "input_ids", "attention_mask", and labels.
            mask_cache: Dict[level → (active_idx, remap)] fold-specific class maps.
            use_amp: Whether to use autocast mixed precision.

        Returns:
            ForwardStep dataclass with:
                - loss_total: Scalar loss for all levels.
                - loss_by_level: Dict[level → scalar Tensor].
                - logits_by_level: Dict[level → Tensor[B, C_full]].
                - labels_by_level: Dict[level → Tensor[B]].
                - batch_size: Number of samples in the batch.
        """
        with autocast(enabled=use_amp):
            # move labels to device
            labels_by_level = {
                lvl: batch["labels_by_level"][lvl].to(self.device, non_blocking=True)
                for lvl in self.levels if lvl in batch["labels_by_level"]
            }

            # forward pass
            logits_by_level, batch_size = self._forward_step(batch=batch)

            # compute losses
            loss_total, loss_by_level = self._loss_step(
                logits_by_level=logits_by_level,
                labels_by_level=labels_by_level,
                mask_cache=mask_cache,
            )

        return ForwardStep(
            loss_total=loss_total,
            loss_by_level=loss_by_level,
            logits_by_level=logits_by_level,
            labels_by_level=labels_by_level,
            batch_size = batch_size
        )

    def _run_phase(
        self,
        phase: str,
        *,
        dataloader,
        mask_cache: MaskCache | None = None,
        current_epoch: int,
    ) -> dict:
        assert phase in {"train", "val"}
        is_train = (phase == "train")
        use_amp = bool(getattr(self, "amp", False)) and torch.cuda.is_available()

        # mode per phase
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        _batch_log_buffer: list[dict] = []
        
        bar_fmt =  (
            "{desc} {percentage:3.0f}%"
            "[{elapsed}<{remaining}]{postfix}"
        )
        pbar = tqdm(
            iterable=dataloader,
            desc=f"Ep {current_epoch:<3}[{phase:<5}]",
            total=len(dataloader),
            bar_format=bar_fmt,
            leave=False,
            dynamic_ncols=False,  # prevent bouncing
        )
        # epoch accumulators: {level: {metric_name: {"num": int, "den": int}}}
        epoch_accums: Dict[str, Dict[str, Dict[str, int]]] = {
            lvl: {m: {"num": 0, "den": 0} for m in self.rank_metrics} for lvl in self.levels
        }
        loss_by_level_sums = {lvl: 0.0 for lvl in self.levels}
        loss_sum = 0.0
        n_batches = 0

        # disable grads in validation
        grad_context_manager = (torch.enable_grad() if is_train else torch.inference_mode())
        with grad_context_manager:
            for batch in pbar:
                if is_train:
                    self.optimizer.zero_grad(set_to_none=True)  # <-- only in train

                forward_step = self._forward_loss_step(batch=batch, mask_cache=mask_cache, use_amp=use_amp)
                for lvl, t in forward_step.loss_by_level.items():
                    loss_by_level_sums[lvl] += float(t.detach().item())
                if is_train:
                    if use_amp and getattr(self, "scaler", None) is not None:
                        self.scaler.scale(forward_step.loss_total).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        forward_step.loss_total.backward()
                        self.optimizer.step()
                        
                    if self.scheduler is not None and getattr(self.scheduler, "by_step", False):
                        self.scheduler.step()
                    self.global_step += 1  # <-- train-only
                
                with torch.no_grad(): # Metrics computation step: 
                    batch_acc_map: Dict[str, float] = {}  # for tqdm display
                    for lvl in self.levels:

                        res = compute_rank_metrics(
                            which=self.rank_metrics,
                            logits_global=forward_step.logits_by_level[lvl],
                            labels_global=forward_step.labels_by_level[lvl],
                            mask=(mask_cache.get(lvl) if mask_cache is not None else None),
                            ignore_index=IGNORE_INDEX,
                        )
                        
                        # aggregate epoch sums 
                        for name, nd in res.items(): 
                            #nd contains numerator and denominator for scalar metrics
                            epoch_accums[lvl][name]["num"] += nd["num"]
                            epoch_accums[lvl][name]["den"] += nd["den"]

                        # show "accuracy" on the bar if present
                        if "accuracy" in res:
                            num, den = res["accuracy"]["num"], res["accuracy"]["den"]
                            batch_acc_map[lvl] = (num / den) if den > 0 else 0.0

                    # loss bookkeeping
                    loss_val = float(forward_step.loss_total.item())
                    loss_sum += loss_val

                    # compact tqdm postfix
                    acc_str  = _fmt_by_level("acc", batch_acc_map, self.levels)
                    loss_str = f"loss:tot={loss_val:.3f}"
                    pbar.set_postfix_str(_build_postfix(acc_str, loss_str,
                                                        lr=self.optimizer.param_groups[0]["lr"]))

                    # ---- batch logging buffer (write to disk once per epoch) ----
                    if is_train and (self.global_step % max(1, self.log_every) == 0):
                        # Keep it lean: loss, lr, plus per-level scalar metric(s)
                        metrics_by_level = {}
                        for lvl in self.levels:
                            metrics_by_level[lvl] = {}
                            for name, nd in epoch_accums[lvl].items():
                                # per-batch accuracy is available from batch_acc_map; extend similarly for other metrics if you compute them per batch
                                if name == "accuracy" and lvl in batch_acc_map:
                                    metrics_by_level[lvl][name] = batch_acc_map[lvl]
                        loss_by_level_scalar = {}

                        for lvl, t in forward_step.loss_by_level.items():
                            loss_by_level_scalar[lvl] = float(t.detach().item())

                        _batch_log_buffer.append({
                            "epoch": int(current_epoch),
                            "phase": phase,               # stays "train" here
                            "step": int(self.global_step),
                            "lr": float(self.optimizer.param_groups[0]["lr"]),
                            "loss_total": loss_val,
                            "loss_by_level": loss_by_level_scalar,   # <- new
                            "metrics_by_level": metrics_by_level,    # (still includes per-rank train accuracy if computed)
                        })

                n_batches += 1

            if is_train:
                # epoch wise scheduler step
                if self.scheduler is not None and not getattr(self.scheduler, "by_step", False):
                    self.scheduler.step()
        # ---- epoch roll-up ----
        metrics_by_level: Dict[str, Dict[str, float]] = {}
        for lvl in self.levels:
            metrics_by_level[lvl] = {}
            for name in self.rank_metrics:
                nd = epoch_accums[lvl][name]
                metrics_by_level[lvl][name] = (nd["num"] / nd["den"]) if nd["den"] > 0 else 0.0

        # Example: compute mean per metric across levels
        mean_by_metric: Dict[str, float] = {}
        for name in self.rank_metrics:
            vals = [metrics_by_level[lvl][name] for lvl in self.levels if epoch_accums[lvl][name]["den"] > 0]
            mean_by_metric[name] = (sum(vals) / len(vals)) if vals else 0.0
        
        loss_by_level_avg = {lvl: (loss_by_level_sums[lvl] / max(1, n_batches)) for lvl in self.levels}
        epoch_metrics = {
            "num_batches": n_batches,
            "metrics_by_level": metrics_by_level,
            "mean_by_metric": mean_by_metric,
            "loss": (loss_sum / max(1, n_batches)),
            "loss_by_level": loss_by_level_avg,   # <-- add this
        }
        return {"epoch": epoch_metrics, "batch_buffer": _batch_log_buffer}



    def train(self, max_epochs: int):
        for epoch in range(1, max_epochs + 1):
            # ---- train epoch ----
            train_pkg = self._run_phase(
                "train",
                dataloader=self.train_loader,
                mask_cache=self.mask_cache_train,   # training is masked
                current_epoch=epoch,
            )

            # ---- val epoch ----
            val_pkg = self._run_phase(
                "val",
                dataloader=self.val_loader,
                mask_cache=None,                    # validation is unmasked by design
                current_epoch=epoch,
            )

            # ---- selection metric (defaults: 'accuracy' at self.select_best_by level) ----
            sel_metric_name = "accuracy" if "accuracy" in self.rank_metrics else self.rank_metrics[0]
            sel_level = self.select_best_by if self.select_best_by in self.levels else self.levels[-1]
            sel_value = float(
                val_pkg["epoch"]["metrics_by_level"]
                    .get(sel_level, {})
                    .get(sel_metric_name, 0.0)
            )

            # ---- checkpoint + logs ----
            if self.ckpt:
                ep_path = os.path.join(self.ckpt, f"epoch_{epoch:03d}.pt")
                extra = {
                    "train": train_pkg["epoch"], 
                    "val":   val_pkg["epoch"],
                    "select_metric": {
                        "level": sel_level,
                        "name": sel_metric_name,
                        "value": sel_value,
                    },
                    **(self.static_metadata or {}),
                }
                _save_checkpoint(
                    path = ep_path,
                    model =  self.model,
                    optimizer = self.optimizer,
                    scheduler = self.scheduler,
                    epoch = epoch,
                    extra = extra,
                    global_step = self.global_step,
                    scaler=self.scaler, 
                    )
                # append epoch logs (two lines: train + val)
                self._append_jsonl(self.metrics_path, [
                    {"type": "epoch", "phase": "train", "epoch": epoch, **train_pkg["epoch"],
                    "lr": float(self.optimizer.param_groups[0]["lr"])},
                    {"type": "epoch", "phase": "val",   "epoch": epoch, **val_pkg["epoch"],
                    "lr": float(self.optimizer.param_groups[0]["lr"])},
                ])

                # append batch logs for this epoch in one go
                self._append_jsonl(self.batchlog_path, train_pkg.get("batch_buffer", []))
                
                self._update_history(epoch, train_pkg, val_pkg)
                try:
                    self._plot_batch_overview()
                    self._plot_epoch_overview()
                except Exception as e:
                    # plotting must never crash training
                    logger.info(f"[warn] plotting failed: {e}")

                # update best.json (no best.pt / last.pt)
                if sel_value > self.best_metric:
                    self.best_metric = sel_value
                    self.best_path = ep_path  # best is an epoch_###.pt
                    best_payload = {
                        "epoch": int(epoch),
                        "ckpt_file": os.path.basename(ep_path),
                        "ckpt_path": ep_path,
                        "selection": {
                            "level": sel_level,
                            "metric": sel_metric_name,
                            "value": float(sel_value),
                        },
                        "val_snapshot": val_pkg["epoch"],   # includes metrics_by_level & loss
                        "global_step": int(self.global_step),
                    }
                    with open(os.path.join(self.ckpt, "best.json"), "w") as jf:
                        json.dump(best_payload, jf, indent=2)

        return {"best_path": self.best_path, "best_metric": float(self.best_metric)}

class MLMTrainer:
    """
    Masked Language Modeling (MLM) trainer.

    Behavior:
      - AMP mixed precision (optional)
      - Scheduler is stepped once per optimizer update (per minibatch here)
      - Per-epoch checkpoints: epoch_###.pt, best.pt, last.pt
      - Resume support from last.pt (restores model/optim/sched/scaler/global_step/best_val)
      - JSONL metrics (step-level optional hook, epoch-level always if checkpoints_dir is set)

    Assumptions:
      - The dataset yields dicts with 'input_ids', 'attention_mask', 'labels'
      - The model returns either a Tensor [B,T,V] or a dict with key 'mlm_logits'
      - IGNORE_INDEX is used for positions that should not contribute to loss
    """

    def __init__(self,
                model,
                train_loader: DataLoader,
                val_loader: DataLoader,
                optimizer,
                scheduler=None,
                amp: bool = True,
                log_every: int = 100,
                checkpoints_dir: Optional[str] = None,
                ):
        self.model = model
        self.train_loader= train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)
        self.log_every = int(log_every)

        self.ckpt = checkpoints_dir
        if self.ckpt:
            os.makedirs(self.ckpt, exist_ok=True)
            self.metrics_path = os.path.join(self.ckpt, "metrics.jsonl")
        else:
            self.metrics_path = None

        self.device = next(self.model.parameters()).device
        self.global_step = 0
        self.best = math.inf
        self.best_path = None

        self._ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def _move(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: (v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()}

    def _step_loss(self, batch):
        out = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logits = extract_mlm_logits(out)
        loss = self._ce(logits.view(-1, logits.size(-1)),
                        batch["labels"].view(-1))
        return logits, loss

    def resume(self, last_path: str) -> int:
        if self.ckpt and os.path.exists(last_path):
            saved_epoch_1b, self.global_step, self.best = _load_checkpoint(
                path=last_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler if self.amp else None,
            )
            # start from the next epoch (1-based)
            return saved_epoch_1b + 1
        return 1

    def train(self, max_epochs: int, resume: bool = False):
        start_epoch = 1
        
        if resume and self.ckpt:
            start_epoch = self.resume(os.path.join(self.ckpt, "last.pt"))

        for epoch in range(start_epoch, max_epochs + 1):
            # ---- train
            self.model.train()
            train_loss_sum = 0.0
            train_items = 0
            t0 = time.time()

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{max_epochs} [tra]", leave=True)
            for i, b in enumerate(pbar, 1):
                b = self._move(b)
                self.optimizer.zero_grad(set_to_none=True)

                if self.amp:
                    with autocast():
                        logits, loss = self._step_loss(b)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer); self.scaler.update()
                else:
                    logits, loss = self._step_loss(b)
                    loss.backward(); self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                bs = b["input_ids"].size(0)
                train_loss_sum += float(loss.item()) * bs
                train_items += bs
                self.global_step += 1

                avg_loss = train_loss_sum / max(1, train_items)
                cur_lr   = self.optimizer.param_groups[0]["lr"]
                tok_acc  = token_accuracy(logits, b["labels"], IGNORE_INDEX)

                pbar.set_postfix({
                    "tra_acc":  f"{tok_acc:6.3f}",   # width=6, 3 decimals
                    "tra_loss": f"{avg_loss:6.4f}",  # width=6, 4 decimals
                    "lr":   f"{cur_lr:8.2e}",    # scientific, width=8
                })

                if self.metrics_path and (self.global_step % self.log_every == 0):
                    with open(self.metrics_path, "a") as f:
                        f.write(json.dumps({
                            "type": "step",
                            "epoch": epoch,
                            "step": self.global_step,
                            "lr": cur_lr,
                            "train_loss_mean": avg_loss,
                            "elapsed_s": time.time() - t0,
                        }) + "\n")

            # ---- validation
            self.model.eval()
            val_loss_sum, val_items, val_acc_sum = 0.0, 0, 0.0
            with torch.no_grad():
                pbar_val = tqdm(self.val_loader, desc=f"Epoch {epoch}/{max_epochs} [val]", leave=True)
                for b in pbar_val:
                    b = self._move(b)
                    logits, loss = self._step_loss(b)
                    val_loss_sum += float(loss.item()) * b["input_ids"].size(0)
                    val_items += b["input_ids"].size(0)
                    val_acc_sum += token_accuracy(logits, b["labels"], IGNORE_INDEX)

                    val_loss = val_loss_sum / max(1, val_items)
                    val_acc  = val_acc_sum / max(1, len(self.val_loader)) if len(self.val_loader) > 0 else 0.0
                    val_ppl  = math.exp(min(20.0, val_loss))

                    pbar_val.set_postfix({
                        "val_acc":  f"{val_acc:6.3f}",
                        "val_loss": f"{val_loss:6.4f}",
                        "ppl":  f"{val_ppl:8.2f}",
                    })
            
            # ---- epoch summary to logger ----
            train_loss  = train_loss_sum / max(1, train_items) 
            val_loss    = val_loss_sum   / max(1, val_items)
            val_acc     = val_acc_sum    / max(1, len(self.val_loader)) if len(self.val_loader) > 0 else 0.0
            val_ppl     = math.exp(min(20.0, val_loss))
            
            # ---- checkpoints
            if self.ckpt:
                ep_path = os.path.join(self.ckpt, f"epoch_{(epoch):03d}.pt")
                _save_checkpoint(
                    path = ep_path,
                    model = self.model,
                    optimizer = self.optimizer,
                    scheduler = self.scheduler,
                    epoch = epoch,
                    extra = {
                        "val_loss": val_loss,
                        "val_token_acc": val_acc,
                        "val_ppl": val_ppl
                        },
                    scaler=(self.scaler if self.amp else None),
                    global_step=self.global_step,
                    best_val=self.best)
                logger.info("[mlm] saved checkpoint %s", os.path.basename(ep_path))
                # epoch metrics
                try:
                    with open(self.metrics_path, "a") as f:
                        f.write(json.dumps({
                            "type": "epoch",
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "val_token_acc": val_acc,
                            "val_ppl": val_ppl,
                            "lr": self.optimizer.param_groups[0]["lr"],
                        }) + "\n")
                except Exception:
                    pass
                
                if val_loss < self.best:
                    logger.info(
                        "[mlm] new best @ epoch %03d: val_loss %.4f → %.4f (best.pt)",
                        epoch, self.best, val_loss
                    )
                    self.best = val_loss
                    self.best_path = ep_path
                    with open(os.path.join(self.ckpt, "best.pt"), "wb") as g, open(ep_path, "rb") as s:
                        g.write(s.read())
                with open(os.path.join(self.ckpt, "last.pt"), "wb") as g, open(ep_path, "rb") as s:
                    g.write(s.read())
            
            # ---- epoch summary to logger (always) ----
            logger.info(
                "[mlm] epoch %03d/%03d  train_loss=%.4f  val_loss=%.4f  val_acc=%.3f  ppl=%.2f  lr=%.2e  step=%d",
                epoch, max_epochs, train_loss, val_loss, val_acc, val_ppl,
                self.optimizer.param_groups[0]["lr"], self.global_step
            )

        return {"best_path": self.best_path, "best_val_loss": self.best}
    
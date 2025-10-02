#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finetuning entry point for hierarchical (or single-rank) taxonomy classification.

Responsibilities (entry-point scope only):
- Parse config & CLI, apply profile overrides
- Resolve artifact paths and create directories
- Build vocab & preprocessors (and derive max_position_embeddings)
- Load label space, pick levels, compute class sizes
- Build encoder & load a pretrained backbone (tolerant mapping), then wrap with the classifier head
- Build datasets/dataloaders
- Build optimizer & scheduler
- Log robust pre-flight diagnostics (env, config, data, model, backbone mapping, scheduler preview)
- Run training via ClassificationTrainer
- Emit a compact post-training summary JSON

Assumptions:
- The project modules are importable (installed or on PYTHONPATH).
- ClassifyDataset yields {"input_ids", "attention_mask", "labels" {lvl: int}, "species_idx"}.
- ClassificationTrainer expects labels under "labels_by_level" → we adapt with a tiny Dataset wrapper.
"""

from __future__ import annotations

#################################################################
##### Chunk 0: Imports, typing, small utilities #################
#################################################################
import argparse
import datetime as _dt
import json
import math
import os
import random
import re
import logging
import shutil
from collections import Counter, defaultdict
import json
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from transformers import BertConfig, BertModel
import pandas as pd
import torch
from torch.utils.data import DataLoader

try:
    import yaml
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyYAML is required: pip install pyyaml") from e

# Project modules
from taxonml.core.config import build_profile_paths
from taxonml.preprocessing.vocab import Vocabulary, KmerVocabConstructor
from taxonml.preprocessing import augmentation, tokenization, padding, truncation
from taxonml.preprocessing.preprocessor import Preprocessor
from taxonml.data.datasets import ClassifyDataset
from taxonml.labels.space import LabelSpace
from taxonml.labels.encoder import LabelEncoder
from taxonml.encoders.bert import load_pretrained_backbone, derive_arch_id_from_cfg
from taxonml.models.taxonomy_model import TaxonomyModel
from taxonml.training.trainers import ClassificationTrainer
from taxonml.training.schedulers import build_scheduler_unified
from taxonml.data.samplers import SpeciesBalancedSampler

def setup_logging(console_level: int = logging.INFO) -> None:
    # Root config
    logging.captureWarnings(True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

#################################################################
##### Chunk 1: Config loading & path resolution #################
#################################################################
def derive_arch_id(cfg: Dict[str, Any]) -> str:
    enc = cfg["model"]["encoder"]
    return f"bert_h{enc['hidden_size']}_L{enc['num_hidden_layers']}_H{enc['num_attention_heads']}"


def validate_datasets_and_log(
    *,
    train_dataset,
    val_dataset,
    levels: List[str],
    class_sizes: Dict[str, int],
    optimal_length: int,           # tokens without CLS/SEP
    logger,
    batch_probe_size: int = 8,
    sample_size_for_stats: int = 10_000,
    warn_unk_rate: float = 0.05,   # warn if UNK > 5% in sampled tokens
) -> None:
    """
    Quick, cheap dataset validation + diagnostics. Raises on structural errors,
    logs warnings on suspicious-but-not-fatal conditions.

    Assumptions:
      - Each __getitem__ returns:
        {
          "input_ids": LongTensor[T],
          "attention_mask": LongTensor[T],
          "labels_by_level": {lvl: LongTensor([]) scalar},
          ...
        }
      - PAD → attention_mask==0; non-PAD → 1
      - Dataset object exposes .prep.vocab (optional; used to check PAD/UNK IDs precisely)
    """
    logger.info("=== Dataset validation ===")
    logger.info("Train size=%d | Val size=%d", len(train_dataset), len(val_dataset))

    # ---------- 1) Post-preproc token length (with CLS/SEP) ----------
    def _pick_indices(n: int) -> List[int]:
        if n <= 0:
            return []
        return [0, max(0, n//2), max(0, n-1)] if n >= 3 else list(range(n))

    def _length_checks(ds, name: str) -> List[int]:
        idxs = _pick_indices(len(ds))
        lens = []
        for i in idxs:
            item = ds[i]
            ids = item["input_ids"]
            lens.append(int(ids.numel()))
        logger.info("%s sample token lengths (incl. CLS/SEP): %s", name, lens)
        expected = optimal_length + 2
        if not all(l == expected for l in lens):
            raise ValueError(f"{name}: expected all lengths == {expected}, got {lens}")
        return lens

    _length_checks(train_dataset, "train")
    _length_checks(val_dataset,   "val")

    # ---------- 2) Label integrity ----------
    def _label_integrity(ds, name: str) -> None:
        item = ds[0]
        if "labels_by_level" not in item:
            raise KeyError(f"{name}: missing 'labels_by_level' in dataset item")
        lbls = item["labels_by_level"]
        if set(lbls.keys()) != set(levels):
            raise KeyError(f"{name}: labels_by_level keys {list(lbls.keys())} != levels {levels}")
        for lvl, t in lbls.items():
            if not (torch.is_tensor(t) and t.dtype == torch.long and t.dim() == 0):
                raise TypeError(f"{name}: {lvl} label must be scalar LongTensor; got {type(t)} shape={getattr(t, 'shape', None)}")
            ci = int(t.item())
            C = class_sizes[lvl]
            if not (0 <= ci < C):
                raise ValueError(f"{name}: {lvl} label {ci} out of range [0,{C})")

    _label_integrity(train_dataset, "train")
    _label_integrity(val_dataset,   "val")

    # ---------- 3) UNK/PAD rate + mask consistency (sampled) ----------
    def _vocab_ids(ds) -> Tuple[Optional[int], Optional[int]]:
        v = getattr(getattr(ds, "prep", None), "vocab", None)
        if v is None:
            return None, None
        try:
            return v.unk_id(), v.pad_id()
        except Exception:
            return None, None

    unk_id_tr, pad_id_tr = _vocab_ids(train_dataset)

    def _sample_stats(ds, name: str) -> None:
        # sample first N (bounded)
        N = min(len(ds), sample_size_for_stats)
        if N == 0:
            logger.warning("%s: empty dataset; skipping stats", name)
            return
        unk_cnt = 0
        pad_cnt = 0
        tok_cnt = 0

        # consistency check: attention_mask aligns with PAD positions (if pad_id known)
        check_mask = pad_id_tr is not None

        for i in range(N):
            item = ds[i]
            ids = item["input_ids"]
            am  = item["attention_mask"]
            if check_mask:
                pad_positions = (ids == pad_id_tr)
                # mask must be 0 where PAD, and 1 otherwise
                if not torch.equal((am == 0), pad_positions):
                    raise ValueError(f"{name}: attention_mask does not match PAD positions at sample {i}")
            if unk_id_tr is not None:
                unk_cnt += int((ids == unk_id_tr).sum().item())
            pad_cnt += int((am == 0).sum().item())
            tok_cnt += int(ids.numel())

        unk_rate = (unk_cnt / max(1, tok_cnt)) if unk_id_tr is not None else None
        pad_rate = pad_cnt / max(1, tok_cnt)
        if unk_rate is not None:
            logger.info("%s UNK rate (sampled): %.2f%%", name, 100.0 * unk_rate)
            if unk_rate > warn_unk_rate:
                logger.warning("%s: High UNK rate (%.2f%%) > %.2f%% threshold. Check vocab/alphabet/cleaning.",
                               name, 100.0 * unk_rate, 100.0 * warn_unk_rate)
        else:
            logger.info("%s UNK rate (sampled): n/a (no vocab or unknown ID)", name)
        logger.info("%s PAD rate (sampled): %.2f%%", name, 100.0 * pad_rate)

    _sample_stats(train_dataset, "train")
    _sample_stats(val_dataset,   "val")

    # ---------- 4) Class balance snapshot (sampled) ----------
    def _top_counts(ds, lvl: str, k: int = 5, N: int = 10_000) -> Dict[int, int]:
        n = min(len(ds), N)
        c = Counter(int(ds[i]["labels_by_level"][lvl]) for i in range(n))
        return dict(c.most_common(k))

    for lvl in levels:
        top_tr = _top_counts(train_dataset, lvl)
        top_va = _top_counts(val_dataset,   lvl)
        logger.info("Top-5 train counts for %s (sampled): %s", lvl, json.dumps(top_tr))
        logger.info("Top-5   val counts for %s (sampled): %s", lvl, json.dumps(top_va))

    # ---------- 5) One mini-batch probe ----------
    def _batch_probe(ds, name: str) -> None:
        if len(ds) == 0:
            logger.warning("%s: empty; skipping batch probe", name)
            return
        loader = DataLoader(ds, batch_size=min(batch_probe_size, len(ds)), shuffle=False, num_workers=0)
        b = next(iter(loader))
        logger.info("%s batch shapes: input_ids=%s, attention_mask=%s",
                    name, tuple(b["input_ids"].shape), tuple(b["attention_mask"].shape))
        for lvl in levels:
            logger.info("%s batch labels_by_level[%s].shape=%s",
                        name, lvl, tuple(b["labels_by_level"][lvl].shape))
        # light sanity: mask is 0/1
        am = b["attention_mask"]
        if not torch.all((am == 0) | (am == 1)):
            raise ValueError(f"{name}: attention_mask must be binary 0/1")

    _batch_probe(train_dataset, "train")
    _batch_probe(val_dataset,   "val")

    logger.info("Dataset validation: OK")

def group_params_yaml(model, indent: int = 2) -> str:
    def tree():
        return defaultdict(tree)
    root = tree()

    def insert(path, group, name):
        node = root
        for p in path:
            node = node[p]
        if group not in node:
            node[group] = []
        node[group].append(name)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        parts = name.split(".")
        group = "no_decay" if any(nd in name for nd in ["bias", "LayerNorm.weight"]) else "decay"
        insert(parts[:-1], group, parts[-1])

    def dump(node, level=0):
        lines = []
        pad = " " * (indent * level)
        for key, child in node.items():
            if isinstance(child, list):
                # terminal: list of param names
                lines.append(f"{pad}{key}: [{', '.join(child)}]")
            else:
                lines.append(f"{pad}{key}:")
                lines.extend(dump(child, level + 1))
        return lines

    return "\n".join(dump(root))



def resolve_paths(cfg: Dict[str, Any]) -> Dict[str, str]:
    """Render templates under artifacts.* with base substitutions and ensure dirs exist."""
    def _expand(s: str) -> str:
        return os.path.expanduser(os.path.expandvars(s or ""))

    art = cfg["artifacts"]
    roots = art.get("roots", {})
    exp_root = _expand(roots.get("experiments", cfg.get("data", {}).get("experiments_root", "")))
    pre_root = _expand(roots.get("pretrained", ""))

    base = {
        "experiments": exp_root,
        "pretrained": pre_root,
        "experiment": cfg["experiment"]["name"],
        "task": cfg["runtime"].get("task", "finetune"),
        "fold": int(cfg["runtime"].get("fold_index", 1)),
        "prepared_filename": cfg.get("data", {}).get("filenames", {}).get("prepared", "prepared_dataset.csv"),
        "debug_filename": cfg.get("data", {}).get("filenames", {}).get("debug", "debug.csv"),
        "label_space_filename": cfg.get("data", {}).get("filenames", {}).get("label_space", "label_space.json"),
        "prep_summary_filename": cfg.get("data", {}).get("filenames", {}).get("prep_summary", "prep_summary.json"),
    }

    ns = art["experiments"]
    out: Dict[str, str] = {}
    if "run_dir" in ns:
        out["run_dir"] = ns["run_dir"].format(**base)

    for k, tpl in ns.items():
        if k in out:
            continue
        out[k] = tpl.format(**base)

    # mkdirs
    for key in ("run_dir", "checkpoints_dir", "logs_dir"):
        if key in out:
            Path(out[key]).mkdir(parents=True, exist_ok=True)

    out["pretrained_root"] = pre_root
    return out


#################################################################
##### Chunk 2: CLI & level selection ############################
#################################################################
from taxonml.core.constants import CANON_LEVELS, IGNORE_INDEX

def parse_levels(levels_arg: str) -> list[str]:
    """
    Parse CLI argument string for taxonomy levels into a clean list.

    - Splits on commas.
    - Strips whitespace and lowercases.
    - Expands 'all' into CANON_LEVELS.
    - Rejects duplicates and mixed 'all' with other entries.
    - Does not validate canonical order or membership in CANON_LEVELS
      (handled later by LabelSpace).
    """
    if not levels_arg or not levels_arg.strip():
        raise ValueError("--levels cannot be empty.")

    # split and normalize
    entries = [part.strip().lower() for part in levels_arg.split(",") if part.strip()]

    if not entries:
        raise ValueError("--levels produced no valid entries.")

    # handle 'all'
    if len(entries) == 1 and entries[0] == "all":
        return list(CANON_LEVELS)
    if "all" in entries:
        raise ValueError("'all' cannot be combined with other levels.")

    # check duplicates
    if len(entries) != len(set(entries)):
        raise ValueError(f"Duplicate levels not allowed: {entries}")

    return entries

def _load_fold_masks(masks_path: str, fold_id: int, levels: list[str],
                     class_sizes: dict[str, int], logger) -> tuple[dict, dict]:
    with open(masks_path, "r") as f:
        payload = json.load(f)

    # Basic structure checks
    folds = payload.get("folds", {})
    if str(fold_id) not in folds:
        raise KeyError(f"Fold '{fold_id}' not found in masks file: {masks_path}")

    masks_train, masks_val = {}, {}
    for lvl in levels:
        entry_train = folds[str(fold_id)]["train"].get(lvl)
        entry_val   = folds[str(fold_id)]["val"].get(lvl)
        if entry_train is None or entry_val is None:
            raise KeyError(f"Level '{lvl}' missing in masks for fold {fold_id}.")

        mt = entry_train["mask"]
        mv = entry_val["mask"]

        # Sanity vs class_sizes
        C = class_sizes[lvl]
        if len(mt) != C or len(mv) != C:
            raise ValueError(
                f"Mask length mismatch for level '{lvl}': "
                f"train={len(mt)}, val={len(mv)}, expected={C}"
            )

        masks_train[lvl] = mt
        masks_val[lvl]   = mv

    logger.info("Loaded fold masks: %s | levels=%s", Path(masks_path).name, levels)
    return masks_train, masks_val

# # ---- Phase A (no arch_id needed): data/prep paths ----
# def build_prep_paths(experiments_root: str | Path, experiment_name: str) -> Dict[str, Path]:
#     exp_root = Path(experiments_root) / experiment_name
#     data_dir = exp_root / "data"
#     return {
#         "exp_root": exp_root,
#         "data_dir": data_dir,
#         "prepared_csv": data_dir / "prepared_dataset.csv",
#         "debug_csv": data_dir / "debug.csv",
#         "label_space_json": data_dir / "label_space.json",
#         "fold_masks_exp1": data_dir / "fold_masks_exp1.json",
#         "fold_masks_exp2": data_dir / "fold_masks_exp2.json",
#         "prep_summary_json": data_dir / "prep_summary.json",
#         "logs_dir": exp_root / "logs",
#     }
# from pathlib import Path




# ---- Phase B (needs arch_id + fold + levels): training/run paths ----
def ranks_code(levels: List[str]) -> str:
    """e.g. ['phylum','class','order','family','genus','species'] -> 'ranks_6_pcofgs'"""
    abbrev = "".join(l.strip().lower()[0] for l in levels)
    return f"ranks_{len(levels)}_{abbrev}"

def build_run_paths(
    experiments_root: str | Path,
    experiment_name: str,
    arch_id: str,
    fold: int,
    levels: List[str],
    profile: str,
) -> Dict[str, Path]:
    exp_root = Path(experiments_root) / experiment_name
    arch_root = exp_root / arch_id / profile
    task = ranks_code(levels)                    # <<< was 'single_x' or 'hierarchical'
    fold_dir = arch_root / "folds" / f"fold_{fold:02d}"
    run_dir  = fold_dir / task
    return {
        "arch_root": arch_root,
        "fold_dir": fold_dir,
        "run_dir": run_dir,
        "checkpoints_dir": run_dir / "checkpoints",
        "history_file": run_dir / "history.json",
        "results_file": run_dir / "results.json",
    }

def clear_run_dir(run_dir: Path, logger) -> None:
    if run_dir.exists():
        logger.warning("Clearing existing run directory: %s", run_dir)
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

### DEBUG PROBERS: 
def mask_coverage_report(train_dataset, levels, ls, mask_cache, trainer):
    print("\n[CHECK] mask coverage on debug-train data")
    for lvl in levels:
        l2i = ls.label_to_idx[lvl]
        # collect encoded label ids from the dataset (global ids in LabelSpace)
        ids = []
        for i in range(len(train_dataset)):
            ids.append(int(train_dataset[i]["labels_by_level"][lvl]))
        ids = torch.tensor(ids, device=trainer.device)

        active_idx, remap = mask_cache[lvl]
        valid = (remap[ids] != IGNORE_INDEX)
        print(f"{lvl:7s} | C_full={remap.numel():>6} | C_active={active_idx.numel():>6} "
            f"| n_items={len(ids):>6} | valid%={valid.float().mean().item():.4f}")


_ARCH_DIR_RE = re.compile(r"^bert_h\d+_L\d+_H\d+_i\d+_P\d+$")
#################################################################
##### Chunk 5: Main #############################################
#################################################################
def main() -> None:
    # -------- CLI
    #ap = argparse.ArgumentParser(description="Finetune hierarchical/single-rank taxonomy classifier")
    #ap.add_argument("--config", required=True, help="Path to master experiment YAML")
    #ap.add_argument("--levels", required=True, help="Comma-separated levels (e.g., 'species' or 'phylum,class,...'). Or all for full hierarchy.")
    #ap.add_argumetn("--fold", required=True, help"k-Fold index to train over, in [1,2,...,10])
    #ap.add_argument("--pretrained", help="Path to pretrained backbone (overrides derived arch path)")    
    #ap.add_argument("--resume", action="store_true", help="(Reserved) Resume from last.pt if supported by trainer")
    #args = ap.parse_args()
    logging_level = logging.INFO
    setup_logging(logging_level)
    logger = logging.getLogger(__name__)
    logger.info("=== Startup ===")

    # ===== Stage 0: runtime inputs (raw knobs; config-sourced later) =====
    #     
    # CLI mock OR global variables
    levels_input = "all"  # CLI
    fold = 7             # CLI
    debug = False # CLI
    # raise "Reconsider max learning rate - you raised it ecently, right?"
    
    PRETRAINED_MODEL_PATH = "/mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/pretrained_models/bert_h512_L10_H8_i2048_P502/best.pt"  # CLI 
    EXPERIMENTS_ROOT = "/mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/experiments"  # GLOBAL?
    
    # # # Config mock
    EXPERIMENT_NAME  = "sequence_fold_full"
    seed = 42
    k = 3
    max_bases = 1500
    modification_probability = 0.01
    alphabet = ["A","C","T","G"]

    hidden_size=512
    num_hidden_layers=10
    num_attention_heads=8
    intermediate_size=2048
    hidden_dropout_prob=0.1
    attention_probs_dropout_prob=0.1

    hierarchical_dropout=0.1
    bottleneck = 256

    weight_decay = 0.1
    base_lr = 0.0001
    batch_size   = 128
    num_workers  = 4
    pin_memory   = True
    drop_last_tr = True
    max_epochs = 10

    # Scheduler knobs
    common_floor  = {"min_factor": 1e-2}
    common_warmup = {"type": "linear", "duration": 1}
    SCHED_KIND = "tri"
    SCHED_BASE = "step"

    # === Profile switch (subset mode) ===
    if debug:
        max_bases = 600
        hidden_size=64
        num_hidden_layers=2
        num_attention_heads=2
        intermediate_size=256
        PRETRAINED_MODEL_PATH = "/mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/pretrained_models/bert_h64_L2_H2_i256_P202/best_clean.pt"

    profile = "debug" if debug else "full"
    logger.info("Active profile: %s", profile)

    schedule_by_kind = {
        "tri": {
            "base": SCHED_BASE,
            "warmup": common_warmup,
            "main": {"type": "tri", "plateau": 3, "decay": 8},
            "floor": common_floor,
        },
        "cosine": {
            "base": SCHED_BASE,
            "warmup": common_warmup,
            "main": {"type": "cosine", "epochs": 10},
            "floor": common_floor,
        },
        "cosine_restarts": {
            "base": SCHED_BASE,
            "warmup": common_warmup,
            "main": {"type": "cosine_restarts", "cycle_epochs": 5, "num_cycles": 3, "t_mult": 1.0, "peak_decay": 0.8},
            "floor": common_floor,
        },
    }

    use_amp = True
    log_every = 5

    # === NEW: profile-aware artifact resolution ===
    art = build_profile_paths(EXPERIMENTS_ROOT, EXPERIMENT_NAME, profile)
    DATA_PATH        = art["prepared_csv"]
    LABEL_SPACE_PATH = art["label_space_json"]

    # Choose which fold scheme to use
    FOLD_SCHEME = "exp1"   # or "exp2"
    FOLD_MASKS_PATH = art["fold_masks_exp1"] if FOLD_SCHEME == "exp1" else art["fold_masks_exp2"]

    logger.info("Artifacts (profile=%s):", profile)
    logger.info("  prepared_csv     = %s", DATA_PATH)
    logger.info("  label_space_json = %s", LABEL_SPACE_PATH)
    logger.info("  fold_masks_json  = %s", FOLD_MASKS_PATH)

    # ===== Stage 1: seeds & device =====
    set_all_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ===== Stage 2: label space =====
    levels = parse_levels(levels_input)

    art["logs_dir"].mkdir(parents=True, exist_ok=True)
    ls = LabelSpace.from_json(LABEL_SPACE_PATH)
    ls.validate_levels(levels)

    logger.info("=== Label Space ===")
    class_sizes = {lvl: ls.num_classes(lvl) for lvl in levels}
    logger.info("Number of classes for active ranks: \n%s", json.dumps(class_sizes, indent=2, ensure_ascii=False))

    # ===== Stage 3: preprocessing =====
    logger.info("=== Vocab ===")
    vocab = Vocabulary()
    vocab.build_from_constructor(KmerVocabConstructor(k=k, alphabet=alphabet))
    logger.info("Vocabulary:\n%s", vocab)

    assert max_bases % k == 0, "max_bases must be divisible by k"
    optimal_length = max_bases // k
    
    logger.info("=== Preprocesessors ===")

    tok   = tokenization.KmerStrategy(k=k, padding_alphabet=alphabet)
    pad   = padding.PaddingEndStrategy(optimal_length=optimal_length)
    trunc = truncation.SlidingWindowTruncationStrategy(optimal_length=optimal_length)

    modifier  = augmentation.SequenceModifier(alphabet)
    aug_train = augmentation.BaseStrategy(modifier=modifier,
                                          alphabet=alphabet,
                                          modification_probability=modification_probability)
    aug_eval  = augmentation.IdentityStrategy()

    preproc_train = Preprocessor(aug_train, tok, pad, trunc, vocab)
    preproc_val   = Preprocessor(aug_eval,  tok, pad, trunc, vocab)
    logger.info("Training preprocessor: \n%r", preproc_train)
    logger.info("Validation preprocessor: \n%r",   preproc_val)

    # ===== Stage 4: encoder =====
    logger.info("=== Encoder ===")
    bert_config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        max_position_embeddings=optimal_length + 2,
    )
    arch_id = derive_arch_id_from_cfg(bert_config)
    encoder = BertModel(bert_config)
    logger.info("arch_id=%s", arch_id)
    logger.debug("Encoder repr:\n%s", repr(encoder))  # keep detailed tree at DEBUG

    # ===== Stage 5: load pretrained encoder =====
    pretrained_path = Path(PRETRAINED_MODEL_PATH).expanduser().resolve()
    ckpt = torch.load(pretrained_path, map_location="cpu")
    #print("Checkpoint keys (first 20):", list(ckpt["model"].keys())[:20])
    #print("Encoder keys (first 20):", list(encoder.state_dict().keys())[:20])
    loaded_keys = load_pretrained_backbone(encoder, pretrained_path, strict=True)
    #logger.info("Loaded tensors: %d from %s", len(loaded_keys), pretrained_path)
    logger.info(encoder)
    
    # ===== Stage 6: Setup model with hierarchical head =====
    head_cfg = dict(
    hierarchical_dropout=hierarchical_dropout,
    bottleneck=bottleneck,
    )
    model = TaxonomyModel.for_classify(
        encoder=encoder,
        levels=levels,
        class_sizes=class_sizes,
        **head_cfg
    )
    logger.info(model)
    model.to(device)
    
    
    # -------- Data split
    df = pd.read_csv(DATA_PATH)
    assert {"sequence", "fold_exp1", *levels}.issubset(df.columns), \
        "CSV must contain 'sequence', 'fold_exp1', and all requested levels"
    df_train = df[df["fold_exp1"] != fold].reset_index(drop=True)
    df_val = df[df["fold_exp1"] == fold].reset_index(drop=True)
    logger.info("len(df_train): %r", len(df_train))
    logger.info("head: %r", df_train.head())
    logger.info("len(df_val): %r", len(df_val))
    logger.info("head: %r", df_val.head())
    # summarize_dataframe(df_train, df_val, fold)
    

    # -------- Build datasets & loaders
    label_enc = LabelEncoder(space=ls, unknown="error")
    train_dataset = ClassifyDataset(df_train, preproc_train, label_enc, levels)
    val_dataset = ClassifyDataset(df_val, preproc_val, label_enc, levels)
    
    # Out commented but valid:
    # validate_datasets_and_log(
    #     train_dataset=train_dataset,
    #     val_dataset=val_dataset,
    #     levels=levels,
    #     class_sizes=class_sizes,
    #     optimal_length=optimal_length,
    #     logger=logger,
    #     sample_size_for_stats=min(2048, len(train_dataset))
    # )
    logger.info(f"Train set: {train_dataset.__repr__()}")
    logger.info(f"Val set: {val_dataset.__repr__()}")


    logger.info("=== Dataloaders ===")
    persistent_workers = (num_workers > 0)
    train_sampler = SpeciesBalancedSampler.from_dataset(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_tr,
        persistent_workers=persistent_workers,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    logger.info("Dataloaders set up complete")
    
    
    # steps_per_epoch = math.ceil(len(train_dataset) / max(1, batch_size))
    steps_per_epoch = (
        len(train_sampler) // batch_size if drop_last_tr
         else math.ceil(len(train_sampler) / batch_size)
        )
    
    # -------- Optimizer & scheduler
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue 
        if any(nd in name for nd in ["bias", "LayerNorm.weight"]):
             no_decay_params.append(param) 
        else: decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}],
        lr=base_lr, betas=(0.9, 0.999), eps=1e-8
    )
    
    
    logger.info("=== Optimizer ===")
    logger.info(f"AdamW: lr={base_lr:.3e}, weight_decay={weight_decay}.")
    logger.info("Param groups (YAML): \n%s", group_params_yaml(model))
    
    schedule = schedule_by_kind[SCHED_KIND]
    logger.info("Optimzier schedule: \n%r", schedule)
    scheduler = build_scheduler_unified(optimizer, steps_per_epoch, schedule)
    
    masks_train, masks_val = _load_fold_masks(FOLD_MASKS_PATH, fold, levels, class_sizes, logger)
    
    run_paths = build_run_paths(EXPERIMENTS_ROOT, EXPERIMENT_NAME, arch_id, fold, levels, profile)  # <-- pass profile
    
    # nuke run dir
    clear_run_dir(run_paths["run_dir"], logger)
    logger.info("Run dirs: arch_root=%s | fold_dir=%s | run_dir=%s",
            run_paths["arch_root"], run_paths["fold_dir"], run_paths["run_dir"])
    
    CHECKPOINTS_DIR = run_paths["checkpoints_dir"]
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    trainer = ClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        levels=levels,
        masks_train=masks_train,
        masks_val=masks_val,
        optimizer=optimizer,
        scheduler=scheduler,
        amp=use_amp,
        log_every=log_every,
        checkpoints_dir=CHECKPOINTS_DIR,
        select_best_by="species" if "species" in levels else levels[-1],
    )
    # mask_coverage_report(train_dataset, levels, ls, trainer.mask_cache_train, trainer)


    logging.info("=== Training: start ===")
    # # Note:  The code actually compiles without issue this far.
    # summary = trainer...

    # Smoke
    logger.info("SMOKE: Trainer")
    trainer.train(
        max_epochs = max_epochs
        )
    logging.info("=== Training: done ===")


if __name__ == "__main__":
    main()
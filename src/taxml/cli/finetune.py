#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

# Standard library
import argparse
import datetime as _dt
import json
import logging
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Third-party
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel

try:
    import yaml
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyYAML is required: pip install pyyaml") from e

# Project modules
from taxml.core.config import load_config
from taxml.core.logging import setup_logging, attach_file_logger
from taxml.core.randomness import set_all_seeds
from taxml.data.datasets import ClassifyDataset
from taxml.data.samplers import SpeciesBalancedSampler, SpeciesPowerSampler
from taxml.encoders.bert import load_pretrained_backbone
from taxml.labels.encoder import LabelEncoder
from taxml.labels.space import LabelSpace
from taxml.models.taxonomy_model import TaxonomyModel
from taxml.preprocessing import augmentation, tokenization, padding, truncation
from taxml.preprocessing.preprocessor import Preprocessor
from taxml.preprocessing.vocab import Vocabulary, KmerVocabConstructor
from taxml.training.schedulers import build_scheduler_unified
from taxml.training.trainers import ClassificationTrainer
from taxml.metrics.rank import register_accuracy_by_class_size_bins

# Logging setup
logger = logging.getLogger(__name__)



#################################################################
##### Chunk 1: Config loading & path resolution #################
#################################################################

def validate_datasets_and_log(
    *,
    train_dataset,
    val_dataset,
    levels: List[str],
    num_classes_by_rank: Dict[str, int],
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
            C = num_classes_by_rank[lvl]
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



#################################################################
##### Chunk 2: CLI & level selection ############################
#################################################################
from taxml.core.constants import CANON_LEVELS, IGNORE_INDEX

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
                     num_classes_by_rank: dict[str, int], logger) -> tuple[dict, dict]:
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

        # Sanity vs num_classes_by_rank
        C = num_classes_by_rank[lvl]
        if len(mt) != C or len(mv) != C:
            raise ValueError(
                f"Mask length mismatch for level '{lvl}': "
                f"train={len(mt)}, val={len(mv)}, expected={C}"
            )

        masks_train[lvl] = mt
        masks_val[lvl]   = mv

    logger.info("Loaded fold masks: %s | levels=%s", Path(masks_path).name, levels)
    return masks_train, masks_val


def ranks_code(levels: List[str]) -> str:
    """e.g. ['phylum','class','order','family','genus','species'] -> 'ranks_6_pcofgs'"""
    abbrev = "".join(l.strip().lower()[0] for l in levels)
    return f"ranks_{len(levels)}_{abbrev}"

def clear_run_dir(run_dir: Path, logger) -> None:
    if run_dir.exists():
        logger.warning("Clearing existing run directory: %s", run_dir)
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

def main() -> None:
    # -------- CLI
    ap = argparse.ArgumentParser(description="Finetune hierarchical/single-rank taxonomy classifier")
    ap.add_argument("--config", required=True, help="Path to master experiment YAML")
    ap.add_argument("--levels", required=True, help="Comma-separated levels, or 'all'")
    ap.add_argument("--fold", type=int, required=True, help="Fold index in [1..k]")
    ap.add_argument("--scheme", choices=["exp1","exp2"], default=None, help="Fold scheme (optional; defaults to exp1)")
    ap.add_argument("--debug", action="store_true", help="Apply debug_overrides")
    ap.add_argument("--sampler_alpha", type=float, default=None,
                    help="Exponent for 1/(class_size^alpha) train sampler; overrides YAML if set.")
    args = ap.parse_args()
    setup_logging(
        console_level=logging.INFO,
        buffer_early=True
        )
    logger.info("=== Startup ===")

    # Resolve levels
    levels_req = parse_levels(args.levels)

    # Load effective config
    cfg = load_config(
        mode = "finetune",
        config_path = args.config,
        debug = args.debug,
        levels = levels_req,
        fold_index = args.fold,
        fold_scheme = args.scheme,
    )

    paths = cfg["paths_active"]
    task  = cfg["finetune"]
    prep  = cfg["preprocessing"]
    enc   = cfg["model"]["encoder"]
    arch_id  = cfg["arch"]["id"]
    profile = "debug" if cfg["runtime"]["debug"] else "full"
    
    use_gate = True
    print(f"use_gate = {use_gate}")
    logger.info("=== Runtime summary ===" )
    logger.info("experiment=%s | runtime=%s | profile=%s | arch_id=%s",
                cfg["experiment"]["name"], cfg["runtime"], profile, arch_id)
    logger.info("active data: prepared_csv=%s | label_space_json=%s | fold_masks=%s",
                paths["data"]["prepared_csv"], paths["data"]["label_space_json"], paths["data"]["fold_masks"])

    # ===== Stage 1: seeds & device =====
    set_all_seeds(cfg["experiment"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ===== Stage 2: label space =====
    levels = list(task["levels"])
    paths["logs_dir"].mkdir(parents=True, exist_ok=True)
    ls = LabelSpace.from_json( paths["data"]["label_space_json"])
    ls.validate_levels(levels)
    
    num_classes_by_rank = {lvl: ls.num_classes(lvl) for lvl in levels}
    logger.info("Number of classes for active ranks: \n%s", json.dumps(num_classes_by_rank, indent=2, ensure_ascii=False))

    # ===== Stage 3: preprocessing =====
    logger.info("=== Vocab ===")
    vocab = Vocabulary()
    vocab.build_from_constructor(KmerVocabConstructor(k=prep["k"], alphabet=prep["alphabet"]))
    logger.info("Vocabulary:\n%s", vocab)

    logger.info("=== Preprocesessors ===")

    tok   = tokenization.KmerStrategy(k = prep["k"], alphabet = prep["alphabet"])
    pad   = padding.PaddingEndStrategy(optimal_length = prep["optimal_length"])
    trunc = truncation.SlidingWindowTruncationStrategy(optimal_length = prep["optimal_length"])

    modifier  = augmentation.SequenceModifier(alphabet = prep["alphabet"])
    aug_train = augmentation.BaseStrategy(
        modifier = modifier,
        alphabet = prep["alphabet"],
        modification_probability = prep["mod_prob"]
    )
    aug_eval  = augmentation.IdentityStrategy()

    preproc_train = Preprocessor(aug_train, tok, pad, trunc, vocab)
    preproc_val   = Preprocessor(aug_eval,  tok, pad, trunc, vocab)
    logger.info("Training preprocessor: \n%r", preproc_train)
    logger.info("Validation preprocessor: \n%r",   preproc_val)

    # ===== Stage 4: encoder =====
    logger.info("=== Encoder ===")
    bert_config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=enc["hidden_size"],
        num_hidden_layers=enc["num_hidden_layers"],
        num_attention_heads=enc["num_attention_heads"],
        intermediate_size=enc["intermediate_size"],
        hidden_dropout_prob=enc["hidden_dropout_prob"],
        attention_probs_dropout_prob=enc["attention_probs_dropout_prob"],
        max_position_embeddings=prep["max_position_embeddings"],
    )
    encoder = BertModel(bert_config)
    logger.info("arch_id=%s", arch_id)
    logger.debug("Encoder repr:\n%s", repr(encoder))  # keep detailed tree at DEBUG

    # ===== Stage 5: load pretrained encoder =====
    pretrained_path = Path(task["backbone"]["path"]).expanduser().resolve()
    ckpt = torch.load(pretrained_path, map_location="cpu")
    loaded_keys = load_pretrained_backbone(encoder, pretrained_path, strict=task["backbone"]["strict"])
    logger.info(encoder)
    
    # ===== Stage 6: Setup model with hierarchical head =====
    head_cfg = dict(
        hierarchical_dropout = task["head"]["hierarchical_dropout"],
        bottleneck = task["head"]["bottleneck"],
    )
    model = TaxonomyModel.for_classify(
        encoder=encoder,
        levels=levels,
        num_classes_by_rank=num_classes_by_rank,
        **head_cfg
    )
    logger.info(model)
    model.to(device)
    
    
    # -------- Data split
    df = pd.read_csv(paths["data"]["prepared_csv"])
    

    fold_col = "fold_exp1" if task["fold_scheme"] == "exp1" else "fold_exp2"
    need_cols = {"sequence", fold_col, *levels}
    assert need_cols.issubset(df.columns), f"CSV must contain {need_cols}"
    
    fold = int(task["fold_index"])
    df_train = df[df[fold_col] != fold].reset_index(drop=True)
    df_val   = df[df[fold_col] == fold].reset_index(drop=True)

    logger.info("len(df_train): %r", len(df_train))
    logger.info("head: %r", df_train.head())
    logger.info("len(df_val): %r", len(df_val))
    logger.info("head: %r", df_val.head())
    # summarize_dataframe(df_train, df_val, fold)
    

    # -------- Build datasets & loaders
    label_enc = LabelEncoder(space=ls, unknown="error")
    train_dataset = ClassifyDataset(df_train, preproc_train, label_enc, levels)
    val_dataset = ClassifyDataset(df_val, preproc_val, label_enc, levels)
    
    # #Out commented but valid:
    # validate_datasets_and_log(
    #     train_dataset=train_dataset,
    #     val_dataset=val_dataset,
    #     levels=levels,
    #     num_classes_by_rank=num_classes_by_rank,
    #     optimal_length=prep["optimal_length"],
    #     logger=logger,
    #     sample_size_for_stats=min(2048, len(train_dataset))
    # )

    # -- Species counts for binned metrics
    sp_counts = df["species"].value_counts()
    sp_to_idx = {lab: label_enc.encode("species", str(lab)) for lab in sp_counts.index}
    C_sp = ls.num_classes("species")
    species_counts_vec = torch.zeros(C_sp, dtype=torch.long)
    for lab, cnt in sp_counts.items():
        species_counts_vec[sp_to_idx[str(lab)]] = int(cnt)

    species_size_bins = [(1,1),(2,2),(3,3),(4,4),(5,5),(6,10),(11,20),(21,50),(51,100),(101,1000),(1001,10**9)]
    species_size_labels = ["1","2","3","4","5","6-10","11-20","21-50","51-100","101-1000",">1000"]
    
    size_metric_names = register_accuracy_by_class_size_bins(
        prefix="acc_size",
        class_sizes_global=species_counts_vec,
        bins=species_size_bins,
        labels=species_size_labels,
    )
    
    # --

    logger.info(f"Train set: {train_dataset.__repr__()}")
    logger.info(f"Val set: {val_dataset.__repr__()}")


    logger.info("=== Dataloaders ===")
    dl = task["dataloader"]
    batch_size   = dl["batch_size"]
    num_workers  = dl["num_workers"]
    pin_memory   = dl["pin_memory"]
    drop_last_tr = dl["drop_last_train"]
    persistent_workers = (num_workers > 0)
    
    alpha_cfg = dl.get("sampler_alpha", 1.0) #TODO: Implement alpha in cfg when found good value
    alpha = args.sampler_alpha if args.sampler_alpha is not None else alpha_cfg
    if alpha < 0:
        raise ValueError(f"--sampler_alpha must be >= 0 (got {alpha})")
    # train_sampler = SpeciesBalancedSampler.from_dataset(train_dataset)
    
    train_sampler = SpeciesPowerSampler.from_dataset(
        train_dataset,
        species_col="species",
        alpha=alpha,
        clip_max_ratio=dl.get("sampler_clip_max_ratio")  # optional
    )

    w = train_sampler.weights.double()
    if not torch.isfinite(w).all():
        raise ValueError("Sampler weights contain non-finite values.")
    logger.info("Sampler weights (min/mean/max): %.4f / %.4f / %.4f",
                w.min().item(), w.mean().item(), w.max().item())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_tr,
        persistent_workers=persistent_workers,
        prefetch_factor=dl.get("prefetch_factor"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers,
        prefetch_factor=dl.get("prefetch_factor"),
    )
    
    logger.info("Dataloaders set up complete")
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

    opt = task["optimizer"]
    optimizer = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": opt["weight_decay"]},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=opt["learning_rate"], betas=tuple(opt["betas"]), eps=opt["eps"]
    )
    
    logger.info("=== Optimizer ===")
    logger.info("AdamW: lr=%.3e, weight_decay=%.3g.", opt["learning_rate"], opt["weight_decay"])
    logger.info("Param groups (YAML): \n%s", group_params_yaml(model))
    
    sched_kind = task["scheduler"]["kind"]
    schedule = cfg["scheduler_catalog"][sched_kind]
    logger.info("Optimizer schedule kind=%s", sched_kind)
    scheduler = build_scheduler_unified(optimizer, steps_per_epoch, schedule)
    
    masks_train, masks_val = _load_fold_masks(paths["data"]["fold_masks"], fold, levels, num_classes_by_rank, logger)

    if hasattr(model.classifier, "set_prev_logit_gates") and use_gate:
        model.classifier.set_prev_logit_gates(masks_train)

    # Run paths from config
    run = paths["finetune"]
    clear_run_dir(Path(run["run_dir"]), logger)
    Path(run["checkpoints_dir"]).mkdir(parents=True, exist_ok=True)
    attach_file_logger(Path(run["log_file"]), file_level=logging.DEBUG, flush_buffer=True)
    logger.info("Run dirs: arch_root=%s | fold_dir=%s | run_dir=%s | log_file=%s",
                run["arch_root"], run["fold_dir"], run["run_dir"], run["log_file"])
    
    CHECKPOINTS_DIR = Path(run["checkpoints_dir"])
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    trainer = ClassificationTrainer(
        model           = model,
        train_loader    = train_loader,
        val_loader      = val_loader,
        levels          = levels,

        masks_train     = masks_train,
        masks_val       = masks_val,

        optimizer       = optimizer,
        scheduler       = scheduler,
        amp             = task["trainer"]["amp"],

        log_every       = task["trainer"]["log_every"],
        checkpoints_dir = CHECKPOINTS_DIR,
        select_best_by  ="species" if "species" in levels else levels[-1],

        rank_metrics=("accuracy", *size_metric_names),
    )

    # Run training
    logger.info("=== Training: Starting ===")
    summary = trainer.train(max_epochs=task["trainer"]["max_epochs"])
    logger.info("=== Training: done ===")

    # Results summary
    with open(run["results_file"], "w") as f:
        json.dump(
            {
                "epochs": task["trainer"]["max_epochs"],
                "batch_size": dl["batch_size"],
                "learning_rate": opt["learning_rate"],
                "weight_decay": opt["weight_decay"],
                "levels": levels,
                "fold_index": fold,
                "fold_scheme": task["fold_scheme"],
            },
            f,
            indent=2,
        )
if __name__ == "__main__":
    main()
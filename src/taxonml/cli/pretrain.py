#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLM pretraining entry point (config-free), styled like finetune.py.

Responsibilities:
- Deterministic setup, logging
- Resolve profile-aware data artifacts (prepared.csv under experiments/<exp>/data/<profile>)
- Build k-mer vocab & preprocessors (derive max_position_embeddings = optimal_length + 2)
- Build BERT encoder → wrap with tied-weight MLM head (TaxonomyModel.for_pretrain)
- Create MLMDataset + DataLoaders (shuffle train; no sampler)
- AdamW + unified scheduler (same pattern as finetune.py)
- Train with MLMTrainer (saves best.pt + last.pt in run dir)
- Write a compact results.json and promote best.pt to a global pretrained/<arch_id>/best.pt

Assumptions:
- prepared.csv has at least: ["sequence", "fold_exp1"]
- Project modules are importable (installed or on PYTHONPATH)
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel

# ==== USER SETTINGS (edit these to your environment) =========================
EXPERIMENTS_ROOT = "/mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/experiments"
PRETRAINED_ROOT  = "/mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/pretrained_models"
EXPERIMENT_NAME  = "sequence_fold_full"
# PROFILE          = "full"   # "full" | "debug"
FOLD             = 7        # used to carve train/val as in prior code (fold_exp1)

# Tokenizer / sequence params (from your old config)
# K                    = 3
# MAX_BASES            = 1500
ALPHABET             = ["A", "C", "G", "T"]
MLM_MASKING_PCT      = 0.15

# Encoder (from your old config)
# HIDDEN_SIZE          = 512
# NUM_LAYERS           = 10
# NUM_HEADS            = 8
# INTERMEDIATE_SIZE    = 2048
DROPOUT_HIDDEN       = 0.10
DROPOUT_ATTENTION    = 0.10

# Training defaults (from your old config; pretrain task overrides)
BATCH_SIZE           = 160
LEARNING_RATE        = 1.8e-4
WEIGHT_DECAY         = 0.01
MAX_EPOCHS           = 600
AMP                  = True
NUM_WORKERS          = 4
PIN_MEMORY           = True
LOG_EVERY            = 100
SAVE_EVERY_EPOCHS    = 1
# ============================================================================

# Project modules
from taxonml.preprocessing.vocab import Vocabulary, KmerVocabConstructor
from taxonml.preprocessing import augmentation, tokenization, padding, truncation
from taxonml.preprocessing.preprocessor import Preprocessor
from taxonml.data.datasets import MLMDataset
from taxonml.encoders.bert import derive_arch_id_from_cfg
from taxonml.models.taxonomy_model import TaxonomyModel
from taxonml.training.trainers import MLMTrainer
from taxonml.training.schedulers import build_scheduler_unified
from taxonml.runners.finetune import build_profile_paths

# ---------- Utilities (same style as finetune.py) ----------------------------

def setup_logging(console_level: int = logging.INFO) -> None:
    logging.captureWarnings(True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def build_profile_paths(
    experiments_root: str | Path,
    experiment_name: str,
    profile: str,  # "full" | "debug"
) -> Dict[str, Path]:
    exp_root = Path(experiments_root) / experiment_name
    data_dir = exp_root / "data" / profile
    logs_dir = exp_root / "logs"
    return {
        "exp_root": exp_root,
        "data_dir": data_dir,
        "prepared_csv": data_dir / "prepared.csv",
        "label_space_json": data_dir / "label_space.json",  # not used here
        "summary_json": data_dir / "summary.json",
        "logs_dir": logs_dir,
    }

def build_pretrain_run_paths(
    experiments_root: str | Path,
    experiment_name: str,
    arch_id: str,
    profile: str,
) -> Dict[str, Path]:
    exp_root = Path(experiments_root) / experiment_name
    arch_root = exp_root / arch_id / profile
    run_dir  = arch_root / "pretrain"
    return {
        "arch_root": arch_root,
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


# ---------- Main -------------------------------------------------------------

def main() -> None:
    setup_logging(logging.INFO)
    logger = logging.getLogger("pretrain_v2")
    logger.info("=== Startup (MLM pretraining) ===")
    debug = False

    # Full-size defaults (production)
    k = 3
    max_bases = 1500
    hidden_size = 512
    num_hidden_layers = 10
    num_attention_heads = 8
    intermediate_size = 2048
    hidden_dropout_prob = 0.1
    attention_probs_dropout_prob = 0.1

    batch_size = 126          # task-specific override from your old config
    base_lr    = 1.8e-4
    max_epochs = 600
    num_workers = 4
    amp = True

    # --- DEBUG OVERRIDES (like in finetune.py) ---
    if debug:
        max_bases = 600
        hidden_size = 64
        num_hidden_layers = 2
        num_attention_heads = 2
        intermediate_size = 256

        batch_size = 16
        max_epochs = 2
        num_workers = 0
        amp = False   # keep things simple on T4


    profile = "debug" if debug else "full"
    print(f"[runtime] profile={profile} (debug={debug})")
    # 0) Seeds & device
    set_all_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | AMP=%s", device, AMP)

    # 1) Artifacts (profile-aware)
    art = build_profile_paths(EXPERIMENTS_ROOT, EXPERIMENT_NAME, profile)
    DATA_PATH        = art["prepared_csv"] 
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"prepared.csv not found: {DATA_PATH}")

    #logger.info("Artifacts (profile=%s):", PROFILE)
    logger.info("  prepared_csv = %s", DATA_PATH)

    # 2) Load data & fold split (consistent with prior code)
    df = pd.read_csv(DATA_PATH)
    assert {"sequence", "fold_exp1"}.issubset(df.columns), \
        "prepared.csv must contain columns: sequence, fold_exp1"
    df_train = df[df["fold_exp1"] != FOLD].reset_index(drop=True)
    df_val   = df[df["fold_exp1"] == FOLD].reset_index(drop=True)
    logger.info("Train/Val sizes: %d / %d (fold=%d)", len(df_train), len(df_val), FOLD)

    # 3) Vocab + preprocessors (derive optimal_length, MPE=optimal_length+2)
    logger.info("=== Vocab & Preprocessors ===")
    vocab = Vocabulary()
    vocab.build_from_constructor(KmerVocabConstructor(k=k, alphabet=ALPHABET))
    logger.info("Vocab size: %d", len(vocab))

    assert max_bases % k == 0, "MAX_BASES must be divisible by K"
    optimal_length = max_bases // k
    max_position_embeddings = optimal_length + 2  # + [CLS, SEP]

    tok   = tokenization.KmerStrategy(k=k, padding_alphabet=ALPHABET)
    pad   = padding.PaddingEndStrategy(optimal_length=optimal_length)
    trunc = truncation.SlidingWindowTruncationStrategy(optimal_length=optimal_length)

    # For pretraining we avoid extra base-level noise; use Identity for eval and light/no aug for train.
    aug_train = augmentation.IdentityStrategy()
    aug_eval  = augmentation.IdentityStrategy()

    preproc_train = Preprocessor(aug_train, tok, pad, trunc, vocab)
    preproc_val   = Preprocessor(aug_eval,  tok, pad, trunc, vocab)

    # 4) Encoder (HuggingFace BERT) + arch_id
    logger.info("=== Encoder ===")
    bert_config = BertConfig(
    vocab_size=len(vocab),
    hidden_size=hidden_size,
    num_hidden_layers=num_hidden_layers,
    num_attention_heads=num_attention_heads,
    intermediate_size=intermediate_size,
    hidden_dropout_prob=hidden_dropout_prob,
    attention_probs_dropout_prob=attention_probs_dropout_prob,
    max_position_embeddings=(optimal_length + 2),
    )
    encoder = BertModel(bert_config)
    arch_id = derive_arch_id_from_cfg(bert_config)   # expected pattern: bert_h512_L10_H8_i2048_P502
    logger.info("arch_id=%s | MPE=%d | optimal_length=%d", arch_id, max_position_embeddings, optimal_length)

    # 5) Model (MLM head; tie to input embeddings)
    model = TaxonomyModel.for_pretrain(
        encoder=encoder,
        vocab_size=len(vocab),
        mlm_hidden=None,             # default to hidden_size
        mlm_dropout=0.10,            # matches your single_rank.heads.dropout
        tie_mlm_to_embeddings=True,
    ).to(device)
    logger.debug("Model repr:\n%s", repr(model))

    # 6) Datasets & loaders
    ds_train = MLMDataset(df_train["sequence"], preproc_train, masking_percentage=MLM_MASKING_PCT)
    ds_val   = MLMDataset(df_val["sequence"],   preproc_val,   masking_percentage=MLM_MASKING_PCT)

    persistent_workers = (num_workers > 0)
    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=persistent_workers,
        prefetch_factor=(4 if num_workers > 0 else None),
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=persistent_workers,
        prefetch_factor=(4 if num_workers > 0 else None),
    )

    logger.info("Dataloaders ready (batch=%d, workers=%d, prefetch=%s)",
                batch_size, num_workers, (4 if num_workers > 0 else None))

    # 7) Optimizer (AdamW w/ weight decay on non-norm/bias)
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay_params if any(nd in name for nd in ["bias", "LayerNorm.weight"])
         else decay_params).append(param)

    optimizer = torch.optim.AdamW(
        [{"params": decay_params,   "weight_decay": WEIGHT_DECAY},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8
    )

    # 8) Scheduler (unified; match finetune.py style)
    steps_per_epoch = math.ceil(len(ds_train) / max(1, batch_size))
    schedule_by_kind: Dict[str, Any] = {
        "tri": {
            "base": "step",
            "warmup": {"type": "linear", "duration": 1},     # 1 epoch warmup (see notes below)
            "main": {"type": "tri", "plateau": 3, "decay": 8},
            "floor": {"min_factor": 1e-2},
        },
        "cosine": {
            "base": "step",
            "warmup": {"type": "linear", "duration": 1},
            "main": {"type": "cosine", "epochs": 10},
            "floor": {"min_factor": 1e-2},
        },
    }
    SCHED_KIND = "tri"
    scheduler = build_scheduler_unified(optimizer, steps_per_epoch, schedule_by_kind[SCHED_KIND])

    # 9) Run paths & checkpoints
    run_paths = build_pretrain_run_paths(EXPERIMENTS_ROOT, EXPERIMENT_NAME, arch_id, profile)
    clear_run_dir(run_paths["run_dir"], logger)
    ckpt_dir = run_paths["checkpoints_dir"]; ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run dirs: arch_root=%s | run_dir=%s", run_paths["arch_root"], run_paths["run_dir"])

    # 10) Trainer
    trainer = MLMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        amp=amp,
        log_every=20 if debug else 100,   # optional: chatty in debug
        checkpoints_dir=str(ckpt_dir),
        save_every_n_epochs=1,
)
    logger.info("=== Training: start ===")
    summary = trainer.fit(MAX_EPOCHS, resume=False)
    logger.info("=== Training: done ===")

    # 11) Results + promote best.pt to PRETRAINED_ROOT/<arch_id>/best.pt
    with open(run_paths["results_file"], "w") as f:
        json.dump({
            "best_val_loss": float(summary.get("best_val_loss", float("inf"))),
            "best_ckpt": summary.get("best_path"),
            "epochs": MAX_EPOCHS,
            "batch_size": batch_size,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
        }, f, indent=2)

    try:
        src_best = ckpt_dir / "best.pt"  # written by MLMTrainer
        if src_best.exists():
            dst_dir = Path(PRETRAINED_ROOT) / arch_id
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_best = dst_dir / "best.pt"
            with open(src_best, "rb") as s, open(dst_best, "wb") as g:
                g.write(s.read())
            logger.info("Promoted best checkpoint → %s", dst_best)
        else:
            logger.warning("best.pt not found in %s; skip promote.", ckpt_dir)
    except Exception as e:
        logger.warning("Promote best failed: %s", e)

if __name__ == "__main__":
    main()

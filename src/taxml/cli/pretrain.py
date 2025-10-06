from __future__ import annotations
import logging, json, math, shutil
from pathlib import Path

import pandas as pd
import torch, transformers 
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel
from sklearn.model_selection import train_test_split

from taxml.core.config import load_config
from taxml.core.logging import setup_logging, attach_file_logger
from taxml.data.datasets import MLMDataset
from taxml.models.taxonomy_model import TaxonomyModel
from taxml.preprocessing.vocab import Vocabulary, KmerVocabConstructor
from taxml.preprocessing import augmentation, tokenization, padding, truncation
from taxml.training.trainers import MLMTrainer
from taxml.training.schedulers import build_scheduler_unified

logger = logging.getLogger(__name__)

import json
import math
import os
import random
import shutil
import argparse
from pathlib import Path
from typing import Any, Dict

# # ==== USER SETTINGS (edit these to your environment) =========================
# EXPERIMENTS_ROOT = "/mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/experiments"
# PRETRAINED_ROOT  = "/mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/pretrained_models"
# EXPERIMENT_NAME  = "sequence_fold_full"
# # PROFILE          = "full"   # "full" | "debug"
# FOLD             = 7        # used to carve train/val as in prior code (fold_exp1)

# # Tokenizer / sequence params (from your old config)
# # K                    = 3
# # MAX_BASES            = 1500
# ALPHABET             = ["A", "C", "G", "T"]
# MLM_MASKING_PCT      = 0.15

# # Encoder (from your old config)
# # HIDDEN_SIZE          = 512
# # NUM_LAYERS           = 10
# # NUM_HEADS            = 8
# # INTERMEDIATE_SIZE    = 2048
# DROPOUT_HIDDEN       = 0.10
# DROPOUT_ATTENTION    = 0.10

# # Training defaults (from your old config; pretrain task overrides)
# BATCH_SIZE           = 160
# LEARNING_RATE        = 1.8e-4
# WEIGHT_DECAY         = 0.01
# MAX_EPOCHS           = 600
# AMP                  = True
# NUM_WORKERS          = 4
# PIN_MEMORY           = True
# LOG_EVERY            = 100
# SAVE_EVERY_EPOCHS    = 1
# # ============================================================================

# Project modules
from taxml.preprocessing.vocab import Vocabulary, KmerVocabConstructor
from taxml.preprocessing import augmentation, tokenization, padding, truncation
from taxml.preprocessing.preprocessor import Preprocessor
from taxml.data.datasets import MLMDataset
from taxml.encoders.bert import derive_arch_id_from_cfg
from taxml.models.taxonomy_model import TaxonomyModel
from taxml.training.trainers import MLMTrainer
from taxml.training.schedulers import build_scheduler_unified
# from taxml.cli.finetune import build_profile_paths
from taxml.core.logging import setup_logging, attach_file_logger


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

def clear_dir(p: Path) -> None:
    if p.exists():
        logger.warning("Clearing existing dir: %s", p)
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


# ---------- Main -------------------------------------------------------------

def main() -> None:
    setup_logging(console_level=logging.INFO, buffer_early=True)
    logger.info("=== Startup (MLM pretraining) ===")
    
    # ---- CLI Args ----
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to master experiment YAML"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Toggle debug mode; runs reduced model and dataset"
    )
    args = parser.parse_args()
    
    #  ---- Config ----
    cfg = load_config(
        mode        = "pretrain",
        config_path = args.config,
        debug       = args.debug 
        )    

    # resolve paths for this run and ensure dirs exists before attaching file logger
    paths = cfg["paths_active"]
    run = paths["pretrain"]
    task = cfg["pretrain"]

    logger.info("=== Runtime summary ===")
    logger.info(
        "experiment=%s | runtime=%s | debug=%s",
        cfg["experiment"]["name"],
        cfg["runtime"],
        args.debug,
    )
    logger.info("active data artifacts: prepared_csv=%s | label_space_json=%s | fold_masks=%s",
                paths["data"]["prepared_csv"], paths["data"]["label_space_json"], paths["data"]["fold_masks"])
    logger.debug("Full cfg snapshot:\n%s", json.dumps(cfg, indent=2, default=str))

    
    # ---- Seeds & Device ----
    set_all_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        logger.info("Versions: torch=%s | transformers=%s | cuda_available=%s",
                    torch.__version__, transformers.__version__, torch.cuda.is_available())
        if torch.cuda.is_available():
            logger.info("CUDA device: %s | capability=%s | total_mem=%.2f GB",
                        torch.cuda.get_device_name(0),
                        torch.cuda.get_device_capability(0),
                        torch.cuda.get_device_properties(0).total_memory / (1024**3))
    except Exception as e:
        logger.warning("Env introspection failed: %r", e)
    

    # ---- Data (naive sklearn split) ----
    logger.info("=== Data ===")
    data_path = paths["data"]["prepared_csv"]
    if not data_path.exists():
        raise FileNotFoundError(f"prepared.csv not found: {data_path}")
    df = pd.read_csv(data_path)
    assert "sequence" in df.columns, "prepared.csv missing required column 'sequence'"

    logger.info("Loaded prepared.csv from: %s", paths["data"]["prepared_csv"])
    logger.info("prepared.csv rows=%d", len(df))
    logger.debug("prepared.csv head:\n%s", df.head().to_string(index=False))
    

    df_train, df_val = train_test_split(
        df,
        test_size       =   task["mlm"]["val_split"],
        shuffle         =   True,
        random_state    =   42
    )
    logger.info("Train/Val split: test_size=%.3f -> train=%d, val=%d",
                cfg["pretrain"]["mlm"]["val_split"], len(df_train), len(df_val))
    
    # 3) Vocab + preprocessors (derive optimal_length, MPE=optimal_length+2)
    logger.info("=== Vocab & Preprocessors ===")
    prep = cfg["preprocessing"]

    vocab = Vocabulary()
    constructor = KmerVocabConstructor(
        k = prep["k"],
        alphabet = prep["alphabet"]
        )
    vocab.build_from_constructor(
        constructor = constructor
        )
    
    tok     = tokenization.KmerStrategy(k = prep["k"], alphabet = prep["alphabet"])
    pad     = padding.PaddingEndStrategy(optimal_length = prep["optimal_length"])
    trunc   = truncation.SlidingWindowTruncationStrategy(optimal_length = prep["optimal_length"])
    
    modifier    = augmentation.SequenceModifier(alphabet = prep["alphabet"])
    aug_train   = augmentation.BaseStrategy(modifier = modifier, alphabet = prep["alphabet"], modification_probability = prep["mod_prob"])
    aug_val     = augmentation.IdentityStrategy()
    
    preproc_train = Preprocessor(
        augmentation_strategy   = aug_train,
        tokenization_strategy   = tok,
        padding_strategy        = pad,
        truncation_strategy     = trunc,
        vocab = vocab
        )
    
    preproc_val = Preprocessor(
        augmentation_strategy   = aug_val,
        tokenization_strategy   = tok,
        padding_strategy        = pad,
        truncation_strategy     = trunc,
        vocab = vocab
        )
    
    logger.info("Alphabet=%s | k=%d | max_bases=%d | optimal_length=%d | MPE=%d | mod_prob=%.3f",
                prep["alphabet"], prep["k"], prep["max_bases"],
                prep["optimal_length"], prep["max_position_embeddings"],
                prep["mod_prob"])
    logger.info("Vocab size: %d", len(vocab))
    logger.debug("Tokenizer: %r", tok)
    logger.debug("Padding:   %r", pad)
    logger.debug("Truncation:%r", trunc)
    logger.debug("Aug train: %r", aug_train)
    logger.debug("Aug val:   %r", aug_val)

    # 4) Encoder (HuggingFace BERT) + arch_id
    logger.info("=== Encoder ===")
    enc = cfg["model"]["encoder"]
    
    bert_config = BertConfig(
        vocab_size  = len(vocab),
        hidden_size = enc["hidden_size"],
        num_hidden_layers   = enc["num_hidden_layers"],
        num_attention_heads = enc["num_attention_heads"],
        intermediate_size   = enc["intermediate_size"],
        hidden_dropout_prob = enc["hidden_dropout_prob"],
        attention_probs_dropout_prob = enc["attention_probs_dropout_prob"],
        max_position_embeddings      = prep["max_position_embeddings"],
    )
    encoder = BertModel(bert_config)
    arch_id = cfg["arch"]["id"]
    logger.info("BERT: hidden=%d | layers=%d | heads=%d | i=%d | MPE=%d",
            enc["hidden_size"], enc["num_hidden_layers"], enc["num_attention_heads"],
            enc["intermediate_size"], prep["max_position_embeddings"])
    logger.info("arch_id=%s", cfg["arch"]["id"])
    logger.debug("Encoder repr:\n%s", repr(encoder))


    # 5) Model (MLM head; tie to input embeddings)
    logger.info("=== Full model ===")

    model = TaxonomyModel.for_pretrain(
        encoder     = encoder,
        vocab_size  = len(vocab),
        mlm_hidden  = None,
        mlm_dropout = task["mlm"]["mlm_dropout"],
        tie_emb     = task["mlm"]["tie_to_embeddings"],
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model params (total/trainable): %s / %s", f"{n_params:,}", f"{n_train:,}")
    logger.info("Head: MLMHead (tied=%s)", cfg["pretrain"]["mlm"]["tie_to_embeddings"])

    # 6) Datasets & loaders
    ds_train = MLMDataset(
        sequences           = df_train["sequence"],
        preprocessor        = preproc_train,
        masking_percentage  = task["mlm"]["masking_pct"]
    )
    ds_val = MLMDataset(
        sequences           = df_val["sequence"],
        preprocessor        = preproc_val,
        masking_percentage  = task["mlm"]["masking_pct"]
    )

    persistent_workers = (task["dataloader"]["num_workers"]> 0)
    
    logger.info("=== Dataloaders ===")
    train_loader = DataLoader(
        ds_train,
        batch_size  = task["dataloader"]["batch_size"],
        shuffle     = True,
        num_workers = task["dataloader"]["num_workers"],
        pin_memory  = task["dataloader"]["pin_memory"],
        drop_last   = task["dataloader"]["drop_last_train"],
        persistent_workers  = persistent_workers,
        prefetch_factor     = task["dataloader"].get("prefetch_factor"),
    )
    
    val_loader = DataLoader(
        ds_val,
        batch_size  = task["dataloader"]["batch_size"],
        shuffle     = False,
        num_workers = task["dataloader"]["num_workers"],
        pin_memory  = task["dataloader"]["pin_memory"],
        drop_last   = False,
        persistent_workers  = persistent_workers,
        prefetch_factor     = task["dataloader"].get("prefetch_factor"),
    )
    dl = cfg["pretrain"]["dataloader"]
    logger.info("train: batch_size=%d | shuffle=%s | num_workers=%d | pin_memory=%s | drop_last_train=%s | persistent_workers=%s | prefetch_factor=%s",
                dl["batch_size"], True, dl["num_workers"], dl["pin_memory"], dl["drop_last_train"],
                (dl["num_workers"]>0), dl.get("prefetch_factor"))
    logger.info("val:   batch_size=%d | shuffle=%s | num_workers=%d | pin_memory=%s | drop_last=%s | persistent_workers=%s | prefetch_factor=%s",
                dl["batch_size"], False, dl["num_workers"], dl["pin_memory"], False,
                (dl["num_workers"]>0), dl.get("prefetch_factor"))

    try:
        _b = next(iter(DataLoader(ds_train, batch_size= min(8, len(ds_train)), shuffle=False, num_workers=0)))
        logger.info("Probe batch shapes: input_ids=%s | attention_mask=%s",
                    tuple(_b["input_ids"].shape), tuple(_b["attention_mask"].shape))
        logger.debug("Probe input_ids[0][:20]=%s", _b["input_ids"][0][:20].tolist() if len(_b["input_ids"]) else [])
    except Exception as e:
        logger.warning("Probe batch failed (non-fatal): %r", e)

    # ---- Optimizer & Scheduler ----
    logger.info("=== Optimizer ===")
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_no_decay = any(nd in name for nd in ["bias", "LayerNorm.weight"])
        target_list = no_decay_params if is_no_decay else decay_params
        target_list.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": task["optimizer"]["weight_decay"]},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        param_groups,
        lr      = task["optimizer"]["learning_rate"],
        betas   = tuple(task["optimizer"]["betas"]),
        eps     = task["optimizer"]["eps"],
    )
    
    opt = cfg["pretrain"]["optimizer"]
    logger.info("AdamW: lr=%.3e | weight_decay=%.3g | betas=%s | eps=%.1e",
            opt["learning_rate"], opt["weight_decay"], tuple(opt["betas"]), opt["eps"])
    
    # 8) Scheduler
    logger.info("=== Scheduler ===")
    steps_per_epoch = math.ceil(len(ds_train) / max(1, task["dataloader"]["batch_size"]))
    logger.info("steps_per_epoch=%d | scheduler.kind=%s", steps_per_epoch, cfg["pretrain"]["scheduler"]["kind"])
    # logger.debug("Scheduler spec:\n%s", json.dumps(cfg["scheduler_catalog"][sched_kind], indent=2))
    sched_kind = task["scheduler"]["kind"]
    if sched_kind not in cfg["scheduler_catalog"]:
        raise KeyError(f"Unknown scheduler kind: {sched_kind!r}")
    schedule = cfg["scheduler_catalog"][sched_kind]
    scheduler = build_scheduler_unified(
        optimizer = optimizer,
        steps_per_epoch = steps_per_epoch,
        schedule = schedule
        )

    # 9) Run paths & checkpoints
    run = paths["pretrain"]
    clear_dir(Path(run["run_dir"]))  # remove any previous run dir
    Path(run["checkpoints_dir"]).mkdir(parents=True, exist_ok=True)
    # attach AFTER dirs exist and BEFORE training; flush early buffered logs
    attach_file_logger(
        file_path=Path(run["log_file"]),
        file_level=logging.DEBUG,
        flush_buffer=True,
    )
    logger.info("Run dirs: arch_root=%s | run_dir=%s | checkpoints=%s | log_file=%s",
            run["arch_root"], run["run_dir"], run["checkpoints_dir"], run["log_file"])
    


    # 10) Trainer
    trainer = MLMTrainer(
        model           = model,
        train_loader    = train_loader,
        val_loader      = val_loader,
        optimizer       = optimizer,
        scheduler       = scheduler,
        amp             = task["trainer"]["amp"],
        log_every       = task["trainer"]["log_every"],
        checkpoints_dir = str(run["checkpoints_dir"]),
    )
    logger.info("=== Training: start ===")
    
    summary = trainer.train(
        max_epochs  = task["trainer"]["max_epochs"],
        resume      = False
        )
    
    logger.info("=== Training: done ===")
    logger.info("Summary: %s", {k: summary.get(k) for k in ["best_val_loss","best_path","last_epoch","num_updates"] if k in summary})

    # ---- Results & promotion ----
    with open(run["results_file"], "w") as f:
        json.dump(
            {
                "best_val_loss": float(summary.get("best_val_loss", float("inf"))),
                "best_ckpt":    summary.get("best_path"),
                "epochs":       task["trainer"]["max_epochs"],
                "batch_size":   task["dataloader"]["batch_size"],
                "learning_rate":task["optimizer"]["learning_rate"],
                "weight_decay": task["optimizer"]["weight_decay"],
            },
            f,
            indent = 2
        )

    src_best = Path(run["checkpoints_dir"]) / "best.pt"
    dst_best = Path(paths["promotion_best"])  # resolved to <pretrained_root>/<arch_id>/best.pt
    if src_best.exists():
        dst_best.parent.mkdir(parents=True, exist_ok=True)
        dst_best.write_bytes(src_best.read_bytes())
        logger.info("Promoted best checkpoint → %s", dst_best)
    else:
        logger.warning("best.pt not found in %s; skip promote.", src_best.parent)

if __name__ == "__main__":
    main()

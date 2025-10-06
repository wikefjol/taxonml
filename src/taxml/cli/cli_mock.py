#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys, json
from pathlib import Path

from taxml.core.config import load_config
import yaml

def _read_yaml(p: Path) -> dict:
    with open(p, "r") as f:
        return yaml.safe_load(f)

def _expect(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)

def summarize(case, cfg, keys):
    flat = {}
    for k in keys:
        v = cfg
        for part in k.split("."):
            v = v[part]
        flat[k] = str(v)
    print(f"\n[{case}]")
    for k, v in flat.items():
        print(f"  {k:40s} = {v}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to experiment YAML")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--levels", default="all")
    args = ap.parse_args()

    cfg_yaml = _read_yaml(Path(args.config))
    base_h = cfg_yaml["model"]["encoder"]["hidden_size"]
    dbg_h  = cfg_yaml["debug_overrides"]["model"]["encoder"]["hidden_size"]
    base_mb = cfg_yaml["preprocessing"]["max_bases"]
    dbg_mb  = cfg_yaml["debug_overrides"]["preprocessing"]["max_bases"]
    base_bs = cfg_yaml["tasks"]["pretrain"]["dataloader"]["batch_size"]
    dbg_bs  = cfg_yaml["debug_overrides"]["tasks"]["pretrain"]["dataloader"]["batch_size"]

    # ---------- PREP (debug ignored) ----------
    cfg = load_config(mode="prep", config_path=args.config, debug=False)
    summarize("prep", cfg, [
        "experiment.name",
        "paths.profile_dirs.full",
        "paths.profile_dirs.debug",
        "paths.prepared_csv.full",
        "paths.prepared_csv.debug",
    ])

    # ---------- PRETRAIN FULL ----------
    cfg = load_config(mode="pretrain", config_path=args.config, debug=False)
    summarize("pretrain/full", cfg, [
        "arch.id",
        "preprocessing.max_bases",
        "pretrain.dataloader.batch_size",
        "paths_active.data.prepared_csv",
        "paths_active.pretrain.run_dir",
    ])
    _expect(cfg["model"]["encoder"]["hidden_size"] == base_h, "FULL: encoder.hidden_size mismatch")
    _expect(cfg["preprocessing"]["max_bases"] == base_mb, "FULL: max_bases mismatch")
    _expect(cfg["pretrain"]["dataloader"]["batch_size"] == base_bs, "FULL: batch_size mismatch")
    _expect("/data/full/" in str(cfg["paths_active"]["data"]["prepared_csv"]), "FULL: prepared.csv path not in /data/full")

    # ---------- PRETRAIN DEBUG ----------
    cfg = load_config(mode="pretrain", config_path=args.config, debug=True)
    summarize("pretrain/debug", cfg, [
        "arch.id",
        "preprocessing.max_bases",
        "pretrain.dataloader.batch_size",
        "paths_active.data.prepared_csv",
        "paths_active.pretrain.run_dir",
    ])
    _expect(cfg["model"]["encoder"]["hidden_size"] == dbg_h, "DEBUG: encoder.hidden_size mismatch")
    _expect(cfg["preprocessing"]["max_bases"] == dbg_mb, "DEBUG: max_bases mismatch")
    _expect(cfg["pretrain"]["dataloader"]["batch_size"] == dbg_bs, "DEBUG: batch_size mismatch")
    _expect("/data/debug/" in str(cfg["paths_active"]["data"]["prepared_csv"]), "DEBUG: prepared.csv path not in /data/debug")
    _expect("P202" in cfg["arch"]["id"] or cfg["preprocessing"]["max_bases"] != base_mb, "DEBUG: arch_id not recomputed (MPE)")

    # ---------- FINETUNE FULL ----------
    lvls = cfg_yaml["label_space"]["levels"] if args.levels == "all" else [s.strip() for s in args.levels.split(",") if s.strip()]
    cfg = load_config(mode="finetune", config_path=args.config, debug=False, levels=lvls, fold_index=args.fold)
    summarize("finetune/full", cfg, [
        "arch.id",
        "finetune.fold_index",
        "paths_active.data.fold_masks",
        "paths_active.finetune.run_dir",
        "finetune.backbone.path",
    ])
    _expect("/data/full/" in str(cfg["paths_active"]["data"]["fold_masks"]), "FULL: finetune masks not in /data/full")

    # ---------- FINETUNE DEBUG ----------
    cfg = load_config(mode="finetune", config_path=args.config, debug=True, levels=lvls, fold_index=args.fold)
    summarize("finetune/debug", cfg, [
        "arch.id",
        "finetune.fold_index",
        "paths_active.data.fold_masks",
        "paths_active.finetune.run_dir",
        "finetune.backbone.path",
    ])
    _expect("/data/debug/" in str(cfg["paths_active"]["data"]["fold_masks"]), "DEBUG: finetune masks not in /data/debug")

    print("\nAll assertions passed ✅")

if __name__ == "__main__":
    sys.exit(main())

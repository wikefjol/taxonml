#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_experiment_prep.py

Reads a master experiment config (YAML), ingests BLAST-filtered union CSVs,
cleans and deduplicates data, and, **for each profile** (full, debug):

- assigns folds (exp1: sequence stratified; exp2: species-group)
- writes profile-scoped artifacts:
    data/<profile>/prepared.csv
    data/<profile>/label_space.json
    data/<profile>/fold_masks_exp1.json
    data/<profile>/fold_masks_exp2.json
    data/<profile>/summary.json

Also writes a top-level prep_summary.json with pointers to both profiles.

Logs to experiments/<name>/logs/prep.log

Usage:
  python scripts/01_experiment_prep.py --config configs/experiment_name.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedKFold

# Local lib
from taxml.labels.space import LabelSpace

# ---------------------------
# Helpers
# ---------------------------

REQUIRED_COLS = [
    "sequence", "kingdom", "phylum", "class", "order", "family", "genus", "species", "species_resolution"
]

def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logging.info("Logging initialized.")

def load_yaml(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def env_expand(s: str) -> str:
    return os.path.expandvars(os.path.expanduser(s))

def require_columns(df: pd.DataFrame, cols: List[str], context: str = "") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {context}")

def read_union_csvs(input_dir: Path, union_members: List[str]) -> pd.DataFrame:
    dfs = []
    for name in union_members:
        csv_path = input_dir / f"{name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing input file: {csv_path}")
        df = pd.read_csv(csv_path)
        require_columns(df, REQUIRED_COLS, context=str(csv_path))
        dfs.append(df)
        logging.info(f"Loaded {len(df):,} rows from {csv_path.name}")
    union_df = pd.concat(dfs, ignore_index=True)
    logging.info(f"Union total rows: {len(union_df):,}")
    return union_df

def apply_uppercase(df: pd.DataFrame) -> pd.DataFrame:
    if "sequence" in df.columns:
        n_nan = df["sequence"].isna().sum()
        if n_nan:
            logging.warning(f"{n_nan:,} rows have NaN sequence; dropping them.")
            df = df.dropna(subset=["sequence"])
        df["sequence"] = df["sequence"].astype(str).str.upper()
    return df

class NpEncoder(json.JSONEncoder):
    def default(self, o):
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)

def filter_by_resolution(df: pd.DataFrame, min_resolution: int) -> pd.DataFrame:
    before = len(df)
    df = df[df["species_resolution"] > min_resolution].copy()
    logging.info(f"Filtered by species_resolution > {min_resolution}: {before:,} -> {len(df):,}")
    return df

def dedupe_within_species_by_sequence(df: pd.DataFrame, keep_policy: str = "highest_resolution") -> pd.DataFrame:
    require_columns(df, ["species", "sequence", "species_resolution"], context="dedupe")
    before = len(df)
    if keep_policy == "highest_resolution":
        df = df.sort_values(["species", "sequence", "species_resolution"], ascending=[True, True, False])
        df = df.drop_duplicates(subset=["species", "sequence"], keep="first").reset_index(drop=True)
    elif keep_policy in {"first", "last"}:
        df = (
            df.groupby("species", group_keys=False)
            .apply(lambda g: g.drop_duplicates(subset=["sequence"], keep=keep_policy))
            .reset_index(drop=True)
        )
    else:
        raise ValueError(f"Unknown keep_policy: {keep_policy}")
    after = len(df)
    logging.info(f"Dedup within species by sequence: removed {before - after:,} duplicates ({before:,} -> {after:,})")
    return df

def assign_folds_sequence_stratified(df: pd.DataFrame, k: int, stratify_by: str, seed: int) -> pd.DataFrame:
    require_columns(df, [stratify_by], context="folds exp1")
    y = df[stratify_by].values
    X = np.arange(len(df))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    fold_col = np.full(len(df), -1, dtype=int)
    for i, (_, val_idx) in enumerate(skf.split(X, y), 1):
        fold_col[val_idx] = i
    df = df.copy()
    df["fold_exp1"] = fold_col
    logging.info("Fold_exp1 counts:\n" + df["fold_exp1"].value_counts().sort_index().to_string())
    return df

def assign_folds_species_group(df: pd.DataFrame, k: int, stratify_by: str, seed: int) -> pd.DataFrame:
    """
    Group K-fold by species (all rows for a species share a fold), stratified by genus when possible.
    Implemented via species-level SKFold on the genus labels.
    """
    require_columns(df, ["species", stratify_by], context="folds exp2")
    sp_to_gen = df.groupby("species")[stratify_by].first()
    unique_species = sp_to_gen.index.values
    genera = sp_to_gen.values

    # If stratification support is weak, fall back to random species split.
    if len(unique_species) < k or pd.Series(genera).value_counts().min() < k:
        logging.warning("Insufficient stratification support; falling back to random split over species.")
        rng = np.random.default_rng(seed)
        rng.shuffle(unique_species)
        chunks = np.array_split(unique_species, k)
        sp2fold = {sp: i + 1 for i, chunk in enumerate(chunks) for sp in chunk}
    else:
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        sp2fold = {}
        X = np.arange(len(unique_species))
        y = genera
        for i, (_, val_idx) in enumerate(skf.split(X, y), 1):
            for j in val_idx:
                sp2fold[unique_species[j]] = i

    df = df.copy()
    df["fold_exp2"] = df["species"].map(sp2fold).astype(int)
    logging.info("Fold_exp2 counts:\n" + df["fold_exp2"].value_counts().sort_index().to_string())
    return df

def make_debug_subset(
    df: pd.DataFrame, n_genera: int, min_species_per_genus: int, target_size: int, seed: int
) -> pd.DataFrame:
    """
    Select N genera with >= min species each, then sample ~target_size rows
    balanced across the chosen genera (and roughly across species).
    """
    require_columns(df, ["genus", "species"], context="debug")
    rng = np.random.default_rng(seed)

    gstats = df.groupby("genus").agg(n_species=("species", "nunique"), n_rows=("species", "size"))
    eligible = gstats[gstats["n_species"] >= min_species_per_genus]
    if eligible.empty:
        logging.warning("No eligible genera for debug subset; skipping.")
        return pd.DataFrame(columns=df.columns)

    chosen = (
        eligible.sort_values(["n_species", "n_rows"], ascending=[False, False])
        .head(max(n_genera * 2, n_genera))
        .index.tolist()
    )
    rng.shuffle(chosen)
    chosen = chosen[:n_genera]

    per_genus = max(target_size // max(1, len(chosen)), 1)
    parts = []
    for g in chosen:
        gdf = df[df["genus"] == g]
        species_order = gdf["species"].value_counts().index.tolist()
        picked_idx = []
        remain = per_genus
        for sp in species_order:
            if remain <= 0:
                break
            sp_rows = gdf[gdf["species"] == sp]
            n_pick = min(len(sp_rows), max(1, remain // 10))
            pick = sp_rows.sample(n=n_pick, random_state=rng.integers(0, 2**31 - 1))
            picked_idx.append(pick)
            remain -= len(pick)
        if remain > 0:
            leftover = gdf.drop(pd.concat(picked_idx).index, errors="ignore") if picked_idx else gdf
            if len(leftover) > 0:
                picked_idx.append(
                    leftover.sample(n=min(remain, len(leftover)), random_state=rng.integers(0, 2**31 - 1))
                )
        if picked_idx:
            parts.append(pd.concat(picked_idx, ignore_index=False))

    if not parts:
        logging.warning("Could not assemble debug subset; skipping.")
        return pd.DataFrame(columns=df.columns)

    debug_df = pd.concat(parts, ignore_index=False).reset_index(drop=True)
    logging.info(
        f"Debug subset assembled: {len(debug_df):,} rows; "
        f"{debug_df['species'].nunique()} species across {len(chosen)} genera"
    )
    return debug_df

def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=NpEncoder)

# -------- Fold mask construction -------- #

def _mask_for_indices(n_classes: int, present: Iterable[int]) -> List[int]:
    mask = [0] * n_classes
    for idx in present:
        if 0 <= idx < n_classes:
            mask[idx] = 1
    return mask

def build_fold_masks(
    df: pd.DataFrame,
    ls: LabelSpace,
    fold_col: str,
) -> Dict:
    """
    Build boolean masks (0/1) per fold and split (train/val), per taxonomy level.
    Masks are aligned with LabelSpace index ordering.
    """
    if fold_col not in df.columns:
        raise ValueError(f"Column '{fold_col}' not found in dataframe.")

    levels = ls.levels

    def label_to_idx(level: str, series: pd.Series) -> List[int]:
        l2i = ls.label_to_idx[level]
        out = []
        for lab in series.astype(str):
            if lab in l2i:
                out.append(l2i[lab])
        return out

    folds_present = sorted(int(x) for x in df[fold_col].unique())
    payload: Dict = {
        "levels": levels,
        "folds": {},
        "label_space_ref": "label_space.json",
        "counts": {"overall": {}, "by_fold": {}},
    }

    # overall counts
    for lvl in levels:
        payload["counts"]["overall"][lvl] = dict(df[lvl].value_counts().astype(int))

    # per fold
    for fold_id in folds_present:
        payload["folds"][str(fold_id)] = {"train": {}, "val": {}}
        payload["counts"]["by_fold"][str(fold_id)] = {"train": {}, "val": {}}

        val_df = df[df[fold_col] == fold_id]
        train_df = df[df[fold_col] != fold_id]

        for split_name, part in (("train", train_df), ("val", val_df)):
            for lvl in levels:
                n_classes = ls.num_classes(lvl)
                idxs = set(label_to_idx(lvl, part[lvl]))
                mask = _mask_for_indices(n_classes, idxs)
                payload["folds"][str(fold_id)][split_name][lvl] = {
                    "mask": mask,
                    "present_indices": sorted(list(idxs)),
                }
                payload["counts"]["by_fold"][str(fold_id)][split_name][lvl] = dict(
                    part[lvl].value_counts().astype(int)
                )

    return payload

# ---------------------------
# Profile processing
# ---------------------------

def process_profile(
    *,
    profile_name: str,
    df_in: pd.DataFrame,
    levels: List[str],
    folds_cfg: Dict,
    folds2_cfg: Dict,
    base_dir: Path,    # experiments/<name>/data
) -> Dict:
    """
    Run the full pipeline for a single profile.
    Returns a summary dict and writes artifacts under base_dir/<profile_name>/...
    """
    assert profile_name in {"full", "debug"}
    pdir = base_dir / profile_name
    pdir.mkdir(parents=True, exist_ok=True)

    # Ensure we don't accidentally reuse old fold columns
    df = df_in.copy()
    for col in ("fold_exp1", "fold_exp2"):
        if col in df.columns:
            df = df.drop(columns=[col])

    logging.info("")
    logging.info("=" * 70)
    logging.info(f"=== PROFILE: {profile_name} ===")
    logging.info("=" * 70)
    logging.info(f"{profile_name}: rows={len(df):,} | unique species={df['species'].nunique():,} | "
                 f"unique genera={df['genus'].nunique():,} | unique families={df['family'].nunique():,}")

    # Assign folds (profile-specific)
    logging.info(f"{profile_name}: assigning folds (exp1: sequence-stratified by {folds_cfg.get('stratify_by','species')})")
    df = assign_folds_sequence_stratified(
        df, k=folds_cfg["k"], stratify_by=folds_cfg.get("stratify_by", "species"), seed=folds_cfg["seed"]
    )
    logging.info(f"{profile_name}: assigning folds (exp2: species-group, stratify by {folds2_cfg.get('stratify_by','genus')})")
    df = assign_folds_species_group(
        df, k=folds2_cfg["k"], stratify_by=folds2_cfg.get("stratify_by", "genus"), seed=folds2_cfg["seed"]
    )

    # Persist prepared dataset for this profile
    prepared_csv = pdir / "prepared.csv"
    df.to_csv(prepared_csv, index=False)
    logging.info(f"{profile_name}: wrote prepared dataset → {prepared_csv} ({len(df):,} rows)")

    # Label space
    ls = LabelSpace.from_csv(str(prepared_csv), levels=levels)
    label_space_path = pdir / "label_space.json"
    ls.to_json(str(label_space_path))
    logging.info(f"{profile_name}: wrote LabelSpace → {label_space_path}")
    logging.info(f"{profile_name}: head spec: {ls.spec()}")

    # Fold masks (exp1 / exp2)
    masks_exp1 = build_fold_masks(df, ls, fold_col="fold_exp1")
    fold_masks_exp1_path = pdir / "fold_masks_exp1.json"
    write_json(fold_masks_exp1_path, masks_exp1)
    logging.info(f"{profile_name}: wrote fold masks (exp1) → {fold_masks_exp1_path}")

    masks_exp2 = build_fold_masks(df, ls, fold_col="fold_exp2")
    fold_masks_exp2_path = pdir / "fold_masks_exp2.json"
    write_json(fold_masks_exp2_path, masks_exp2)
    logging.info(f"{profile_name}: wrote fold masks (exp2) → {fold_masks_exp2_path}")

    # Small profile summary
    summary = {
        "profile": profile_name,
        "paths": {
            "dir": str(pdir),
            "prepared_csv": str(prepared_csv),
            "label_space_json": str(label_space_path),
            "fold_masks_exp1_json": str(fold_masks_exp1_path),
            "fold_masks_exp2_json": str(fold_masks_exp2_path),
        },
        "rows": int(len(df)),
        "unique_species": int(df["species"].nunique()),
        "unique_genera": int(df["genus"].nunique()),
        "unique_families": int(df["family"].nunique()),
        "folds": {
            "exp1": {"k": int(folds_cfg["k"]), "stratify_by": folds_cfg.get("stratify_by", "species")},
            "exp2": {"k": int(folds2_cfg["k"]), "stratify_by": folds2_cfg.get("stratify_by", "genus")},
        },
        "label_space": {"levels": levels, "head_spec": ls.spec()},
    }
    profile_summary_path = pdir / "summary.json"
    write_json(profile_summary_path, summary)
    logging.info(f"{profile_name}: wrote profile summary → {profile_summary_path}")

    return summary

# ---------------------------
# Main
# ---------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to master experiment YAML")
    args = parser.parse_args()

    # Load env + config
    root = Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)

    # Resolve env-backed roots
    experiments_root = Path(env_expand(cfg["data"]["experiments_root"]))
    input_dir = Path(env_expand(cfg["data"]["input_dir"]))
    exp_name = cfg["experiment"]["name"]

    # Paths
    base_data_dir = Path(
        cfg["artifacts"]["paths"]["prepared_dir"].format(
            experiments_root=str(experiments_root),
            experiment_name=exp_name,
        )
    )
    logs_dir = Path(
        cfg["artifacts"]["paths"]["logs_dir"].format(
            experiments_root=str(experiments_root),
            experiment_name=exp_name,
        )
    )
    log_path = Path(
        cfg["artifacts"]["paths"]["prep_log"].format(
            experiments_root=str(experiments_root),
            experiment_name=exp_name,
        )
    )
    top_summary_path = Path(
        cfg["artifacts"]["paths"]["prep_summary_json"].format(
            experiments_root=str(experiments_root),
            experiment_name=exp_name,
            prep_summary_filename=cfg["data"]["filenames"]["prep_summary"],
        )
    )

    base_data_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_path)

    # Echo config useful for prep
    logging.info(f"Experiment: {exp_name}")
    logging.info(f"Union: {cfg['experiment']['union']}")
    logging.info(f"Profile (requested): {cfg['experiment'].get('profile', 'full')}")
    logging.info(f"Input dir: {input_dir}")
    logging.info(f"Base data dir: {base_data_dir}")

    # Build union
    union_key = cfg["experiment"]["union"]
    members = cfg["data"]["inputs"][union_key]
    df = read_union_csvs(input_dir, members)

    # Clean + uppercase
    if cfg["prep"].get("uppercase_sequences", True):
        df = apply_uppercase(df)

    # Filter by species resolution
    df = filter_by_resolution(df, cfg["prep"]["min_species_resolution"])

    # Deduplicate within species by sequence content
    keep_policy = cfg["prep"]["dedupe"].get("keep_policy", "highest_resolution")
    df = dedupe_within_species_by_sequence(df, keep_policy=keep_policy)

    # --- Derive profiles ---
    levels = cfg["label_space"]["levels"]
    folds_cfg  = cfg["prep"]["folds"]
    folds2_cfg = cfg["prep"]["folds_species_group"]

    # full profile
    df_full = df.copy()

    # debug profile (optional)
    debug_cfg = cfg["prep"]["debug_subset"]
    df_debug = pd.DataFrame(columns=df.columns)
    debug_enabled = bool(debug_cfg.get("enabled", False))
    if debug_enabled:
        df_debug = make_debug_subset(
            df_full,
            n_genera=debug_cfg["n_genera"],
            min_species_per_genus=debug_cfg["min_species_per_genus"],
            target_size=debug_cfg["target_size"],
            seed=debug_cfg["seed"],
        )
        if len(df_debug) == 0:
            logging.warning("Debug subset was requested but empty; skipping debug profile.")
            debug_enabled = False

    # --- Process profiles symmetrically ---
    summaries: Dict[str, Dict] = {}

    summaries["full"] = process_profile(
        profile_name="full",
        df_in=df_full,
        levels=levels,
        folds_cfg=folds_cfg,
        folds2_cfg=folds2_cfg,
        base_dir=base_data_dir,
    )

    if debug_enabled:
        summaries["debug"] = process_profile(
            profile_name="debug",
            df_in=df_debug,
            levels=levels,
            folds_cfg=folds_cfg,
            folds2_cfg=folds2_cfg,
            base_dir=base_data_dir,
        )

    # --- Top-level summary file ---
    top_summary = {
        "experiment": {
            "name": exp_name,
            "union": union_key,
            "seed": cfg["experiment"]["seed"],
        },
        "input": {"dir": str(input_dir), "members": members},
        "filters": {
            "uppercase_sequences": bool(cfg["prep"].get("uppercase_sequences", True)),
            "min_species_resolution": int(cfg["prep"]["min_species_resolution"]),
        },
        "dedupe": {
            "scope": cfg["prep"]["dedupe"]["scope"],
            "keep_policy": keep_policy,
        },
        "label_space": {"levels": levels},
        "profiles": summaries,  # contains per-profile paths & stats
        "artifacts_root": str(base_data_dir),
        "log_path": str(log_path),
    }
    write_json(top_summary_path, top_summary)
    logging.info(f"Wrote top-level prep summary → {top_summary_path}")
    logging.info("✓ Experiment prep complete.")

if __name__ == "__main__":
    main()

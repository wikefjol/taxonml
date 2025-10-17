#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
logger = logging.getLogger(__name__)
"""
exp_setup.py
... to be written later
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

# Local lib
from taxml.core.logging import setup_logging, attach_file_logger
from taxml.core.config import load_config
from taxml.labels.space import LabelSpace

# ---------------------------
# Helpers
# ---------------------------

REQUIRED_COLS = [
    "sequence", "kingdom", "phylum", "class", "order", "family", "genus", "species", "species_resolution"
]

# def setup_logging(log_path: Path) -> None:
#     
#     logging.basicConfig( # set basic config
#         level=logging.INFO,
#         format="%(asctime)s | %(levelname)s | %(message)s",
#         handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
#     )
#     logger.info("Logging initialized.") # pr

def _collect_outputs_for_overwrite(paths) -> list[Path]:
    # everything prep writes
    targets = [
        paths["prep_summary_json"],
        *paths["prepared_csv"].values(),
        *paths["label_space_json"].values(),
        paths["fold_masks"]["exp1"]["full"],
        paths["fold_masks"]["exp1"]["debug"],
        paths["fold_masks"]["exp2"]["full"],
        paths["fold_masks"]["exp2"]["debug"],
    ]
    return list(dict.fromkeys(targets))

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
        logger.info(f"Loaded {len(df):,} rows from {csv_path.name}")
    union_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Union total rows: {len(union_df):,}")
    return union_df

def apply_uppercase(df: pd.DataFrame) -> pd.DataFrame:
    if "sequence" in df.columns:
        n_nan = df["sequence"].isna().sum()
        if n_nan:
            logger.warning(f"{n_nan:,} rows have NaN sequence; dropping them.")
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
    
def _rank_count_table(df_before: pd.DataFrame, df_after: pd.DataFrame) -> str:
    # Only include ranks present in the dataframe(s)
    ranks_all = ["kingdom","phylum","class","order","family","genus","species"]
    ranks = [r for r in ranks_all if r in df_before.columns and r in df_after.columns]

    def counts(df: pd.DataFrame) -> Dict[str, int]:
        d = {r: int(df[r].nunique()) for r in ranks}
        d["rows"] = int(len(df))
        return d

    b = counts(df_before)
    a = counts(df_after)
    tbl = pd.DataFrame({"before": b, "after": a})
    tbl["delta"] = tbl["after"] - tbl["before"]
    # Order rows: high→low taxonomy, then rows last
    idx = ranks + ["rows"]
    tbl = tbl.loc[idx]
    return "\n" + tbl.to_string()

def filter_species_min_count(df: pd.DataFrame, n: int, species_col: str = "species") -> pd.DataFrame:
    require_columns(df, [species_col], context="filter_species_min_count")
    before_df = df
    sp_counts = df[species_col].value_counts()
    keep_sp = sp_counts[sp_counts >= n].index
    out = df[df[species_col].isin(keep_sp)].copy()

    dropped_sp = int((sp_counts < n).sum())
    logger.info(
        f"Filtered species with < {n} sequences "
        f"(dropped {dropped_sp:,} species; rows {len(before_df):,}->{len(out):,})"
    )
    logger.info("Rank counts before/after (unique labels & rows):" + _rank_count_table(before_df, out))
    return out


def filter_by_resolution(df: pd.DataFrame, min_resolution: int) -> pd.DataFrame:
    before = len(df)
    df = df[df["species_resolution"] > min_resolution].copy()
    logger.info(f"Filtered by species_resolution > {min_resolution}: {before:,} -> {len(df):,}")
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
    logger.info(f"Dedup within species by sequence: removed {before - after:,} duplicates ({before:,} -> {after:,})")
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
    logger.info("Fold_exp1 counts:\n" + df["fold_exp1"].value_counts().sort_index().to_string())
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
        logger.warning("Insufficient stratification support; falling back to random split over species.")
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
    logger.info("Fold_exp2 counts:\n" + df["fold_exp2"].value_counts().sort_index().to_string())
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
        logger.warning("No eligible genera for debug subset; skipping.")
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
        logger.warning("Could not assemble debug subset; skipping.")
        return pd.DataFrame(columns=df.columns)

    debug_df = pd.concat(parts, ignore_index=False).reset_index(drop=True)
    logger.info(
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

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"=== PROFILE: {profile_name} ===")
    logger.info("=" * 70)
    logger.info(f"{profile_name}: rows={len(df):,} | unique species={df['species'].nunique():,} | "
                 f"unique genera={df['genus'].nunique():,} | unique families={df['family'].nunique():,}")

    # Assign folds (profile-specific)
    logger.info(f"{profile_name}: assigning folds (exp1: sequence-stratified by {folds_cfg.get('stratify_by','species')})")
    df = assign_folds_sequence_stratified(
        df, k=folds_cfg["k"], stratify_by=folds_cfg.get("stratify_by", "species"), seed=folds_cfg["seed"]
    )
    logger.info(f"{profile_name}: assigning folds (exp2: species-group, stratify by {folds2_cfg.get('stratify_by','genus')})")
    df = assign_folds_species_group(
        df, k=folds2_cfg["k"], stratify_by=folds2_cfg.get("stratify_by", "genus"), seed=folds2_cfg["seed"]
    )

    # Persist prepared dataset for this profile
    prepared_csv = pdir / "prepared.csv"
    df.to_csv(prepared_csv, index=False)
    logger.info(f"{profile_name}: wrote prepared dataset → {prepared_csv} ({len(df):,} rows)")

    # Label space
    ls = LabelSpace.from_csv(str(prepared_csv), levels=levels)
    label_space_path = pdir / "label_space.json"
    ls.to_json(str(label_space_path))
    logger.info(f"{profile_name}: wrote LabelSpace → {label_space_path}")
    logger.info(f"{profile_name}: head spec: {ls.spec()}")

    # Fold masks (exp1 / exp2)
    masks_exp1 = build_fold_masks(df, ls, fold_col="fold_exp1")
    fold_masks_exp1_path = pdir / "fold_masks_exp1.json"
    write_json(fold_masks_exp1_path, masks_exp1)
    logger.info(f"{profile_name}: wrote fold masks (exp1) → {fold_masks_exp1_path}")

    masks_exp2 = build_fold_masks(df, ls, fold_col="fold_exp2")
    fold_masks_exp2_path = pdir / "fold_masks_exp2.json"
    write_json(fold_masks_exp2_path, masks_exp2)
    logger.info(f"{profile_name}: wrote fold masks (exp2) → {fold_masks_exp2_path}")

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
    logger.info(f"{profile_name}: wrote profile summary → {profile_summary_path}")

    return summary

# ---------------------------
# Main
# ---------------------------

def main() -> None:
    setup_logging(
        console_level=logging.INFO,
        buffer_early=True
        )
    
    logger.info("=== Prep startup ===")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to master experiment YAML")
    parser.add_argument("--yes", action="store_true", help="Overwrite existing prep artifacts without prompting")
    args = parser.parse_args()

    cfg = load_config(mode = "prep", config_path = args.config,)
    
    paths = cfg["paths"]  # alias for readability

    # attach file logger to the resolved prep log
    attach_file_logger(
        file_path = paths["prep_log"],
        file_level = logging.DEBUG,
        flush_buffer = True
        )    
    
    # Overwrite guard
    existing = [p for p in _collect_outputs_for_overwrite(paths) if Path(p).exists()]
    if existing and not args.yes:
        logger.warning("About to overwrite existing prep artifacts under %s", paths["data_root"])
        resp = input("Continue? Type 'yes' to proceed: ").strip().lower()
        if resp != "yes":
            logger.info("Aborted by user.")
            return

    # Echo config essentials
    logger.info("Experiment: %s", cfg["experiment"]["name"])
    logger.info("Input dir: %s", cfg["data"]["input_dir"])
    logger.info("Data root: %s", paths["data_root"])
    
    # === Ingest & clean ===
    union_key = cfg["experiment"]["union"]
    members   = cfg["data"]["inputs"][union_key]
    df = read_union_csvs(cfg["data"]["input_dir"], members)

    if cfg["prep"]["uppercase_sequences"]:
        df = apply_uppercase(df)

    # === Filtering & Deduping ===
    # Filter by species resolution
    df = filter_by_resolution(
        df = df,
        min_resolution = cfg["prep"]["min_species_resolution"],
        )

    # Deduplicate within species by sequence content
    df = dedupe_within_species_by_sequence(
        df = df,
        keep_policy=cfg["prep"]["dedupe"].get("keep_policy", "highest_resolution"),
        )

    min_sp = cfg["prep"].get("min_sequences_per_species", 0)
    if min_sp and min_sp > 0:
        df = filter_species_min_count(df, n=min_sp)
    
    # === Prepare Debug Subset ===
    dbg = cfg["prep"]["debug_subset"]
    df_debug = make_debug_subset(
        df,
        dbg["n_genera"],
        dbg["min_species_per_genus"],
        dbg["target_size"],
        dbg["seed"]
    )
   
    if len(df_debug) == 0:
        logging.warning("Debug subset empty; writing empty profile directory & summary.")
   
    # === Build profiles ===
    levels    = cfg["label_space"]["levels"]
    folds1    = cfg["prep"]["folds"]
    folds2    = cfg["prep"]["folds_species_group"]

    # full profile
    summary_full = process_profile(
        profile_name="full",
        df_in=df,
        levels=levels,
        folds_cfg=folds1,
        folds2_cfg=folds2,
        base_dir=paths["data_root"],   # uses the centralised data root
    )
    # debug profile
    summary_debug = process_profile(
        profile_name="debug",
        df_in=df_debug if len(df_debug) else df.head(0),  # emit empty files with headers
        levels=levels,
        folds_cfg=folds1,
        folds2_cfg=folds2,
        base_dir=paths["data_root"],
    )
    
    # === Produce summary file ===
    top_summary = {
        "experiment": cfg["experiment"],
        "input": {"dir": str(cfg["data"]["input_dir"]), "members": members},
        "filters": {
            "uppercase_sequences": cfg["prep"]["uppercase_sequences"],
            "min_species_resolution": cfg["prep"]["min_species_resolution"],
        },
        "dedupe": cfg["prep"]["dedupe"],
        "label_space": {"levels": levels},
        "profiles": {"full": summary_full, "debug": summary_debug},
        "artifacts_root": str(paths["data_root"]),
        "log_path": str(paths["prep_log"]),
    }
    write_json(paths["prep_summary_json"], top_summary)
    logger.info("Wrote top-level prep summary -> %s", paths["prep_summary_json"])
    logger.info("✓ Experiment prep complete.")

if __name__ == "__main__":
    main()
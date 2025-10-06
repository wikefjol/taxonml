# src/taxml/core/paths.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional


def ranks_code(levels: List[str]) -> str:
    """
    ['phylum','class','order','family','genus','species'] -> 'ranks_6_pcofgs'
    """
    abbrev = "".join(l.strip().lower()[0] for l in levels)
    return f"ranks_{len(levels)}_{abbrev}"


def render_paths(
    *,
    experiments_root: str | Path,
    experiment_name: str,
    # Optional extras (used by training modes)
    pretrained_root: str | Path | None = None,
    arch_id: Optional[str] = None,
    profile: Optional[str] = None,             # "full" | "debug"
    fold: Optional[int] = None,
    levels: Optional[List[str]] = None,
    fold_scheme: Optional[str] = None,         # "exp1" | "exp2"
) -> Dict:
    """
    Produce the complete path map for this experiment. Callers (loaders) can pick the
    subset they need for a particular mode.

    Always emitted:
      - exp_root, logs_dir, prep_log
      - data_root
      - profile_dirs: {"full","debug"}
      - prepared_csv, label_space_json (per-profile maps)
      - fold_masks: {"exp1":{"full","debug"}, "exp2":{"full","debug"}}
      - prep_summary_json
      - pretrained_root (if provided)

    Conditionally emitted when arch/profile provided:
      - pretrain: {arch_root, run_dir, checkpoints_dir, history_file, results_file, log_file}
    Conditionally emitted when arch/profile/fold/levels provided:
      - finetune: {arch_root, fold_dir, task, run_dir, checkpoints_dir, history_file, results_file, log_file,
                   fold_masks_path}
    """
    exp_root = Path(experiments_root).expanduser().resolve() / experiment_name
    logs_dir = exp_root / "logs"
    data_root = exp_root / "data"

    def _per_profile(rel: str) -> Dict[str, Path]:
        return {
            "full": data_root / "full" / rel,
            "debug": data_root / "debug" / rel,
        }

    out: Dict = {
        "exp_root": exp_root,
        "logs_dir": logs_dir,
        "prep_log": logs_dir / "prep.log",
        "data_root": data_root,
        "profile_dirs": {"full": data_root / "full", "debug": data_root / "debug"},
        "prepared_csv": _per_profile("prepared.csv"),
        "label_space_json": _per_profile("label_space.json"),
        "fold_masks": {
            "exp1": _per_profile("fold_masks_exp1.json"),
            "exp2": _per_profile("fold_masks_exp2.json"),
        },
        "prep_summary_json": data_root / "prep_summary.json",
    }
    if pretrained_root is not None:
        out["pretrained_root"] = Path(pretrained_root).expanduser().resolve()

    # --- Optional: pretrain run family (needs arch_id+profile) ---
    if arch_id and profile:
        arch_root = exp_root / arch_id / profile
        pre_run = arch_root / "pretrain"
        out["pretrain"] = {
            "arch_root": arch_root,
            "run_dir": pre_run,
            "checkpoints_dir": pre_run / "checkpoints",
            "history_file": pre_run / "history.json",
            "results_file": pre_run / "results.json",
            "log_file": pre_run / "pretrain.log",
        }

    # --- Optional: finetune run family (needs arch_id+profile+fold+levels) ---
    if arch_id and profile and (fold is not None) and levels:
        arch_root = exp_root / arch_id / profile
        task = ranks_code(levels)
        fold_dir = arch_root / "folds" / f"fold_{int(fold):02d}"
        run_dir = fold_dir / task
        # pick masks path by scheme if provided; else leave to loader to choose
        fm_path = None
        if fold_scheme in {"exp1", "exp2"}:
            fm_path = out["fold_masks"][fold_scheme][profile]

        out["finetune"] = {
            "arch_root": arch_root,
            "fold_dir": fold_dir,
            "task": task,
            "run_dir": run_dir,
            "checkpoints_dir": run_dir / "checkpoints",
            "history_file": run_dir / "history.json",
            "results_file": run_dir / "results.json",
            "log_file": run_dir / "train.log",
            "fold_masks_path": fm_path,  # may be None if no scheme given
        }

    return out


# --- Minimal, backwards-compatible helper for current finetune/pretrain scripts ---
def build_profile_paths(experiments_root: str | Path, experiment_name: str, profile: str) -> Dict[str, Path]:
    """
    Compatibility shim (used by your current training scripts). Returns the
    small profile-scoped view needed there.
    """
    p = render_paths(experiments_root=experiments_root, experiment_name=experiment_name)
    return {
        "exp_root": p["exp_root"],
        "data_dir": p["profile_dirs"][profile],
        "prepared_csv": p["prepared_csv"][profile],
        "label_space_json": p["label_space_json"][profile],
        "summary_json": p["profile_dirs"][profile] / "summary.json",
        "fold_masks_exp1": p["fold_masks"]["exp1"][profile],
        "fold_masks_exp2": p["fold_masks"]["exp2"][profile],
        "logs_dir": p["logs_dir"],
    }

# src/taxml/core/config.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from .paths import render_paths, ranks_code  # also exports build_profile_paths for compat
from .io import read_yaml  # project-local YAML reader (no env expansion)


# -----------------------
# Small utils
# -----------------------
def _require(d: Dict, key: str, ctx: str = "root"):
    if key not in d:
        raise KeyError(f"Missing required key: {key} (in {ctx})")
    return d[key]


def _as_path(x: str | Path) -> Path:
    return Path(x).expanduser().resolve()

def _deep_merge(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base

def _derive_arch_id(encoder: Dict[str, Any], optimal_length: int) -> str:
    """
    Deterministic arch string; lives in core to avoid depending on encoders/*.
    Format: bert_h{H}_L{L}_H{A}_i{I}_P{MPE}
    """
    h = int(_require(encoder, "hidden_size", "model.encoder"))
    L = int(_require(encoder, "num_hidden_layers", "model.encoder"))
    A = int(_require(encoder, "num_attention_heads", "model.encoder"))
    I = int(_require(encoder, "intermediate_size", "model.encoder"))
    mpe = int(optimal_length + 2)
    return f"bert_h{h}_L{L}_H{A}_i{I}_P{mpe:03d}"


def _fail_if_missing_files(paths: List[Path], label: str) -> None:
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"{label}: missing required artifacts:\n" + "\n".join(missing))


# -----------------------
# Public API
# -----------------------
def load_config(
    *,
    mode: Literal["prep", "pretrain", "finetune"],
    config_path: str | Path,
    debug: bool = False,
    levels: Optional[List[str]] = None,
    fold_index: Optional[int] = None,
    fold_scheme: Optional[Literal["exp1", "exp2"]] = None,
) -> Dict[str, Any]:
    """
    KISS config loader with explicit fail-fast behavior.
    - For prep: ignores --debug; returns paths for both full & debug.
    - For pretrain/finetune: if debug=True, deep-merge `debug_overrides` into base
      BEFORE computing derived fields (optimal_length, MPE, arch_id) and paths.
    - Validates required artifacts exist for training modes.
    - No CLI overrides other than `--debug`; no silent defaults.
    """
    import copy

    raw = read_yaml(_as_path(config_path))

    # ---------- Common required ----------
    exp = _require(raw, "experiment", "root.experiment")
    exp_name = str(_require(exp, "name", "experiment"))
    seed = int(_require(exp, "seed", "experiment"))

    data = _require(raw, "data", "root.data")
    experiments_root = _as_path(_require(data, "experiments_root", "data"))

    # Directory scaffold (shared)
    exp_root = experiments_root / exp_name
    logs_dir = exp_root / "logs"
    data_root = exp_root / "data"
    profile_dirs = {
        "full": data_root / "full",
        "debug": data_root / "debug",
    }
    files_by_profile = {
        "prepared_csv": {
            "full":  profile_dirs["full"]  / "prepared.csv",
            "debug": profile_dirs["debug"] / "prepared.csv",
        },
        "label_space_json": {
            "full":  profile_dirs["full"]  / "label_space.json",
            "debug": profile_dirs["debug"] / "label_space.json",
        },
        "fold_masks": {
            "exp1": {
                "full":  profile_dirs["full"]  / "fold_masks_exp1.json",
                "debug": profile_dirs["debug"] / "fold_masks_exp1.json",
            },
            "exp2": {
                "full":  profile_dirs["full"]  / "fold_masks_exp2.json",
                "debug": profile_dirs["debug"] / "fold_masks_exp2.json",
            },
        },
    }

    out: Dict[str, Any] = {
        "experiment": {"name": exp_name, "seed": seed},
        "data": {"experiments_root": experiments_root},
        "runtime": {"debug": bool(debug), "fold_index": fold_index, "fold_scheme": fold_scheme},
        "paths": {
            "exp_root": exp_root,
            "logs_dir": logs_dir,
            "prep_log": logs_dir / "prep.log",
            "data_root": data_root,
            "profile_dirs": profile_dirs,
            "prepared_csv": files_by_profile["prepared_csv"],
            "label_space_json": files_by_profile["label_space_json"],
            "fold_masks": files_by_profile["fold_masks"],
            "prep_summary_json": data_root / "prep_summary.json",
        },
    }

    # ---------- Mode: PREP ----------
    if mode == "prep":
        union = _require(exp, "union", "experiment")
        input_dir = _as_path(_require(data, "input_dir", "data"))
        filenames = _require(data, "filenames", "data")
        label_space = _require(raw, "label_space", "root")
        levels_cfg = list(_require(label_space, "levels", "label_space"))
        prep = _require(raw, "prep", "root")

        out["experiment"]["union"] = str(union)
        out["data"].update({
            "input_dir": input_dir,
            "inputs": _require(data, "inputs", "data"),
            "filenames": filenames,
        })
        out["label_space"] = {"levels": levels_cfg}
        out["prep"] = {
            "uppercase_sequences": bool(_require(prep, "uppercase_sequences", "prep")),
            "min_species_resolution": int(_require(prep, "min_species_resolution", "prep")),
            "dedupe": {
                "scope": str(_require(_require(prep, "dedupe", "prep"), "scope", "prep.dedupe")),
                "keep_policy": str(_require(_require(prep, "dedupe", "prep"), "keep_policy", "prep.dedupe")),
            },
            "folds": {
                "k": int(_require(_require(prep, "folds", "prep"), "k", "prep.folds")),
                "stratify_by": str(_require(_require(prep, "folds", "prep"), "stratify_by", "prep.folds")),
                "seed": int(_require(_require(prep, "folds", "prep"), "seed", "prep.folds")),
            },
            "folds_species_group": {
                "k": int(_require(_require(prep, "folds_species_group", "prep"), "k", "prep.folds_species_group")),
                "stratify_by": str(_require(_require(prep, "folds_species_group", "prep"), "stratify_by", "prep.folds_species_group")),
                "seed": int(_require(_require(prep, "folds_species_group", "prep"), "seed", "prep.folds_species_group")),
            },
            "debug_subset": {
                "n_genera": int(_require(_require(prep, "debug_subset", "prep"), "n_genera", "prep.debug_subset")),
                "min_species_per_genus": int(_require(_require(prep, "debug_subset", "prep"), "min_species_per_genus", "prep.debug_subset")),
                "target_size": int(_require(_require(prep, "debug_subset", "prep"), "target_size", "prep.debug_subset")),
                "seed": int(_require(_require(prep, "debug_subset", "prep"), "seed", "prep.debug_subset")),
            },
        }
        return out

    # ---------- Modes: PRETRAIN / FINETUNE ----------
    # 1) Effective config (apply overrides first if debug)
    effective = copy.deepcopy(raw)
    if debug:
        dbg = raw.get("debug_overrides")
        if isinstance(dbg, dict):
            _deep_merge(effective, dbg)

    # 2) Pull training-time blocks from effective config
    preproc = _require(effective, "preprocessing", "root")
    alphabet = list(_require(preproc, "alphabet", "preprocessing"))
    k = int(_require(preproc, "k", "preprocessing"))
    max_bases = int(_require(preproc, "max_bases", "preprocessing"))
    if max_bases % k != 0:
        raise ValueError("preprocessing.max_bases must be divisible by preprocessing.k")
    mod_prob = float(_require(preproc, "mod_prob", "preprocessing"))

    optimal_length = max_bases // k
    max_position_embeddings = optimal_length + 2

    model = _require(effective, "model", "root")
    encoder = dict(_require(model, "encoder", "model"))

    # Deterministic arch id derived AFTER overrides
    arch_id = _derive_arch_id(encoder, optimal_length)

    # 3) Scheduler catalog (pass-through)
    scheduler_catalog = dict(effective.get("scheduler_catalog", {}))

    # 4) Select active profile (used only for path building)
    profile_dirname = "debug" if debug else "full"

    # 5) Data artifacts for the active profile
    active_prepared = files_by_profile["prepared_csv"][profile_dirname]
    active_label_space = files_by_profile["label_space_json"][profile_dirname]

    # 6) Common training returns
    out.update({
        "preprocessing": {
            "alphabet": alphabet,
            "k": k,
            "max_bases": max_bases,
            "optimal_length": optimal_length,
            "max_position_embeddings": max_position_embeddings,
            "mod_prob": mod_prob,
        },
        "model": {"encoder": encoder},
        "arch": {"id": arch_id},
        "scheduler_catalog": scheduler_catalog,
    })

    # Small helper: ranks_code for finetune run directory
    def _ranks_code(level_list: List[str]) -> str:
        abbr = "".join(l.strip().lower()[0] for l in level_list)
        return f"ranks_{len(level_list)}_{abbr}"

    # ---------- PRETRAIN ----------
    if mode == "pretrain":
        tasks = _require(effective, "tasks", "root")
        pt_cfg = _require(tasks, "pretrain", "tasks.pretrain")

        # Promotion root is required (no fallback) but we don't create it here
        promo = _require(pt_cfg, "promotion", "tasks.pretrain")
        pretrained_root = _as_path(_require(promo, "pretrained_root", "tasks.pretrain.promotion"))

        # Paths for this run
        arch_root = exp_root / arch_id / profile_dirname
        run_dir = arch_root / "pretrain"
        pretrain_paths = {
            "arch_root": arch_root,
            "run_dir": run_dir,
            "checkpoints_dir": run_dir / "checkpoints",
            "history_file": run_dir / "history.json",
            "results_file": run_dir / "results.json",
            "log_file": run_dir / "pretrain.log",
        }
        promotion_best = pretrained_root / arch_id / "best.pt"

        # Validate artifacts exist (prepared/labels + exp1 masks in selected profile)
        masks_path = files_by_profile["fold_masks"]["exp1"][profile_dirname]
        _fail_if_missing_files([active_prepared, active_label_space, masks_path], "pretrain")

        out.update({
            "pretrain": {
                "mlm": _require(pt_cfg, "mlm", "tasks.pretrain"),
                "dataloader": _require(pt_cfg, "dataloader", "tasks.pretrain"),
                "optimizer": _require(pt_cfg, "optimizer", "tasks.pretrain"),
                "scheduler": _require(pt_cfg, "scheduler", "tasks.pretrain"),
                "trainer": _require(pt_cfg, "trainer", "tasks.pretrain"),
                "promotion": {"pretrained_root": pretrained_root},
            },
            "paths_active": {
                "data": {
                    "prepared_csv": active_prepared,
                    "label_space_json": active_label_space,
                    "fold_masks": masks_path,
                },
                "pretrain": pretrain_paths,
                "logs_dir": logs_dir,
                "pretrained_root": pretrained_root,
                "promotion_best": promotion_best,
            },
        })
        return out

    # ---------- FINETUNE ----------
    if mode == "finetune":
        if not levels:
            raise ValueError("finetune: --levels (CLI) is required.")
        if fold_index is None:
            raise ValueError("finetune: --fold (CLI) is required.")
        if fold_scheme is None:
            fold_scheme = "exp1"  # agreed single default

        tasks = _require(effective, "tasks", "root")
        ft_cfg = _require(tasks, "finetune", "tasks.finetune")

        backbone = _require(ft_cfg, "backbone", "tasks.finetune")
        backbone_path = _as_path(_require(backbone, "path", "tasks.finetune.backbone"))
        if not backbone_path.exists():
            raise FileNotFoundError(f"finetune.backbone.path not found: {backbone_path}")

        # Validate artifacts exist (prepared/labels + chosen scheme masks)
        masks_path = files_by_profile["fold_masks"][fold_scheme][profile_dirname]
        _fail_if_missing_files([active_prepared, active_label_space, masks_path], "finetune")

        # Paths for this run
        arch_root = exp_root / arch_id / profile_dirname
        fold_dir = arch_root / "folds" / f"fold_{int(fold_index):02d}"
        run_dir = fold_dir / _ranks_code(list(levels))
        finetune_paths = {
            "arch_root": arch_root,
            "fold_dir": fold_dir,
            "run_dir": run_dir,
            "checkpoints_dir": run_dir / "checkpoints",
            "history_file": run_dir / "history.json",
            "results_file": run_dir / "results.json",
            "log_file": run_dir / "train.log",
        }

        out["runtime"]["fold_scheme"] = fold_scheme
        out["runtime"]["levels"] = list(levels)

        out.update({
            "finetune": {
                "levels": list(levels),
                "fold_scheme": fold_scheme,
                "fold_index": int(fold_index),
                "head": _require(ft_cfg, "head", "tasks.finetune"),
                "backbone": {"path": backbone_path, "strict": bool(_require(backbone, "strict", "tasks.finetune.backbone"))},
                "dataloader": _require(ft_cfg, "dataloader", "tasks.finetune"),
                "optimizer": _require(ft_cfg, "optimizer", "tasks.finetune"),
                "scheduler": _require(ft_cfg, "scheduler", "tasks.finetune"),
                "trainer": _require(ft_cfg, "trainer", "tasks.finetune"),
            },
            "paths_active": {
                "data": {
                    "prepared_csv": active_prepared,
                    "label_space_json": active_label_space,
                    "fold_masks": masks_path,
                },
                "finetune": finetune_paths,
                "logs_dir": logs_dir,
            },
        })
        return out

    # Should not reach
    raise ValueError(f"Unknown mode: {mode}")
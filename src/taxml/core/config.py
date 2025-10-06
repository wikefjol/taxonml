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
    # runtime-only knobs (not in YAML)
    profile: Optional[Literal["full", "debug"]] = None,  # if None → "full"
    debug: Optional[bool] = None,                        # if True → profile="debug"
    levels: Optional[List[str]] = None,                  # required for finetune
    fold_index: Optional[int] = None,                    # required for pretrain & finetune
    fold_scheme: Optional[Literal["exp1", "exp2"]] = None,  # finetune: default to "exp1"
) -> Dict[str, Any]:
    """
    Single entry-point config loader. Returns a *flat runtime dict* per mode and
    **fails fast** on missing/invalid inputs or artifacts.

    Contract highlights:
      - No env-var expansion; only absolute paths (we allow '~' expansion).
      - For training modes, verifies presence of prepared.csv, label_space.json,
        and chosen fold_masks file for the active profile.
      - Pretrained backbone path must exist for finetune (no fallbacks).
      - Pretrain promotion root is passed through but not created here.
    """
    raw = read_yaml(_as_path(config_path))

    # ===== Common required =====
    exp = _require(raw, "experiment", "root.experiment")
    name = _require(exp, "name", "experiment")
    seed = int(_require(exp, "seed", "experiment"))

    data = _require(raw, "data", "root.data")
    experiments_root = _as_path(_require(data, "experiments_root", "data"))

    # Profile resolution (runtime only)
    if profile and debug is True and profile != "debug":
        raise ValueError("Conflicting runtime inputs: profile!=debug.")
    if profile is None:
        profile = "debug" if debug else "full"
    if profile not in {"full", "debug"}:
        raise ValueError(f"Invalid profile: {profile!r} (expected 'full'|'debug')")

    # Prepare output scaffold
    out: Dict[str, Any] = {
        "experiment": {"name": str(name), "seed": seed},
        "data": {"experiments_root": experiments_root},
        "runtime": {"profile": profile},
    }

    # We always render the base path map so callers can introspect as needed.
    # Pretrained root (if present) is injected below in pretrain/fine branches.
    base_paths = render_paths(experiments_root=experiments_root, experiment_name=name)

    # ===== Mode: prep =====
    if mode == "prep":
        # Additional required only for prep
        union = _require(exp, "union", "experiment")
        input_dir = _as_path(_require(data, "input_dir", "data"))
        filenames = _require(data, "filenames", "data")
        label_space = _require(raw, "label_space", "root")
        levels = list(_require(label_space, "levels", "label_space"))
        prep = _require(raw, "prep", "root")

        # Return exactly the contract you’re already consuming
        out.update(
            {
                "experiment": {"name": str(name), "seed": seed, "union": str(union)},
                "data": {
                    "experiments_root": experiments_root,
                    "input_dir": input_dir,
                    "inputs": _require(data, "inputs", "data"),
                    "filenames": filenames,
                },
                "label_space": {"levels": levels},
                "prep": {
                    "uppercase_sequences": bool(prep.get("uppercase_sequences", True)),
                    "min_species_resolution": int(prep.get("min_species_resolution", 0)),
                    "dedupe": {
                        "scope": str(_require(prep, "dedupe", "prep")["scope"]),
                        "keep_policy": str(_require(prep, "dedupe", "prep")["keep_policy"]),
                    },
                    "folds": {
                        "k": int(_require(prep, "folds", "prep")["k"]),
                        "stratify_by": str(_require(prep, "folds", "prep")["stratify_by"]),
                        "seed": int(_require(prep, "folds", "prep")["seed"]),
                    },
                    "folds_species_group": {
                        "k": int(_require(prep, "folds_species_group", "prep")["k"]),
                        "stratify_by": str(_require(prep, "folds_species_group", "prep")["stratify_by"]),
                        "seed": int(_require(prep, "folds_species_group", "prep")["seed"]),
                    },
                    "debug_subset": {
                        "n_genera": int(_require(prep, "debug_subset", "prep")["n_genera"]),
                        "min_species_per_genus": int(_require(prep, "debug_subset", "prep")["min_species_per_genus"]),
                        "target_size": int(_require(prep, "debug_subset", "prep")["target_size"]),
                        "seed": int(_require(prep, "debug_subset", "prep")["seed"]),
                    },
                },
                "paths": {
                    "exp_root": base_paths["exp_root"],
                    "logs_dir": base_paths["logs_dir"],
                    "prep_log": base_paths["prep_log"],
                    "data_root": base_paths["data_root"],
                    "profile_dirs": base_paths["profile_dirs"],
                    "prepared_csv": base_paths["prepared_csv"],
                    "label_space_json": base_paths["label_space_json"],
                    "fold_masks": base_paths["fold_masks"],
                    "prep_summary_json": base_paths["prep_summary_json"],
                },
            }
        )
        return out

    # ===== Mode: shared training requirements =====
    # preprocessing (training-only)
    preproc = _require(raw, "preprocessing", "root")
    alphabet = list(_require(preproc, "alphabet", "preprocessing"))
    k = int(_require(preproc, "k", "preprocessing"))
    max_bases = int(_require(preproc, "max_bases", "preprocessing"))
    if max_bases % k != 0:
        raise ValueError("preprocessing.max_bases must be divisible by preprocessing.k")
    optimal_length = max_bases // k
    max_position_embeddings = optimal_length + 2

    # encoder (training-only)
    model = _require(raw, "model", "root")
    encoder = dict(_require(model, "encoder", "model"))
    arch_id = _derive_arch_id(encoder, optimal_length)

    # Inject pretrain promotion root into path renderer (it’s also exposed in cfg)
    pretrain_cfg = _require(raw, "tasks", "root").get("pretrain", {})
    promotion = pretrain_cfg.get("promotion", {})
    pretrained_root = promotion.get("pretrained_root")
    paths_all = render_paths(
        experiments_root=experiments_root,
        experiment_name=name,
        pretrained_root=pretrained_root,
        arch_id=arch_id,
        profile=profile,
        fold=fold_index,
        levels=levels,
        fold_scheme=fold_scheme,
    )

    # Common training section
    out.update(
        {
            "preprocessing": {
                "alphabet": alphabet,
                "k": k,
                "max_bases": max_bases,
                "optimal_length": optimal_length,
                "max_position_embeddings": max_position_embeddings,
            },
            "model": {"encoder": encoder},
            "arch": {"id": arch_id},
            "paths": paths_all,  # full view, useful for tooling/inspecting
            "runtime": {
                **out["runtime"],
                "fold_index": fold_index,
                "fold_scheme": fold_scheme,
            },
        }
    )

    # ===== Mode: pretrain =====
    if mode == "pretrain":
        if fold_index is None:
            raise ValueError("pretrain: fold_index (CLI) is required.")
        # verify profile artifacts exist (prepared + label space + masks for exp1)
        prepared = paths_all["prepared_csv"][profile]
        label_space_json = paths_all["label_space_json"][profile]
        masks_path = paths_all["fold_masks"]["exp1"][profile]  # pretrain always uses exp1 split
        _fail_if_missing_files([prepared, label_space_json, masks_path], "pretrain")

        # pass through task-specific knobs (no defaults here)
        mlm = _require(pretrain_cfg, "mlm", "tasks.pretrain")
        dataloader = _require(pretrain_cfg, "dataloader", "tasks.pretrain")
        optimizer = _require(pretrain_cfg, "optimizer", "tasks.pretrain")
        scheduler = _require(pretrain_cfg, "scheduler", "tasks.pretrain")
        trainer = _require(pretrain_cfg, "trainer", "tasks.pretrain")

        out.update(
            {
                "pretrain": {
                    "mlm": mlm,
                    "dataloader": dataloader,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                    "trainer": trainer,
                    "promotion": {"pretrained_root": pretrained_root},
                },
                "paths_active": {
                    "data": {
                        "prepared_csv": prepared,
                        "label_space_json": label_space_json,
                        "fold_masks": masks_path,
                    },
                    "pretrain": paths_all["pretrain"],
                    "logs_dir": paths_all["logs_dir"],
                    "pretrained_root": paths_all.get("pretrained_root"),
                },
            }
        )
        return out

    # ===== Mode: finetune =====
    if mode == "finetune":
        if levels is None or not levels:
            raise ValueError("finetune: levels (CLI) are required (e.g., ['species'] or canonical list).")
        if fold_index is None:
            raise ValueError("finetune: fold_index (CLI) is required.")
        if fold_scheme is None:
            fold_scheme = "exp1"  # the single allowed default we agreed on

        # task knobs (no defaults here)
        tasks = _require(raw, "tasks", "root")
        finetune_cfg = _require(tasks, "finetune", "tasks")
        backbone = _require(finetune_cfg, "backbone", "tasks.finetune")
        backbone_path = _as_path(_require(backbone, "path", "tasks.finetune.backbone"))
        if not backbone_path.exists():
            raise FileNotFoundError(f"finetune.backbone.path not found: {backbone_path}")

        # choose masks by scheme, verify existence + data artifacts
        prepared = paths_all["prepared_csv"][profile]
        label_space_json = paths_all["label_space_json"][profile]
        masks_path = paths_all["fold_masks"][fold_scheme][profile]
        _fail_if_missing_files([prepared, label_space_json, masks_path], "finetune")

        out["runtime"]["fold_scheme"] = fold_scheme
        out["runtime"]["levels"] = list(levels)

        out.update(
            {
                "finetune": {
                    "levels": list(levels),  # explicit runtime
                    "fold_scheme": fold_scheme,
                    "fold_index": int(fold_index),
                    "head": _require(finetune_cfg, "head", "tasks.finetune"),
                    "backbone": {"path": backbone_path, "strict": bool(_require(backbone, "strict", "backbone"))},
                    "dataloader": _require(finetune_cfg, "dataloader", "tasks.finetune"),
                    "optimizer": _require(finetune_cfg, "optimizer", "tasks.finetune"),
                    "scheduler": _require(finetune_cfg, "scheduler", "tasks.finetune"),
                    "trainer": _require(finetune_cfg, "trainer", "tasks.finetune"),
                },
                "paths_active": {
                    "data": {
                        "prepared_csv": prepared,
                        "label_space_json": label_space_json,
                        "fold_masks": masks_path,
                    },
                    "finetune": render_paths(
                        experiments_root=experiments_root,
                        experiment_name=name,
                        pretrained_root=pretrained_root,
                        arch_id=arch_id,
                        profile=profile,
                        fold=fold_index,
                        levels=list(levels),
                        fold_scheme=fold_scheme,
                    )["finetune"],
                    "logs_dir": paths_all["logs_dir"],
                },
            }
        )
        return out

    # Should never reach here
    raise ValueError(f"Unknown mode: {mode}")

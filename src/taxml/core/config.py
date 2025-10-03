from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Literal

from .io import read_yaml
from .paths import prep_paths

def build_profile_paths(
    experiments_root: str | Path,
    experiment_name: str,
    profile: str,  # "full" | "debug"
) -> Dict[str, Path]:
    """
    Resolve data artifacts for a specific profile produced by 01_experiment_prep.py:
      experiments/<experiment_name>/data/<profile>/{prepared.csv,label_space.json,fold_masks_*.json}
    """
    exp_root = Path(experiments_root) / experiment_name
    data_dir = exp_root / "data" / profile
    logs_dir = exp_root / "logs"
    return {
        "exp_root": exp_root,
        "data_dir": data_dir,
        "prepared_csv": data_dir / "prepared.csv",
        "label_space_json": data_dir / "label_space.json",
        "fold_masks_exp1": data_dir / "fold_masks_exp1.json",
        "fold_masks_exp2": data_dir / "fold_masks_exp2.json",
        "summary_json": data_dir / "summary.json",
        "logs_dir": logs_dir,
    }


Required = Literal["experiment","artifacts","data","label_space","prep"]

def _require(d: Dict, key: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing required key: {key}")
    return d[key]

def load_config(mode: Literal["prep","pretrain","finetune"], config_path: str | Path) -> Dict[str, Any]:
    if mode != "prep":
        # stub for future modes
        raise NotImplementedError(f"load_config(mode={mode!r}) only supports 'prep' right now")

    cfg_raw = read_yaml(Path(config_path))

    # --- validate + extract minimal schema for prep ---
    exp        = _require(cfg_raw, "experiment")
    data       = _require(cfg_raw, "data")
    label_space= _require(cfg_raw, "label_space")
    prep       = _require(cfg_raw, "prep")

    name   = _require(exp, "name")
    seed   = _require(exp, "seed")
    union  = _require(exp, "union")

    # absolute roots (no envs)
    experiments_root = Path(_require(data, "experiments_root")).expanduser().resolve()
    input_dir        = Path(_require(data, "input_dir")).expanduser().resolve()

    inputs     = _require(data, "inputs")
    filenames  = _require(data, "filenames")
    levels     = _require(label_space, "levels")

    # derive all prep paths centrally
    paths = prep_paths(experiments_root, name)

    # flatten into a runtime config dict
    cfg: Dict[str, Any] = {
        "experiment": {"name": name, "seed": int(seed), "union": str(union)},
        "data": {
            "experiments_root": experiments_root,
            "input_dir": input_dir,
            "inputs": inputs,
            "filenames": filenames,
        },
        "label_space": {"levels": list(levels)},
        "prep": {
            "uppercase_sequences": bool(prep.get("uppercase_sequences", True)),
            "min_species_resolution": int(prep.get("min_species_resolution", 0)),
            "dedupe": {
                "scope": str(_require(prep, "dedupe")["scope"]),
                "keep_policy": str(_require(prep, "dedupe")["keep_policy"]),
            },
            "folds": {
                "k": int(_require(prep, "folds")["k"]),
                "stratify_by": str(_require(prep, "folds")["stratify_by"]),
                "seed": int(_require(prep, "folds")["seed"]),
            },
            "folds_species_group": {
                "k": int(_require(prep, "folds_species_group")["k"]),
                "stratify_by": str(_require(prep, "folds_species_group")["stratify_by"]),
                "seed": int(_require(prep, "folds_species_group")["seed"]),
            },
            "debug_subset": {
                "n_genera": int(_require(prep, "debug_subset")["n_genera"]),
                "min_species_per_genus": int(_require(prep, "debug_subset")["min_species_per_genus"]),
                "target_size": int(_require(prep, "debug_subset")["target_size"]),
                "seed": int(_require(prep, "debug_subset")["seed"]),
            },
        },
        "paths": paths,
    }
    return cfg

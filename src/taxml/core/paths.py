# src/taxml/core/paths.py
from __future__ import annotations
from pathlib import Path
from typing import Dict

def prep_paths(experiments_root: Path, experiment_name: str) -> Dict:
    """
    Return a dict of absolute paths for experiment *prep* artifacts.
    Creates base/profile directories but does not create output files.
    """
    exp_root  = Path(experiments_root).expanduser().resolve() / experiment_name
    logs_dir  = exp_root / "logs"
    data_root = exp_root / "data"

    profiles = {p: data_root / p for p in ("full", "debug")}

    def _per_profile(filename: str) -> Dict[str, Path]:
        return {name: d / filename for name, d in profiles.items()}

    out: Dict = {
        "exp_root": exp_root,
        "logs_dir": logs_dir,
        "prep_log": logs_dir / "prep.log",
        "data_root": data_root,
        "profile_dirs": profiles,
        "prepared_csv": _per_profile("prepared.csv"),
        "label_space_json": _per_profile("label_space.json"),
        "fold_masks": {
            "exp1": _per_profile("fold_masks_exp1.json"),
            "exp2": _per_profile("fold_masks_exp2.json"),
        },
        "prep_summary_json": data_root / "prep_summary.json",
    }

    # Ensure directories exist
    for d in [logs_dir, data_root, *profiles.values()]:
        d.mkdir(parents=True, exist_ok=True)

    return out

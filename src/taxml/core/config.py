from pathlib import Path
from typing import Dict

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

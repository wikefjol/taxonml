#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from taxml.core.config import load_config

def main():
    # ---- Toggle here ---------------------------------------------------------
    #MODE   = "prep"   # "prep" | "pretrain" | "finetune"
    modes = ["prep", "pretrain", "finetune"]
    CONFIG = "configs/sequence_fold.yaml"  # path to your YAML
    DEBUG  = False    # True => profile="debug"; False => "full"
    # --------------------------------------------------------------------------

    # Pretrain-only knobs (required)
    PRETRAIN_FOLD = 7

    # Finetune-only knobs (required)
    FT_LEVELS       = ["phylum","class","order","family","genus","species"]  # or ["species"]
    FT_FOLD         = 7
    FT_FOLD_SCHEME  = "exp1"  # "exp1" or "exp2"
    for MODE in modes:
        if MODE == "prep":
            cfg = load_config(
                mode="prep",
                config_path=CONFIG,
                debug=DEBUG,           # DEBUG=True -> profile="debug"; else "full"
            )

        elif MODE == "pretrain":
            cfg = load_config(
                mode="pretrain",
                config_path=CONFIG,
                debug=DEBUG,
                fold_index=PRETRAIN_FOLD,   # required
            )

        elif MODE == "finetune":
            cfg = load_config(
                mode="finetune",
                config_path=CONFIG,
                debug=DEBUG,
                levels=FT_LEVELS,          # required
                fold_index=FT_FOLD,        # required
                fold_scheme=FT_FOLD_SCHEME # default is "exp1" if omitted
            )

        else:
            raise SystemExit(f"Unknown MODE: {MODE!r}")

        # Pretty print (Path objects are stringified via default=str)
        print(json.dumps(cfg, indent=2, default=str))

if __name__ == "__main__":
    main()

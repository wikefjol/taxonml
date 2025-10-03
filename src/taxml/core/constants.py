# constants.py
IGNORE_INDEX = -100
SPECIALS = {"PAD": "[PAD]","CLS":"[CLS]","SEP":"[SEP]","MASK":"[MASK]","UNK":"[UNK]"}
LOG_KEYS_BATCH = ["epoch","global_step","batch_idx","lr",
                  "loss_batch","acc_batch",
                  # per-level (classification):
                  # f"{lvl}_loss_batch", f"{lvl}_acc_batch"
                 ]

# for experiment prep
REQUIRED_COLS = ["sequence", "kingdom", "phylum", "class", "order", "family", "genus", "species", "species_resolution"]
CANON_LEVELS = ["phylum", "class", "order", "family", "genus", "species"]
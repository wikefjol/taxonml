#!/usr/bin/env bash
# Create one GitHub issue per helper (no epics).
# Usage:
#   ./open_helper_refactor_issues.sh          # create issues
#   ./open_helper_refactor_issues.sh --dry-run  # preview only

set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; fi

die()  { echo "[error] $*" >&2; exit 1; }
info() { echo "[info]  $*"; }
run()  { if [[ $DRY_RUN -eq 1 ]]; then echo "[dry] $*"; else eval "$@"; fi }

# Preconditions
command -v gh >/dev/null 2>&1 || die "GitHub CLI 'gh' not found. Install: https://cli.github.com/"
gh auth status >/dev/null 2>&1 || die "gh is not authenticated. Run: gh auth login"
gh repo view >/dev/null 2>&1 || die "Not in a GitHub repo. cd into your project root."

# Ensure labels (idempotent)
ensure_label() {
  local name="$1" color="$2" desc="$3"
  if ! gh label list --limit 200 | awk '{print $1}' | grep -qx "${name}"; then
    run gh label create "${name}" --color "${color}" --description "$(printf %q "${desc}")"
  else
    info "Label '${name}' exists."
  fi
}
ensure_label "refactor" "0366d6" "Refactoring tasks / tech-debt"
ensure_label "helpers"  "0e8a16" "Helper modules / shared utilities"

# -------- task collector (no parsing) --------
_titles=()
_labels=()
_bodies=()
add_task() {
  _titles+=("$1")
  _labels+=("$2")
  _bodies+=("$3")
}

issue_exists() { gh issue list --state all --limit 500 --json title | jq -r '.[].title' | grep -Fxq "$1"; }

create_issue() {
  local title="$1" labels="$2" body="$3"
  if issue_exists "$title"; then
    info "Skip (exists): $title"
    return
  fi
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry] gh issue create --title $(printf %q "$title") --label $(printf %q "$labels")"
    echo "      body: $(printf %q "$body" | cut -c1-160)…"
  else
    local tmp; tmp="$(mktemp)"; printf "%s\n" "$body" > "$tmp"
    gh issue create --title "$title" --label "$labels" --body-file "$tmp"
    rm -f "$tmp"
  fi
}

# -------- define issues (one per helper) --------

add_task "[core.logging] New: setup_logging()" "refactor,helpers" \
"Create new helper module: taxonml/core/logging.py
- Provide setup_logging(console_level) with consistent formatter/handlers.
- Used by all entry points.
NOTE: New helper (no prior stable module); replaces ad-hoc setup in CLIs.
DoD:
- pretrain.py, finetune.py, 01_experiment_prep.py import core.logging.setup_logging
- No duplicate logger setup code remains in entry points."

add_task "[core.randomness] New: set_all_seeds()" "refactor,helpers" \
"Create new helper module: taxonml/core/randomness.py
- set_all_seeds(seed) → seeds random, PYTHONHASHSEED, torch, cudnn deterministic flags.
NOTE: New helper; replaces per-script seed logic.
DoD:
- All entry points call core.randomness.set_all_seeds
- Remove local seed helpers from CLIs."

add_task "[core.io] New: read_yaml / write_json / clear_dir" "refactor,helpers" \
"Create new helper module: taxonml/core/io.py
- read_yaml(path) -> dict
- write_json(path, obj, *, atomic=true)
- clear_dir(path) with safety checks
NOTE: New helper; consolidates file I/O utilities scattered in scripts.
DoD:
- CLIs use core.io instead of inline JSON/YAML helpers
- 01_experiment_prep uses core.io.write_json."

add_task "[core.paths] New: central path rendering & run dirs" "refactor,helpers" \
"Create new helper module: taxonml/core/paths.py
- build_profile_paths(experiments_root, experiment_name, profile)
- build_pretrain_run_paths(experiments_root, experiment_name, arch_id, profile)
- build_finetune_run_paths(experiments_root, experiment_name, arch_id, fold, levels, profile)
- clear_run_dir(run_dir) wrapper
NOTE: New helper; replaces duplicated build_*paths in CLIs.
DoD:
- pretrain.py & finetune.py import from core.paths
- Duplicate local implementations removed."

add_task "[core.config] New: load_config(mode) → flat runtime dict" "refactor,helpers" \
"Create new helper module: taxonml/core/config.py
- load_config(mode: 'prep'|'pretrain'|'finetune', profile: str|None, overrides: dict|None) -> dict
- Responsibilities: YAML load, env expansion, merge profile overrides, compute derived params (optimal_length, MPE, arch_id), resolve paths via core.paths.
NOTE: New helper; centralizes config handling; entry points consume a flat dict.
DoD:
- Unit tests cover pretrain/finetune happy paths with sample YAML
- CLIs call core.config.load_config and drop ad-hoc config plumbing."

add_task "[data.ingest] Move: read_union_csvs + schema checks from prep" "refactor,helpers" \
"Create module taxonml/data/ingest.py and MOVE existing helpers from 01_experiment_prep.py:
- REQUIRED_COLS
- require_columns(df, cols, context)
- read_union_csvs(input_dir, members)
DoD:
- 01_experiment_prep imports from data.ingest
- Helpers removed from 01_experiment_prep.py."

add_task "[data.clean] Move: uppercase/filter/dedupe from prep" "refactor,helpers" \
"Create module taxonml/data/clean.py and MOVE helpers from 01_experiment_prep.py:
- apply_uppercase(df)
- filter_by_resolution(df, min_species_resolution)
- dedupe_within_species_by_sequence(df, keep_policy)
DoD:
- 01_experiment_prep imports from data.clean
- Functions deleted from 01_experiment_prep.py."

add_task "[data.folds] Move: assign_folds + debug subset from prep" "refactor,helpers" \
"Create module taxonml/data/folds.py and MOVE helpers from 01_experiment_prep.py:
- assign_folds_sequence_stratified(...)
- assign_folds_species_group(...)
- make_debug_subset(...)
DoD:
- 01_experiment_prep imports from data.folds
- Old implementations removed from script."

add_task "[data.validation] New+Move: dataset probes & CSV checks" "refactor,helpers" \
"Create module taxonml/data/validation.py
- MOVE validate_datasets_and_log(...) from finetune.py
- Re-export require_columns if useful for other modules
DoD:
- finetune.py imports data.validation.validate_datasets_and_log
- Local copy removed from finetune.py."

add_task "[labels.utils] Move: parse_levels / ranks_code from finetune" "refactor,helpers" \
"Create module taxonml/labels/utils.py and MOVE helpers from finetune.py:
- parse_levels('all' or comma-list)
- ranks_code(levels) → 'ranks_6_pcofgs'
DoD:
- finetune.py imports labels.utils
- Local helpers removed."

add_task "[labels.masks] New+Move: build/load fold masks" "refactor,helpers" \
"Create module taxonml/labels/masks.py
- MOVE build_fold_masks(...) from 01_experiment_prep.py
- NEW load_fold_masks(path, fold_id, levels, class_sizes) (currently inline in finetune.py)
DoD:
- 01_experiment_prep writes masks via labels.masks.build_fold_masks
- finetune.py loads masks via labels.masks.load_fold_masks
- Inline code removed."

add_task "[training.optim] Move: param-group builder + group_params_yaml" "refactor,helpers" \
"Create module taxonml/training/optim.py and MOVE helpers from finetune.py:
- build AdamW param groups (decay/no_decay)
- group_params_yaml(model) for diagnostics
DoD:
- pretrain.py & finetune.py import from training.optim
- Inline grouping and YAML-dump helpers removed."

add_task "[training.debug] Move: mask_coverage_report (+optional sched preview)" "refactor,helpers" \
"Create module taxonml/training/debug.py and MOVE helper from finetune.py:
- mask_coverage_report(...)
- (Optional) add scheduler preview util later
DoD:
- finetune.py imports only when debug path is enabled
- Local debug helper removed."

add_task "[pipelines.prep] New: orchestration for experiment prep" "refactor,helpers" \
"Create module taxonml/pipelines/prep.py and MOVE orchestration from 01_experiment_prep.py:
- process_profile(...) (pure function)
- Keep CLI as thin wrapper that calls this pipeline module
DoD:
- 01_experiment_prep.py delegates to pipelines.prep; heavy logic removed."

add_task "[pretrain.py] Adopt config loader + helper modules" "refactor,helpers" \
"Refactor pretrain.py to consume core.config.load_config(mode='pretrain') and use helper modules:
- Use core.paths for path rendering
- Use training.optim for param groups
- Use data.ingest/clean/folds where applicable
- Remove local build_*paths/clear_run_dir duplicates
DoD:
- Behavior parity for 'full' & 'debug'; trainer still saves best.pt & last.pt; promotion path uses config."

add_task "[finetune.py] Adopt config loader (flat dict per mode)" "refactor,helpers" \
"Switch finetune.py to use core.config.load_config(mode='finetune') and remove ad-hoc config plumbing.
- Use core.paths for all paths
- Use labels.utils for parse_levels/ranks_code
- Use labels.masks.load_fold_masks
- Use training.optim for param groups
DoD:
- Behavior parity under 'full' & 'debug' profiles."

add_task "[01_experiment_prep.py] Thin wrapper over pipelines + data.*" "refactor,helpers" \
"Refactor 01_experiment_prep.py:
- Delegate to pipelines.prep.process_profile and data.ingest/clean/folds
- Keep only CLI parsing, logging, and calls into pipeline
DoD:
- No heavy helpers remain in the script."

add_task "[Cleanup] Remove duplicated helpers from entry points" "refactor,helpers" \
"Delete now-redundant helpers from:
- pretrain.py
- finetune.py
- 01_experiment_prep.py
DoD:
- Grep shows no remaining duplicates (build_*paths, parse_levels, validate_datasets_and_log, etc.)."

add_task "[Docs] Document helper layout & import layering rules" "refactor,helpers" \
"Update CONTRIBUTING.md and docs/architecture.md:
- New helper module layout (core, data, labels, training, pipelines)
- Import layering rules and boundaries
- Guidance for adding new helpers
DoD:
- Docs merged; contributors can place helpers confidently."

add_task "[CI/Tests] Smoke tests for load_config + path rendering" "refactor,helpers" \
"Add minimal tests:
- core.config.load_config('pretrain'|'finetune') returns flat dict with derived optimal_length, MPE, arch_id, and resolved paths
- core.paths renders run directories correctly for full/debug
DoD:
- Tests pass in CI; prevents regressions in config/path helpers."


# -------- create issues --------
for i in "${!_titles[@]}"; do
  create_issue "${_titles[$i]}" "${_labels[$i]}" "${_bodies[$i]}"
done

info "Done. View with: gh issue list"

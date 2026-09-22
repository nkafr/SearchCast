#!/bin/bash
# ROBUSTNESS CHECK — SearchCast at gh=24 (the released optuna_ridge.py / reproduce.sh default)
# vs our gh=48 (the paper's §5.2 stated default). Run at each dataset's best sgs from the gh=48 case
# (ETTm1 is run as a full {7,3,1} sweep, kept as an example), so this
# isolates the gh effect (paper §5.2 predicts <=0.4%). Writes to results_substack/gh24/ (does NOT
# overwrite the gh=48 searchcast_*.json). Ordered cheap -> expensive.
# OUTPUTS (all under toto_M256_run/):
#   results_substack/gh24/searchcast_<ds>.json   SearchCast paired result at gh=24, one per dataset (6 files)
#   logs_substack/gh24.log                       run trace ([gh24]/[sc] lines) for all six datasets
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"        # …/toto_M256_run/scripts_substack
ROOT="$(cd "$HERE/.." && pwd)"               # …/toto_M256_run
REPO="$(cd "$HERE/../.." && pwd)"            # …/SearchCast (repo root)
cd "$REPO"
export HF_HUB_OFFLINE=1
PY="$REPO/.venv/bin/python"
OUT="$ROOT/results_substack/gh24"            # gh=24 SearchCast jsons land here (NOT the gh=48 ones)
LG="$ROOT/logs_substack/gh24.log"            # run trace
mkdir -p "$OUT"
: > "$LG"                                     # truncate the log for a fresh run
sc() { echo "[gh24] $1 sgs=$2 start $(date '+%m-%d %H:%M')" | tee -a "$LG"
  SC_OUT_DIR="$OUT" "$PY" "$HERE/run_searchcast_paired.py" "$1" 24 "$2" >> "$LG" 2>&1
  echo "[gh24] $1 done $(date '+%m-%d %H:%M')" | tee -a "$LG"; }
# The 2nd arg to sc is the series-group size (sgs). For each dataset we pass the best sgs found
# at gh=48. ETTm1 is the one we run as a full {7,3,1} sweep, kept as an example.
sc exchange 4     # ~3 min, exchange 4 means we use sgs=4
sc ETTh1 7        # ~13 min
sc ETTh2 7        # ~13 min   (this dataset is the head-to-head tie)
sc ETTm2 7        # ~8 h      (a narrow Toto win, inside its noise floor)
sc weather 1      # ~10 h     (a narrow Toto win inside its noise floor, and sgs=1 = 21 per-series)
# ETTm1: full {7,3,1} sweep, kept as an example. It selects sgs=3 (matches the shipped json).
sc ETTm1 7,3,1    # ~1.5 days (3x the single-sgs cost)
echo "[gh24] ALL DONE $(date '+%m-%d %H:%M')" | tee -a "$LG"

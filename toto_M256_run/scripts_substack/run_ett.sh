#!/bin/bash
# ============================================================================
# RUN 1 of 3 — ETT datasets (ETTh1, ETTh2, ETTm1, ETTm2)
# Toto-2 (zero-shot) vs SearchCast (tuned Ridge), normalized-MSE, paired M=256.
# All CPU/float32 (the stock `from_pretrained().to("cpu")` config).
# Paths are derived from this script's location
# ============================================================================
# OUTPUTS (all under toto_M256_run/):
#   results_substack/toto_ett.jsonl                              Toto rows — one JSON per size/mode/ctx/horizon/point (append)
#   results_substack/searchcast_{ETTh1,ETTh2,ETTm1,ETTm2}.json   SearchCast paired results (test-selected sgs over {7,3,1})
#   results_substack/preds/searchcast_ETT*_sgs*_H*.npz           SearchCast per-window forecasts (via SC_SAVE_PREDS)
#   logs_substack/ett.log                                        Toto run trace ([ett]/[toto] lines + timings)
#   logs_substack/sc_<ds>.log                                    SearchCast run trace, one per dataset
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"          # …/toto_M256_run/scripts_substack
ROOT="$(cd "$HERE/.." && pwd)"                  # …/toto_M256_run
REPO="$(cd "$HERE/../.." && pwd)"               # …/SearchCast (repo root)
cd "$REPO"
export HF_HUB_OFFLINE=1 TOTO_DEVICE=cpu TOTO_DTYPE=fp32
PY="$REPO/.venv/bin/python"
RES="$ROOT/results_substack"
LOG="$ROOT/logs_substack"
OUT="$RES/toto_ett.jsonl"
LG="$LOG/ett.log"
mkdir -p "$RES/preds"; : > "$LG"

toto() { echo "[ett] toto $* $(date '+%H:%M')" | tee -a "$LG"
  "$PY" "$HERE/toto_eval.py" --decode single --point both --M 256 --out "$OUT" "$@" >> "$LG" 2>&1; }

echo "[ett] === head-to-head: ctx2048, all sizes x modes ==="
toto --size 22m  --datasets ETTh1 ETTh2 ETTm1 ETTm2 --modes univariate multivariate --contexts 2048 --batch 12
toto --size 1B   --datasets ETTh1 ETTh2 ETTm1 ETTm2 --modes univariate multivariate --contexts 2048 --batch 6
toto --size 2.5B --datasets ETTh1 ETTh2 ETTm1 ETTm2 --modes univariate               --contexts 2048 --batch 4
# 2.5B-multivariate: ~10-13 h/dataset (7-channel mixing single-pass). It HURTS on channel-independent
# ETT — a documented negative result (uv->mv delta is +0.003..+0.015 across sizes) — but we run all
# four for grid completeness.
toto --size 2.5B --datasets ETTh1 ETTh2 ETTm1 ETTm2 --modes multivariate --contexts 2048 --batch 4

echo "[ett] === context sweep (for the 'context helps' figure): 22m + 1B, ctx {512,1024,2048} ==="
toto --size 22m --datasets ETTh1 ETTh2 ETTm1 ETTm2 --modes univariate --contexts 512 1024 --batch 12
toto --size 1B  --datasets ETTh1 ETTh2 ETTm1 ETTm2 --modes univariate --contexts 512 1024 --batch 6

# SearchCast paired (real Optuna search). sgs grid {7,3,1}: 7 = all 7 channels pooled, 1 = per-series;
# 1 and 7 are the divisors of the 7 ETT channels, 3 (consecutive groups of 3 -> [0,1,2][3,4,5][6]) is an
# extra coarse point that the paper's own Figure 4 also sweeps. All four ETT sets are searched over the
# full {1,3,7} grid; SearchCast test-selects the lowest full-stride-1 TEST MSE per dataset:
#   ETTh1 / ETTh2 / ETTm2 -> sgs=7 (full pool)      ETTm1 -> sgs=3 (moderate grouping, wins by 0.0009)
# No verdict changes vs the earlier sgs=7-only runs (ETTm1's SC edges from 0.347 to 0.346; Toto still -0.015).
echo "[ett] === SearchCast paired (Optuna search; test-selects sgs from {7,3,1} per dataset) ==="
for ds in ETTh1 ETTh2 ETTm1 ETTm2; do
  SC_SAVE_PREDS="$RES/preds" "$PY" "$HERE/run_searchcast_paired.py" "$ds" 48 7,3,1 > "$LOG/sc_$ds.log" 2>&1
done
echo "[ett] DONE $(date '+%H:%M')"

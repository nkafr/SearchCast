#!/bin/bash
# ============================================================================
# RUN 2 of 3 — Exchange (8 daily FX rates, near-random-walk, non-stationary)
# Full context sweep {128,256,512,1024,2048} — Exchange has a U-shaped context
# response (extremes win, middle loses), unlike ETT's monotonic one.
# ============================================================================
# OUTPUTS (all under toto_M256_run/):
#   results_substack/toto_exchange.jsonl                   Toto rows — one JSON per size/mode/ctx/horizon/point (append)
#   results_substack/searchcast_exchange.json              SearchCast paired result (test-selected sgs over {8,4,2,1})
#   results_substack/preds/searchcast_exchange_sgs*_H*.npz   SearchCast per-window forecasts (via SC_SAVE_PREDS)
#   logs_substack/exchange.log                             Toto run trace ([exc]/[toto] lines + timings)
#   logs_substack/sc_exchange.log                          SearchCast run trace
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"        # …/toto_M256_run/scripts_substack
ROOT="$(cd "$HERE/.." && pwd)"               # …/toto_M256_run
REPO="$(cd "$HERE/../.." && pwd)"            # …/SearchCast (repo root)
cd "$REPO"
export HF_HUB_OFFLINE=1 TOTO_DEVICE=cpu TOTO_DTYPE=fp32
PY="$REPO/.venv/bin/python"
RES="$ROOT/results_substack"                 # all result files land here
LOG="$ROOT/logs_substack"                    # all run traces land here
OUT="$RES/toto_exchange.jsonl"               # Toto rows for this run
LG="$LOG/exchange.log"                        # Toto run trace for this run
mkdir -p "$RES/preds"
: > "$LG"                                     # truncate the log for a fresh run
toto() { echo "[exc] toto $* $(date '+%H:%M')" | tee -a "$LG"
  "$PY" "$HERE/toto_eval.py" --decode single --point both --M 256 --datasets exchange --out "$OUT" "$@" >> "$LG" 2>&1; }

echo "[exc] === context sweep, both modes (multivariate helps: FX rates co-move vs USD) ==="
toto --size 22m --modes univariate multivariate --contexts 128 256 512 1024 2048 --batch 16
toto --size 1B  --modes univariate              --contexts 128 256 512 1024 2048 --batch 6
toto --size 2.5B --modes univariate             --contexts 128 256 512 1024 2048 --batch 4
# Multivariate is the winning mode on Exchange; long-context mixing is the slow part.
toto --size 1B   --modes multivariate --contexts 128 2048 --batch 6
toto --size 2.5B --modes multivariate --contexts 128 2048 --batch 2   # ctx2048 ~6 h (8-ch mixing at 2.5B)

echo "[exc] === SearchCast paired (test-selects sgs over divisors of 8) ==="
SC_SAVE_PREDS="$RES/preds" "$PY" "$HERE/run_searchcast_paired.py" exchange 48 8,4,2,1 > "$LOG/sc_exchange.log" 2>&1
echo "[exc] DONE $(date '+%H:%M')"

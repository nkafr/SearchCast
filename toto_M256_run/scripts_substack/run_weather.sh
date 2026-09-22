#!/bin/bash
# ============================================================================
# RUN 3 of 3 — Weather (21 coupled meteorological variables, 10-min)
# The multivariate showcase: Toto's learned channel attention exploits the
# 21 physically-coupled variables — the axis SearchCast (channel-independent)
# structurally cannot use. SearchCast's own best here is sgs=1 (21 per-series
# searches), the most expensive config in the paper.
# Note the mean/median REVERSAL: on Weather, univariate MSE prefers the MEDIAN
# (skewed multi-scale channels miscalibrate the quantile-mean); multivariate
# restores mean≈median. We save both points and pick per config in summary.
# ============================================================================
# OUTPUTS (all under toto_M256_run/):
#   results_substack/toto_weather.jsonl                  Toto rows — one JSON per size/mode/ctx/horizon/point (append)
#   results_substack/searchcast_weather.json             SearchCast paired result (test-selected sgs=1)
#   results_substack/preds/searchcast_weather_sgs1_H*.npz  SearchCast per-window forecasts (via SC_SAVE_PREDS)
#   logs_substack/weather.log                            Toto run trace ([wx]/[toto] lines + timings)
#   logs_substack/sc_weather.log                         SearchCast run trace
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"        # …/toto_M256_run/scripts_substack
ROOT="$(cd "$HERE/.." && pwd)"               # …/toto_M256_run
REPO="$(cd "$HERE/../.." && pwd)"            # …/SearchCast (repo root)
cd "$REPO"
export HF_HUB_OFFLINE=1 TOTO_DEVICE=cpu TOTO_DTYPE=fp32
PY="$REPO/.venv/bin/python"
RES="$ROOT/results_substack"                 # all result files land here
LOG="$ROOT/logs_substack"                    # all run traces land here
OUT="$RES/toto_weather.jsonl"                # Toto rows for this run
LG="$LOG/weather.log"                        # Toto run trace for this run
mkdir -p "$RES/preds"
: > "$LG"                                     # truncate the log for a fresh run
toto() { echo "[wx] toto $* $(date '+%H:%M')" | tee -a "$LG"
  "$PY" "$HERE/toto_eval.py" --decode single --point both --M 256 --datasets weather --out "$OUT" "$@" >> "$LG" 2>&1; }

echo "[wx] === univariate (context sweep) + multivariate (the winning mode) ==="
toto --size 22m  --modes univariate   --contexts 512 1024 2048 --batch 12
toto --size 1B   --modes univariate   --contexts 512 1024 2048 --batch 6
toto --size 2.5B --modes univariate   --contexts 2048          --batch 4     # ctx2048 ~10 h (21 ch)
toto --size 22m  --modes multivariate --contexts 2048          --batch 8
toto --size 1B   --modes multivariate --contexts 2048          --batch 4
toto --size 2.5B --modes multivariate --contexts 2048          --batch 2     # ~14 h (21-ch mixing at 2.5B)

echo "[wx] === SearchCast paired: sgs=1 (fully per-series, the paper's stated best for Weather) ==="
SC_SAVE_PREDS="$RES/preds" "$PY" "$HERE/run_searchcast_paired.py" weather 48 1 > "$LOG/sc_weather.log" 2>&1
echo "[wx] DONE $(date '+%H:%M')"

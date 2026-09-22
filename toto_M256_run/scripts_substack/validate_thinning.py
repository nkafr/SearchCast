"""Protocol proof: evaluating on M=256 evenly-spaced ("thinned") origins gives the
same mean MSE as SearchCast's full stride-1 evaluation (all ~10k origins).

For cheap reference forecasters (persistence, and the global-Ridge L=720 baseline that
is SearchCast's own 'Global MSE') we score BOTH the full stride-1 window set and the
thinned M=256 subset. If full ~= thinned, the thinning is an unbiased estimator, so
Toto's M=256 numbers are comparable to the paper's full-stride-1 Table 1.

Outputs: results_substack/thinning_M256.txt  (the comparison table; also echoed to stdout).
Runtime: ~minutes — no Toto involved, only persistence + the closed-form global-Ridge baseline.
"""
import warnings; 
warnings.filterwarnings("ignore")  # keep library path warnings out of public output
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE)
import numpy as np, torch
from ltsf_common import (prepare, n_windows, build_targets, thin_indices, mse,persistence_preds, CUTOFFS)

import optuna_ridge as og

CSV = {"ETTh1": "data/ETTh1.csv", "ETTh2": "data/ETTh2.csv", "ETTm1": "data/ETTm1.csv",
       "ETTm2": "data/ETTm2.csv", "exchange": "data/exchange_rate.csv", "weather": "data/weather.csv"}
M = 256
OUT = os.path.join(os.path.dirname(__file__), "..", "results_substack", "thinning_M256.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
lines = []
def out(s=""):
    print(s); lines.append(s)

out("THINNING PROOF — full stride-1 vs thinned M=256 (unbiased if full ~= thin)")
out("Reference forecasters: Persistence (last value) and Global-Ridge L=720 (SearchCast's own baseline).")
for name, csv in CSV.items(): #  loop over the 6 datasets (ETTh1, ETTh2, ...)
    data, ss, se, cols = prepare(os.path.join(BASE, csv))
    ts = ss[2]
    glob = og.search_global_baseline(data, 720, 1e-5, torch.device("cpu"),
                                     split_starts=ss, split_ends=se, use_local_norm=True)
    out(f"\n{name}  (C={data.shape[1]}, n_test={se[2]-ss[2]})")
    out(f"  {'H':>4} {'Nfull':>7} | {'persist_full':>12} {'persist_M256':>12} {'d':>7} | "
        f"{'gRidge_full':>11} {'gRidge_M256':>11} {'d':>7}")
    for H in CUTOFFS:
        N = n_windows(ss, se, H) # Number of stride-1 test windows N for horizon H. Remember, number of windows is determined by the horizon
        idx = thin_indices(N, M)
        tg = build_targets(data, ts, H, np.arange(N)) # ground truth for the FULL stride-1 set: here w_indices = arange(N), so all N windows (thinning is applied separately via idx below).
        p = persistence_preds(data, ts, H, np.arange(N))
        g = glob[H].transpose(0, 1)                      # global-ridge preds reshaped (C,N,H)->(N,C,H)
        pf, pt = mse(p, tg), mse(p[idx], tg[idx])        # persistence MSE:   full (all N)  vs  thinned (M=256)
        gf, gt = mse(g, tg), mse(g[idx], tg[idx])        # global-ridge MSE:  full (all N)  vs  thinned (M=256)
        out(f"  {H:>4} {N:>7} | {pf:>12.4f} {pt:>12.4f} {pt-pf:>+7.4f} | "
            f"{gf:>11.4f} {gt:>11.4f} {gt-gf:>+7.4f}")
open(OUT, "w").write("\n".join(lines) + "\n")
out(f"\nsaved {os.path.relpath(OUT, BASE)}  —  thinning is unbiased where the 'd' (thin - full) columns are ~0.")

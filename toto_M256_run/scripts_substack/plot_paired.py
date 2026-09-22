"""Plot one rolling-origin window: history (context) -> prediction start -> forecast vs actual,
overlaying Toto and (if available) SearchCast, both scored on the SAME window.

Usage:
  python plot_paired.py --dir results/preds --ds ETTh1 --H 720 \
      --toto_size 22m --toto_mode univariate --sgs 7 --k 0 --ch 6 --out window.png

Inputs : the .npz prediction dumps in --dir (toto_*.npz + searchcast_*.npz), produced by
         toto_eval.py --save_preds and run_searchcast_paired.py (SC_SAVE_PREDS).
Outputs: one PNG at --out (default window.png). In this package the figures live in
         results_substack/figures/ (e.g. paired_ETTh1_H720.png).
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True) # folder holding the .npz prediction dumps
ap.add_argument("--ds", default="ETTh1") # which dataset to plot (one per run)
ap.add_argument("--H", type=int, default=720) # horizon/cutoff of the saved dumps
ap.add_argument("--toto_size", default="22m") # which Toto model size's file to load
ap.add_argument("--toto_mode", default="univariate") # Toto run mode (part of the filename)
ap.add_argument("--sgs", type=int, default=7) # which SearchCast series-group-size file
ap.add_argument("--k", type=int, default=0, help="which window (0..M-1)") # pick one of the 256 windows
ap.add_argument("--ch", type=int, default=6, help="channel (ETT: 6 = OT)") # which channel to draw
ap.add_argument("--point", default="mean", choices=["mean", "median"]) # Toto's point-forecast type
ap.add_argument("--out", default="window.png") # output PNG path
a = ap.parse_args()

tz = np.load(os.path.join(a.dir, f"toto_{a.ds}_{a.toto_size}_{a.toto_mode}_ctx2048_H{a.H}.npz")) #load toto predictions
k, ch, H = a.k, a.ch, a.H
ctx = tz["context"][k, ch] # history before the origin, for window k, channel ch,  L = its length
L = len(ctx) 
actual = tz["targets"][k, ch] # the true future values (length H)
toto = tz[f"pred_{a.point}"][k, ch]  #Toto's point forecast (mean or median), length H
xc = np.arange(-L, 0) # x-axis: history on [-L,0), forecast on [0,H)
xf = np.arange(0, H) 

plt.figure(figsize=(13, 4))
plt.plot(xc, ctx, color="#444", lw=1, label="context (history)")
plt.plot(xf, actual, color="black", lw=1.6, label="actual")
plt.plot(xf, toto, color="#3598ec", lw=1.4, ls="--", label=f"Toto-{a.toto_size} {a.point}")

#optionally plot Searchcast predictions on top of Toto if SearchCast .npz predictions exist
sc_path = os.path.join(a.dir, f"searchcast_{a.ds}_sgs{a.sgs}_H{a.H}.npz")
if os.path.exists(sc_path):
    sz = np.load(sc_path)
    # the "paired" guarantee: both files' window k must be the same origin row
    assert int(sz["win_start"][k]) == int(tz["win_start"][k]), "windows not aligned!"
    plt.plot(xf, sz["preds"][k, ch], color="#ff0099", lw=1.4, ls="--", label=f"SearchCast sgs={a.sgs}")

plt.axvline(0, color="red", ls=":", lw=1, label="prediction start") # vertical marker where forecasting begins
plt.title(f"{a.ds}  ch{ch}  H={H}  window k={k}  (origin row {int(tz['win_start'][k])})")
plt.xlabel("steps relative to prediction start") 
plt.ylabel("normalized value")
plt.legend(loc="upper left", fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(a.out, dpi=120)
print(f"saved {a.out}  (Toto {'+ SearchCast' if os.path.exists(sc_path) else 'only'})")

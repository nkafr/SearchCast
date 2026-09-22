"""Run SearchCast's real Optuna search and score it on the SAME thinned windows as Toto.

Mirrors optuna_ridge.main() + scripts/reproduce.sh canonical config:
  --scaler_scope local --scaler_method mean --local_horizon_group_size {gh}
  --n_folds 3 --pool_series --n_trials 20, sweeping sgs over divisors of the
  channel count and selecting the sgs with the lowest TEST MSE (test-selection).

Usage: python run_searchcast_paired.py ETTh1 [gh] [sgs_list]
Env vars: SC_SEED (RNG seed, default 0), SC_OUT_DIR (JSON output dir), SC_SAVE_PREDS (per-window .npz dir).

Outputs:
  searchcast_<ds>.json   the paired result; per horizon it stores the full-stride-1 MSE plus
                         thin_M{64,96,256}. Written to $SC_OUT_DIR (default results_substack/;
                         run_gh24.sh points it at results_substack/gh24/ so gh=48 is not overwritten).
  stdout                 "[sc] …" progress lines (orchestrators redirect to logs_substack/…).
  $SC_SAVE_PREDS/searchcast_<ds>_sgs<n>_H<h>.npz   optional per-window forecasts
                         (keys: idx, win_start, context, preds, targets) — only if SC_SAVE_PREDS is set.
"""
import warnings; warnings.filterwarnings("ignore")  # keep library path warnings out of public output
import sys, os, json, time, math
sys.path.insert(0, os.path.dirname(__file__))                        # for ltsf_common (same dir)
# Repo root = two levels up from this script (…/SearchCast). No hardcoded/user paths.
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE)                                            # for optuna_ridge + data/
import numpy as np, torch
from ltsf_common import prepare, n_windows, build_targets, build_contexts, thin_indices, mse, CUTOFFS
import optuna_ridge as og
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
CSV = {"ETTh1": "data/ETTh1.csv", "ETTh2": "data/ETTh2.csv",
       "ETTm1": "data/ETTm1.csv", "ETTm2": "data/ETTm2.csv",
       "exchange": "data/exchange_rate.csv", "weather": "data/weather.csv",
       "electricity": "data/electricity.csv", "traffic": "data/traffic.csv"}
THIN_MS = [64, 96, 256]  # window counts we score SearchCast at; M=256 is the set paired with Toto (the others are legacy sanity checks from the earlier M=64/96 stage)

ds = sys.argv[1] #the dataset argument
GH = int(sys.argv[2]) if len(sys.argv) > 2 else 48          # paper default gh=48
SGS_LIST = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [7, 3, 1]
SC_SAVE_PREDS = os.environ.get("SC_SAVE_PREDS")             # dir for per-(sgs,H) M=256 .npz dumps
SEED = int(os.environ.get("SC_SEED", "0"))                  # SearchCast RNG seed. The paper sets none; we fix 0. Override (SC_SEED=1,2,...) for the seed-variance demo.

data, ss, se, cols = prepare(os.path.join(BASE, CSV[ds]))
test_start = ss[2]
T, S = data.shape
n_train, n_val, n_test = ss[1], ss[2]-ss[1], se[2]-ss[2]
total_len = se[2]
search_train_ratio = (n_train + n_val) / total_len * 0.5
search_test_ratio = n_test / total_len

DEVICE = torch.device("cpu")
LOOKBACKS = np.logspace(5, 11, 19, base=2, dtype=int)        # 32..2048
HORIZONS = list(range(1, 721))
ALPHAS = torch.logspace(-6, 4, 21, device=DEVICE)

print(f"[sc] {ds}: S={S} n_train={n_train} n_test={n_test} gh={GH} seed={SEED} sgs sweep={SGS_LIST}", flush=True)
results = {}
for sgs in SGS_LIST:
    t0 = time.time()
    print(f"\n[sc] ===== sgs={sgs} =====", flush=True)
    best_df, local_raw_ind = og.search_local_models(
        data, cols, LOOKBACKS, HORIZONS, ALPHAS, n_trials=20, device=DEVICE,
        horizon_group_size=GH, series_group_size=sgs,
        search_train_ratio=search_train_ratio, search_test_ratio=search_test_ratio,
        split_starts=ss, split_ends=se, n_folds=3, fold_reg_lambda=0.0,
        scaler_scope="local", scaler_method="mean",
        fixed_local_ratio=None, fixed_noise_type=None, fixed_aug_sigma=None,
        pool_series=True, seed=SEED)
    # assemble (N, S, H) predictions exactly as main() does. The end format is (N, S, H)
    # N= number of test windows for horizon H = n_test - H + 1 -> {H=96 → 2785; H=720 → 2161}
    # S = number of series/channels in the dataset (data.shape[1]) -> for Etth1 it's 7
    # H= the forecast horizon = how many steps ahead we predict (the cutoff) -> {96,192,336,720}
    local_raw = [] 
    for h_idx in range(0, len(HORIZONS), GH):
        #For each horizon-group, it stacks the S individual series back together → one tensor of shape (S, N, GH). 
        # So local_raw is a list: local_raw[0] = horizons 1–48 for all series, local_raw[1] = horizons 49–96, etc.
        # NOTE: N differs per group — it shrinks as horizon grows:
        #   local_raw[0] = (7, 2833, 48)   ...   local_raw[14] = (7, 2161, 48)
        hg = HORIZONS[h_idx:h_idx + GH]
        local_raw.append(torch.stack([local_raw_ind[(s, hg[0])] for s in range(S)], dim=0))

    entry = {}
    for H in CUTOFFS: # remember CUTOFFS = [96, 192, 336, 720]
        # in this loop we build the predicctions (pred), one pred (N, S, H) tensor per horizon
        # In for H in CUTOFFS:, pred is rebuilt every iteration with a different shape — both N and H change:
        # Example in Etth1: H=96, pred =(2785, 7, 96) | H = 192  pred =(2689, 7, 192) ... H=720, pred =(2161, 7, 720)
        # with thinning we store M = 256 window indices
        N = n_windows(ss, se, H)
        pred = torch.cat([p[:, :N] for p in local_raw[:math.ceil(H / GH)]], dim=-1)[:, :, :H]  # (S,N,H)
        pred = pred.transpose(0, 1)                                                            # (N,S,H), the final predictions format
        tg_full = build_targets(data, test_start, H, np.arange(N)) # the observed data, also shape (N, S, H)
        e = {"full": mse(pred, tg_full), "N": int(N)} # mean squared error over all N·S·H numbers, e["full"] = error using every window.
        for M in THIN_MS:
            idx = thin_indices(N, M)  #for H=96 on Etth1, pred = (2785, 7, 96) becomes (256, 7, 96)
            e[f"thin_M{M}"] = mse(pred[idx], tg_full[idx]) # e["thin_M"] = error on M thin indices to compare with Toto
        # Persist the exact M=256 evaluated windows + SearchCast forecasts (paired with Toto).
        # NOTE: only PREDICTIONS are saved, not the Ridge weights: search_local_models returns
        # predictions (local_raw_ind), and refit_test() discards its weight matrix Theta after
        # predicting. Dumping weights would require patching optuna_ridge to return Theta.
        if SC_SAVE_PREDS:
            os.makedirs(SC_SAVE_PREDS, exist_ok=True)
            idx256 = thin_indices(N, 256) # the 256 chosen window indices to save
            # context = last 512 steps before each origin (same normalized series both models saw),
            # this is not the actual context used for predicitions (L is the actual), the context is for diplay history in plots 
            # so Toto and SearchCast forecasts can be plotted over a shared history lead-in.
            np.savez_compressed(
                os.path.join(SC_SAVE_PREDS, f"searchcast_{ds}_sgs{sgs}_H{H}.npz"),
                idx=idx256 , # (256,)  which windows
                win_start=test_start + idx256, #  (256,)  their absolute position in the series
                context=build_contexts(data, test_start, 512, idx256).numpy(), # context shape: (M=256, S, 512)
                preds=pred[idx256].numpy(),  # (256, S, H)  SearchCast forecasts
                targets=tg_full[idx256].numpy()) # (256, S, H)  ground truth
        entry[H] = e ## e = {full, N, thin_M64, thin_M96, thin_M256} for each cutoff
        print(f"[sc] sgs={sgs} H={H}: full={e['full']:.4f} "
              + " ".join(f"M{M}={e[f'thin_M{M}']:.4f}" for M in THIN_MS), flush=True)
    entry["mean_full"] = float(np.mean([entry[H]["full"] for H in CUTOFFS]))
    #like saying entry["mean_full"] = mean( entry[96]["full"], entry[192]["full"], entry[336]["full"], entry[720]["full"] )
    entry["sec"] = round(time.time() - t0, 1)
    results[sgs] = entry
    print(f"[sc] sgs={sgs} MEAN full MSE={entry['mean_full']:.4f}  ({entry['sec']:.0f}s)", flush=True)

# SearchCast test-selection: pick sgs minimizing TEST MSE (as the released pipeline does; Toto gets the same best-of-config)
best_sgs = min(results, key=lambda k: results[k]["mean_full"])
print(f"\n[sc] TEST SELECTION: best sgs={best_sgs} (lowest mean TEST MSE={results[best_sgs]['mean_full']:.4f})")
print(f"[sc] per-sgs mean test MSE: " + ", ".join(f"sgs{k}={results[k]['mean_full']:.4f}" for k in sorted(results)))
out = {"dataset": ds, "gh": GH, "best_sgs": best_sgs, "results": {str(k): v for k, v in results.items()}}
# Write into results_substack (override with SC_OUT_DIR).
outdir = os.environ.get("SC_OUT_DIR", os.path.join(os.path.dirname(__file__), "..", "results_substack"))
os.makedirs(outdir, exist_ok=True)
p = os.path.join(outdir, f"searchcast_{ds}.json")
json.dump(out, open(p, "w"), indent=1, default=float)
print(f"[sc] saved results_substack/searchcast_{ds}.json")

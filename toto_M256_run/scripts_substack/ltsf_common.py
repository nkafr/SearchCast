"""Shared LTSF eval utilities that replicate SearchCast's exact protocol.

Data prep (StandardScaler on train, ETT 20-month truncation), split indices,
test-window construction, and the normalized-MSE metric are copied verbatim
from optuna_ridge.main() so any forecaster we drop in is scored on the same
scale as the paper's Table 1.

Outputs: none — this is a library, imported by toto_eval.py, run_searchcast_paired.py,
validate_thinning.py and summary_total.py. It writes no files.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

CUTOFFS = [96, 192, 336, 720]


def prepare(csv_path):
    """Return (data[T,C] float32 z-scored w/ train stats, split_starts, split_ends, colnames).

    Mirrors optuna_ridge.main() lines ~1343-1372.
    """
    df = pd.read_csv(csv_path)
    scaler = StandardScaler()
    if 'ETT' not in csv_path: # non-ETT (weather, exchange): chronological split 70% train / 10% val / 20% test (the paper's 7:1:2), over the full file
        n_train = int(len(df) * 0.7)
        n_test = int(len(df) * 0.2)
        n_val = len(df) - n_train - n_test
        total_len = len(df)
    else:
        if 'h' in csv_path.split('/')[-1]:      # ETTh1/h2 hourly
            n_train, n_val, n_test = 12*30*24, 4*30*24, 4*30*24 #8,640 / 2,880 / 2,880, so total_len used = 14,400 (3,020 discarded)
            total_len = 12*30*24 + 8*30*24       # 14400
        else:                                     # ETTm1/m2 15-min
            n_train, n_val, n_test = 12*30*24*4, 4*30*24*4, 4*30*24*4 #	34,560 / 11,520 / 11,520, so total_len used = 57,600 (12,080 discarded)
            total_len = 12*30*24*4 + 8*30*24*4    # 57600
    split_starts = [0, n_train, n_train + n_val]
    split_ends = [n_train, n_train + n_val, total_len]
    df_values = df.drop(columns=['date']).values[:total_len]
    scaler.fit(df_values[:n_train])
    data = torch.tensor(scaler.transform(df_values), dtype=torch.float32)
    return data, split_starts, split_ends, list(df.columns[1:])


def n_windows(split_starts, split_ends, H):
    """Number of stride-1 test windows N for horizon H (matches SearchCast unfold)."""
    # N = n_test − H + 1 (where n_test = se[2] − ss[2]
    n_test = split_ends[2] - split_starts[2]
    return n_test - H + 1


def build_targets(data, test_start, H, w_indices):
    """Ground-truth targets [M, C, H] for the given window indices.

    Window w's target = data[test_start+w : test_start+w+H]  (exactly the
    SearchCast unfold alignment, where the first test target sits at test_start).
    Returns (M, C, H)
    """
    tg = torch.stack([data[test_start + w: test_start + w + H] for w in w_indices])  # [M,H,C]
    return tg.transpose(1, 2).contiguous()  # [M,C,H]


def build_contexts(data, test_start, ctx, w_indices):
    """Toto context windows [M, C, ctx] ending just before each target."""
    cs = torch.stack([data[test_start + w - ctx: test_start + w] for w in w_indices])  # [M,ctx,C]
    return cs.transpose(1, 2).contiguous()  # [M,C,ctx]


def thin_indices(N, M):
    """M evenly-spaced window indices in [0, N-1] (unbiased subsample of the
    same overlapping-window population)."""
    if M >= N:
        return np.arange(N)
    return np.linspace(0, N - 1, M).round().astype(int)


def mse(pred, target):
    """NaN-safe MSE: averages over finite entries only (sporadic MPS NaNs)."""
    d = (pred - target) ** 2
    m = torch.isfinite(d)
    return float(d[m].mean()) if bool(m.any()) else float("nan")


def mae(pred, target):
    d = (pred - target).abs()
    m = torch.isfinite(d)
    return float(d[m].mean()) if bool(m.any()) else float("nan")


# ---- cheap control forecasters (vectorizable over ALL stride-1 windows) ----

def persistence_preds(data, test_start, H, w_indices):
    """Last-value forecast: repeat data[test_start+w-1] for H steps -> [M,C,H]."""
    last = torch.stack([data[test_start + w - 1] for w in w_indices])  # [M,C]
    return last.unsqueeze(-1).expand(-1, -1, H).contiguous()


def seasonal_naive_preds(data, test_start, H, w_indices, period):
    """Seasonal-naive: y_hat[t+h] = y[t+h-period] using the last observed period."""
    out = []
    for w in w_indices:
        ctx_end = test_start + w  # exclusive
        block = data[ctx_end - period: ctx_end]           # [period, C]
        reps = (H + period - 1) // period
        tiled = block.repeat(reps, 1)[:H]                  # [H, C]
        out.append(tiled.transpose(0, 1))                 # [C, H]
    return torch.stack(out)                                # [M,C,H]

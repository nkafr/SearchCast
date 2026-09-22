"""Evaluate Toto-2.0-2.5B on the exact SearchCast normalized-MSE protocol.

For each (dataset, mode, context, horizon) we forecast a thinned set of stride-1
test windows and report normalized MSE / MAE, directly comparable to Table 1.

Usage (the orchestrators pin TOTO_DEVICE=cpu TOTO_DTYPE=fp32 — every shipped result is CPU/fp32;
bf16 was dropped after it produced sporadic NaNs on MPS and was slow on CPU):
  TOTO_DEVICE=cpu TOTO_DTYPE=fp32 python toto_eval.py \
      --datasets ETTh1 ETTm2 --modes multivariate univariate \
      --contexts 512 1024 2048 --M 128 --batch 8 --out results/toto.jsonl

Outputs:
  --out FILE        appends one JSON row per (dataset, size, mode, context, horizon, point);
                    fields: mse, mae, M, N_full, n_nan, nan_frac, device, dtype, sec, sec_per_win.
                    The orchestrators point this at results_substack/toto_<group>.jsonl.
  stdout            one "[toto] …" progress line per row (orchestrators redirect it to
                    logs_substack/<group>.log).
  --save_preds DIR  optional: one  toto_<ds>_<size>_<mode>_ctx<c>_H<h>.npz  per config
                    (keys: idx, win_start, context, targets, pred_mean/pred_median).
"""
import warnings
warnings.filterwarnings("ignore")  # keep library path warnings out of public output
import os, sys, time, json, argparse, gc
sys.path.insert(0, os.path.dirname(__file__))                        # for ltsf_common (same dir)
# Repo root = two levels up from this script (…/SearchCast). No hardcoded/user paths.
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE)                                            # for optuna_ridge + data/
import numpy as np
import torch
from ltsf_common import (prepare, n_windows, build_targets, build_contexts,
                         thin_indices, mse, mae, CUTOFFS)
CSV = {"ETTh1": "data/ETTh1.csv", "ETTh2": "data/ETTh2.csv",
       "ETTm1": "data/ETTm1.csv", "ETTm2": "data/ETTm2.csv",
       "weather": "data/weather.csv", "exchange": "data/exchange_rate.csv",
       "electricity": "data/electricity.csv", "traffic": "data/traffic.csv"}


def quantiles_to_point(q, how):
    """q: [9,B,C,H] at deciles 0.1..0.9 -> point forecast [B,C,H].

    'median' = q50 (minimizes MAE).
    'mean'   = an ESTIMATE of E[X] = int_0^1 Q(p)dp (the MSE-optimal point), via the
               trapezoid rule over the 9 decile knots with flat tails. Weights work
               out to 0.1*[1.5,1,1,1,1,1,1,1,1.5] (sum=1). It is an approximation, not
               an exact conditional mean (Toto exposes only quantiles, no mean head);
               on skewed channels the flat-tail trapezoid is biased and the median can
               score a lower MSE, e.g. Weather-univariate.
    """
    if how == "median":
        return q[4]
    w = torch.tensor([1.5, 1, 1, 1, 1, 1, 1, 1, 1.5], device=q.device, dtype=q.dtype) * 0.1
    return (q * w.view(-1, 1, 1, 1)).sum(0)


def toto_forecast(model, contexts, H, mode, device, dtype, batch, decode_block_size, points=("median",)):
    """contexts [M,C,ctx] -> {point_name: [M,C,H]}. Both mean & median share one forward pass."""
    M, C, ctx = contexts.shape
    acc = {p: [] for p in points}
    for i in range(0, M, batch):
        b = contexts[i:i+batch].to(dtype).to(device)
        B = b.shape[0]
        if mode == "multivariate":
            sid = torch.zeros(B, C, dtype=torch.long, device=device)          # shared id -> channel mixing
        else:
            sid = torch.arange(C, device=device).unsqueeze(0).expand(B, C).contiguous()  # distinct -> independent
        inp = {"target": b, "target_mask": torch.ones(B, C, ctx, dtype=torch.bool, device=device),
               "series_ids": sid}
        kw = {"decode_block_size": decode_block_size} if decode_block_size else {}
        q = model.forecast(inp, horizon=H, **kw)          # [9,B,C,H]  (one forward per batch)
        for p in points:
            acc[p].append(quantiles_to_point(q, p).to(torch.float32).cpu())
        # release MPS/GPU memory each batch to avoid unified-memory buildup -> swap/OOM(NaN)
        del q, b, inp, sid
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
    return {p: torch.cat(acc[p], 0) for p in points}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ETTh1", "ETTm2"])
    ap.add_argument("--modes", nargs="+", default=["multivariate", "univariate"])
    ap.add_argument("--contexts", nargs="+", type=int, default=[512, 1024, 2048])
    ap.add_argument("--horizons", nargs="+", type=int, default=CUTOFFS)
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--size", type=str, default="2.5B")
    ap.add_argument("--decode", type=str, default="single", choices=["single", "block"],
                    help="single = one forward pass for whole horizon (decode_block_size=None, DataDog default, "
                         "recommended for H<=768); block = autoregressive 1-patch (32-step) blocks w/ median feedback")
    ap.add_argument("--point", type=str, default="median", choices=["median", "mean", "both"],
                    help="point forecast from the 9 quantiles: median (MAE-optimal), mean (MSE-optimal), "
                         "or both (both computed from a single forward pass)")
    # --save_preds: persist the exact evaluated windows + their forecasts as .npz, so the
    # comparison is auditable we can plot Toto vs SearchCast on the SAME windows.
    ap.add_argument("--save_preds", type=str, default=None,
                    help="directory to dump per-config .npz {idx, win_start, targets, pred_<point>}")
    ap.add_argument("--out", type=str, default="results/toto.jsonl")
    args = ap.parse_args()

    device = os.environ.get("TOTO_DEVICE", "cpu")
    # Default fp32 (every shipped result is fp32); opt into bf16 only with TOTO_DTYPE=bf16.
    dtype = torch.bfloat16 if os.environ.get("TOTO_DTYPE", "fp32") == "bf16" else torch.float32
    from toto2 import Toto2Model
    t0 = time.time()
    # Pinned Toto-2 weight revisions (HF commit SHAs): the exact weights every shipped result used.
    # If the authors ever delete or re-upload the model the pinned SHA can vanish, so we fall back to
    # revision="main" with a loud warning (numbers may then differ from the paper's pinned run).
    REV = {"22m":  "685e4ae3e2be8d8998025e53dd98e7fdcb296a89",
           "1B":   "1604e1a5242884fb9848f88c4ced14f4dc62d9d3",
           "2.5B": "51a2812bbe449437c01b79c0e425ed578f335f5b"}
    model_id = f"Datadog/Toto-2.0-{args.size}"
    pinned = REV.get(args.size)
    try:
        model = Toto2Model.from_pretrained(model_id, revision=pinned).to(device).to(dtype).eval()
        used_rev = pinned[:12] if pinned else "main (size not pinned)"
    except Exception as e:
        print(f"[toto] WARNING: pinned revision {pinned} for {model_id} is unavailable "
              f"({type(e).__name__}); falling back to revision='main'. Weights may differ from "
              f"the paper's pinned run.", flush=True)
        model = Toto2Model.from_pretrained(model_id, revision="main").to(device).to(dtype).eval()
        used_rev = "main (FALLBACK)"
    # decode mode (fixed for the whole run):
    #   single  -> the whole horizon (ceil(H/patch) patches) is produced in ONE forward pass.
    #   block   -> autoregressive: predict one output-head patch, feed its median forward, repeat.
    #              The block size is one output patch (num_output_patches * patch_size). Rarely used;
    #              single-pass is what all our results use.
    use_dbs = 0 if args.decode == "single" \
              else model.config.num_output_patches * model.config.patch_size
    how = "single-pass (whole horizon, one forward)" if args.decode == "single" \
          else f"block ({use_dbs}-step autoregressive w/ median feedback)"
    print(f"[toto] loaded {args.size} rev={used_rev} in {time.time()-t0:.0f}s  decode={how}", flush=True)

    outpath = os.path.join(os.path.dirname(__file__), args.out)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fout = open(outpath, "a")

    for ds in args.datasets:
        data, ss, se, cols = prepare(os.path.join(BASE, CSV[ds]))
        test_start = ss[2]
        for ctx in args.contexts:
            for H in args.horizons:
                N = n_windows(ss, se, H)
                idx = thin_indices(N, args.M)
                contexts = build_contexts(data, test_start, ctx, idx)   # [M,C,ctx]
                targets = build_targets(data, test_start, H, idx)       # [M,C,H]
                points = ("mean", "median") if args.point == "both" else (args.point,)
                for mode in args.modes:
                    t1 = time.time()
                    preds = toto_forecast(model, contexts, H, mode, device, dtype, args.batch, use_dbs, points)
                    dt = time.time() - t1
                    # Persist the exact windows + forecasts for this config (auditable / plottable).
                    # win_start[k] = absolute row index of forecast origin k; target span is
                    # data[win_start : win_start+H], context is data[win_start-ctx : win_start].
                    if args.save_preds:
                        os.makedirs(args.save_preds, exist_ok=True)
                        # context = last 512 input steps before each origin, so a plot can show
                        # history -> prediction-start -> forecast vs actual from this file alone.
                        np.savez_compressed(
                            os.path.join(args.save_preds, f"toto_{ds}_{args.size}_{mode}_ctx{ctx}_H{H}.npz"),
                            idx=idx, win_start=test_start + idx, targets=targets.numpy(),
                            context=contexts[:, :, -512:].numpy(),
                            **{f"pred_{p}": preds[p].numpy() for p in points})
                    for p in points:
                        pred = preds[p]
                        n_nan = int(torch.isnan(pred).sum())
                        tot = pred.numel()
                        rec = dict(dataset=ds, size=args.size, mode=mode, context=ctx, horizon=H,
                                   decode=args.decode, point=p, device=device, dtype=str(dtype).split(".")[-1],
                                   M=len(idx), N_full=int(N), mse=mse(pred, targets), mae=mae(pred, targets),
                                   n_nan=n_nan, nan_frac=round(n_nan / tot, 5),
                                   sec=round(dt, 1), sec_per_win=round(dt/len(idx), 3))
                        flag = f"  [{n_nan} NaN = {100*n_nan/tot:.1f}%]" if n_nan else ""
                        print(f"[toto] {ds:6s} {mode:12s} ctx={ctx:4d} H={H:3d} {p:6s} MSE={rec['mse']:.4f} "
                              f"MAE={rec['mae']:.4f} ({dt:.0f}s){flag}", flush=True)
                        fout.write(json.dumps(rec) + "\n"); fout.flush()
                    del preds
                    gc.collect()
                    if device == "mps":
                        torch.mps.empty_cache()
    fout.close()
    print("[toto] DONE", flush=True)


if __name__ == "__main__":
    main()

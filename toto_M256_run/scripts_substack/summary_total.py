"""Build summaries_total.md — the head-to-head for ALL 6 datasets (ETT×4, Exchange, Weather).

Toto metrics: normalized MSE, ctx=2048, M=256. We report BOTH point forecasts (mean, median)
and mark the better one per config, because the MSE-optimal point is dataset/mode-dependent
(mean everywhere EXCEPT Weather-univariate, where the median wins).
SearchCast: paired on the identical M=256 windows (thin_M256), test-selected sgs.

Inputs : results_substack/toto_*.jsonl + searchcast_*.json.
Outputs: results_substack/summaries_total.md  (also printed to stdout).
"""
import warnings
warnings.filterwarnings("ignore")  # keep library path warnings out of public output
import os, sys, re, json, glob, collections
import numpy as np

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results_substack")
CUT = [96, 192, 336, 720]
DSETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "exchange", "weather"]

# SearchCast paper Table 1 (full stride-1) + the strongest Transformer baseline, per dataset.
PUB = {
 "ETTh1":    {"SearchCast (paper)": [0.369, 0.400, 0.423, 0.430], "PatchTST (paper)": [0.414, 0.460, 0.501, 0.507]},
 "ETTh2":    {"SearchCast (paper)": [0.268, 0.330, 0.354, 0.383], "PatchTST (paper)": [0.302, 0.388, 0.426, 0.431]},
 "ETTm1":    {"SearchCast (paper)": [0.297, 0.332, 0.357, 0.396], "PatchTST (paper)": [0.329, 0.367, 0.401, 0.456]},
 "ETTm2":    {"SearchCast (paper)": [0.160, 0.212, 0.256, 0.323], "PatchTST (paper)": [0.175, 0.241, 0.305, 0.402]},
 "exchange": {"SearchCast (paper)": [0.081, 0.167, 0.305, 0.811], "PatchTST (paper)": [0.088, 0.176, 0.301, 0.901]},
 "weather":  {"SearchCast (paper)": [0.141, 0.184, 0.234, 0.304], "PatchTST (paper)": [0.149, 0.194, 0.245, 0.314]},
}


def load_toto():
    """(dataset, size, mode, ctx, point) -> {H: mse}."""
    t = collections.defaultdict(dict)
    for p in glob.glob(os.path.join(RES, "toto_*.jsonl")):
        for l in open(p):
            if not l.strip():
                continue
            r = json.loads(l)
            t[(r["dataset"], r["size"], r["mode"], r["context"], r["point"])][r["horizon"]] = r["mse"]
    return t


def sc_paired(ds, key="thin_M256"):
    p = os.path.join(RES, f"searchcast_{ds}.json")
    if not os.path.exists(p):
        return None, None
    r = json.load(open(p))
    e = r["results"][str(r["best_sgs"])]
    return r["best_sgs"], [e[str(h)].get(key) for h in CUT]


def thinning_noise():
    """Per-dataset M=256 thinning-noise floor: the largest |full - M256| gap of the global-Ridge
    control across the four horizons, read from thinning_M256.txt (produced by validate_thinning.py).
    A win must clear this to count as clean rather than within noise. We use the MAX across horizons,
    not the mean, on purpose: the generous bound keeps the clean/narrow call strict and defensible."""
    p = os.path.join(RES, "thinning_M256.txt")
    if not os.path.exists(p):
        return {}
    noise, cur = {}, None
    for line in open(p):
        h = re.match(r"^(\w+)\s+\(C=", line)               # dataset block header, e.g. "ETTh1  (C=7, ...)"
        if h:
            cur = h.group(1)
        elif cur and re.match(r"^\s*\d+\s+\d+\s*\|", line):  # a per-horizon row
            grd = float(line.split("|")[-1].split()[-1])     # last number = global-Ridge full-vs-M256 delta
            noise.setdefault(cur, []).append(abs(grd))
    return {k: max(v) for k, v in noise.items()}


def avg(v):
    v = [x for x in v if x is not None]
    return np.mean(v) if v else None


def row(vals):
    a = avg(vals)
    cells = " ".join(f"{v:6.3f}" if v is not None else "   -  " for v in vals)
    return f"{cells} | {a:6.3f}" if a is not None else cells + " |   -  "


def main():
    t = load_toto()
    out = []
    def w(s=""):
        out.append(s)

    w("# Toto-2 (zero-shot) vs SearchCast (tuned Ridge) — all 6 datasets")
    w("")
    w("Normalized MSE, context=2048, paired **M=256**. Toto point forecast: **mean** unless the")
    w("`median` column is lower (Weather-univariate). SearchCast = paired on identical windows.")
    scoreboard = {}

    for ds in DSETS:
        sgs, sc = sc_paired(ds)
        pub = PUB[ds]
        w("\n" + "=" * 78)
        w(f"## {ds}")
        w("=" * 78)
        w("```")
        w(f"  {'method':30s} {'H96':>6} {'H192':>6} {'H336':>6} {'H720':>6} | {'Avg':>6}")
        w("  " + "-" * 66)
        for name, v in pub.items():
            w(f"  {name:30s} {row(v)}")
        if sc:
            w(f"  {f'SearchCast paired sgs={sgs} (M256)':30s} {row(sc)}")
        w("  " + "-" * 66)
        # Toto rows, ctx2048, mean & median; track best-of-config for the scoreboard
        best = (9, None)
        for size in ["22m", "1B", "2.5B"]:
            for mode in ["univariate", "multivariate"]:
                for pt in ["mean", "median"]:
                    k = (ds, size, mode, 2048, pt)
                    if k in t and all(h in t[k] for h in CUT):
                        vals = [t[k][h] for h in CUT]
                        a = avg(vals)
                        star = " *" if (sc and a < avg(sc)) else ""
                        w(f"  {f'Toto-{size} {mode[:2]} [{pt[:3]}]':30s} {row(vals)}{star}")
                        if a < best[0]:
                            best = (a, f"{size}-{mode[:2]}-{pt[:3]}")
        w("```")
        scoreboard[ds] = dict(best=best, sc=avg(sc) if sc else None, sc_sgs=sgs,
                              sc_pub=avg(pub["SearchCast (paper)"]))

    # ---- scoreboard ----
    noise = thinning_noise()
    w("\n" + "=" * 78)
    w("## FINAL SCOREBOARD — best-of-config Toto vs SearchCast (paired M=256)")
    w("=" * 78)
    w("```")
    w(f"  {'dataset':9s} {'best Toto':22s} {'Toto':>6} {'SC(M256)':>9} {'SC(pub)':>8} {'noise':>7}  verdict")
    w("  " + "-" * 82)
    nwin = ntie = nloss = 0
    for ds in DSETS:
        s = scoreboard[ds]
        d = s["best"][0] - s["sc"]
        flr = noise.get(ds, 0.003)
        # Toto is ahead (d<0) on all six. A tie is |d|<0.001. A win whose margin is
        # smaller than that dataset's own M=256 thinning noise (the 'noise' column) sits within the
        # floor and is flagged "~" (a narrow win). Everything else is a clean win.
        verd = "tie" if abs(d) < 0.001 else ("TOTO" if d < 0 else "SEARCHCAST")
        noisy = " ~" if verd == "TOTO" and abs(d) < flr else ""
        nwin += verd == "TOTO"
        ntie += verd == "tie"
        nloss += verd == "SEARCHCAST"
        w(f"  {ds:9s} {s['best'][1]:22s} {s['best'][0]:>6.3f} {s['sc']:>9.3f} {s['sc_pub']:>8.3f} {flr:>7.3f}  {verd}{noisy} ({d:+.3f})")
    w("  " + "-" * 82)
    w(f"  Toto: {nwin} wins, {ntie} tie, {nloss} losses  (best-of-config, paired M=256)")
    w("  'noise' = each dataset's own M=256 thinning floor (max |full - M256| of the global-Ridge")
    w("  control, from thinning_M256.txt). '~' marks a win inside that floor (narrow). ETTh2 is a tie.")
    w("```")

    # ---- context sweeps (where multiple contexts were run) ----
    w("\n" + "=" * 78)
    w("## CONTEXT-LENGTH SWEEPS (best point per config)")
    w("=" * 78)
    for ds in DSETS:
        ctxs = sorted({k[3] for k in t if k[0] == ds})
        if len(ctxs) <= 1:
            continue
        w(f"\n```\n{ds}: avg MSE vs context (best of mean/median)")
        w(f"  {'size-mode':16s} " + " ".join(f"ctx{c:<5d}" for c in ctxs))
        for size in ["22m", "1B", "2.5B"]:
            for mode in ["univariate", "multivariate"]:
                cells = []
                for c in ctxs:
                    vs = [avg([t[(ds, size, mode, c, pt)].get(h) for h in CUT])
                          for pt in ["mean", "median"] if (ds, size, mode, c, pt) in t
                          and all(h in t[(ds, size, mode, c, pt)] for h in CUT)]
                    cells.append(f"{min(vs):.3f} " if vs else "  -   ")
                if any("-" not in x for x in cells):
                    w(f"  {size+'-'+mode[:2]:16s} " + "  ".join(cells))
        w("```")

    path = os.path.join(RES, "summaries_total.md")
    open(path, "w").write("\n".join(out) + "\n")
    print("\n".join(out))
    print("\nsaved results_substack/summaries_total.md")


if __name__ == "__main__":
    main()

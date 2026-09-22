"""Seed-variance demo for SearchCast: how much does the number move with the RNG seed?

The paper fixes no seed. We fix seed=0 everywhere else. This runs the SAME SearchCast search
(ETTh1, sgs=7, gh=48) for seeds 0/1/2/3 and reports the seed-to-seed spread, to show the paper's
published ETTh1 SearchCast (avg 0.4055) sits inside that band, i.e. the small reproduction gap is
seed noise, not a methodology difference.

Reuses run_searchcast_paired.py (via SC_SEED and a throwaway SC_OUT_DIR) so the search path is identical
and the validated results_substack/searchcast_ETTh1.json is never touched.

Outputs:
  results_substack/seed_variance_ETTh1.json  per-seed mean_full + per-horizon thin_M256 + spread stats.
  stdout                                      a small per-seed table.
Runtime: ~30 min (4 seeds x ~7.5 min for ETTh1 sgs=7 on CPU). No log file is written.
"""
import os, sys, json, subprocess, tempfile, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results_substack")
PY = sys.executable
DS, GH, SGS = "ETTh1", "48", "7"
SEEDS = [0, 1, 2, 3]
CUT = ["96", "192", "336", "720"]
PAPER = [0.369, 0.400, 0.423, 0.430]                        # paper Table 1 ETTh1 SearchCast (published)

os.makedirs(RES, exist_ok=True)
tmp = tempfile.mkdtemp(prefix="sc_seed_")                   # throwaway so the real searchcast_ETTh1.json is safe
out = {"dataset": DS, "gh": int(GH), "sgs": int(SGS),
       "paper_published_per_horizon": PAPER, "paper_published_avg": round(sum(PAPER) / 4, 4),
       "seeds": {}}
for s in SEEDS:
    print(f"[seedvar] running {DS} sgs={SGS} gh={GH} seed={s} ...", flush=True)
    env = {**os.environ, "SC_SEED": str(s), "SC_OUT_DIR": tmp, "HF_HUB_OFFLINE": "1"}
    subprocess.run([PY, os.path.join(HERE, "run_searchcast_paired.py"), DS, GH, SGS],
                   env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    e = json.load(open(os.path.join(tmp, f"searchcast_{DS}.json")))["results"][SGS]
    m256 = {h: e[h]["thin_M256"] for h in CUT}
    out["seeds"][str(s)] = {"mean_full": e["mean_full"],
                            "avg_M256": round(sum(m256.values()) / 4, 4),
                            "per_horizon_M256": m256}

mfs = [out["seeds"][str(s)]["mean_full"] for s in SEEDS]
out["mean_full_across_seeds"] = {"mean": round(statistics.mean(mfs), 4),
                                 "min": round(min(mfs), 4), "max": round(max(mfs), 4),
                                 "std": round(statistics.pstdev(mfs), 4)}
json.dump(out, open(os.path.join(RES, f"seed_variance_{DS}.json"), "w"), indent=1)

print(f"\n[seedvar] {DS} sgs={SGS} gh={GH} -- mean_full (full stride-1) by seed:")
for s in SEEDS:
    print(f"  seed {s}: {out['seeds'][str(s)]['mean_full']:.4f}")
b = out["mean_full_across_seeds"]
inside = b["min"] <= out["paper_published_avg"] <= b["max"]
print(f"  seed band: mean={b['mean']:.4f}  [{b['min']:.4f}, {b['max']:.4f}]  std={b['std']:.4f}")
print(f"  paper published avg = {out['paper_published_avg']:.4f}  ({'inside' if inside else 'outside'} the seed band)")
print(f"saved results_substack/seed_variance_{DS}.json")

# Toto-2 vs SearchCast — reproducible package (AI Horizon Forecast)

A clean, path-anonymized, fully-reproducible repackaging of the whole study: **Can a zero-shot
foundation model (Datadog Toto-2) match/beat a heavily-tuned linear model (SearchCast) on its own
benchmark?** 

Everything runs from this folder. Nothing is machine-specific (paths derive from script
location). To reproduce, follow **Reproduce** below. The head-to-head numbers live in
**`results_substack/summaries_total.md`**. The full guided walkthrough (`NOTEBOOK.ipynb`) lives in
**AI Projects** for subscribers, along with the other files marked ✅ below (linked at the end).

## Layout
```
NOTEBOOK.ipynb          the guided walkthrough  ✅
README.md               this file
traffic_electricity.py / .ipynb   large-channel extension: Toto vs SearchCast on Electricity (321ch) + Traffic (862ch)  ✅
scripts_substack/
  ltsf_common.py          protocol: 6:2:2/7:1:2 split, z-score w/ train stats, rolling windows, NaN-safe MSE
  toto_eval.py            Toto runner  (--point mean|median|both, --save_preds, single-pass decode)
  run_searchcast_paired.py  their REAL Optuna search, scored paired on the identical M windows
  validate_thinning.py    proof that M=256 == full stride-1 (persistence + global-Ridge controls)
  summary_total.py        builds summaries_total.md (all 6 datasets)
  seed_variance.py        re-runs ETTh1 across seeds 0-3 (SearchCast seed-variance demo)
  plot_paired.py          per-window plot: context -> forecast, Toto vs SearchCast (needs preds from toto_eval --save_preds)
  run_ett.sh / run_exchange.sh / run_weather.sh   the 3 canonical orchestrators
  run_gh24.sh             gh=24 robustness re-run (writes results_substack/gh24/)
  SEARCHCAST_EXPLAINED.md / SEARCHCAST_EXPLAINED_SHORT.md   how SearchCast works, end to end  ✅
results_substack/
  toto_ett.jsonl · toto_exchange.jsonl · toto_weather.jsonl   one row per (dataset,size,mode,ctx,horizon,point)
  searchcast_*.json        paired SearchCast (thin_M64/96/256 + full), test-selected sgs
  gh24/searchcast_*.json   the same at gh=24 (robustness re-run)
  seed_variance_ETTh1.json SearchCast seed-variance result (seeds 0-3)
  summaries_total.md       the head-to-head tables + final scoreboard
  toto_elec_traffic.jsonl  large-channel extension results (Toto-2 on Electricity + Traffic, from traffic_electricity.ipynb)
  searchcast_electricity.json  paired SearchCast for Electricity (sgs=32), the extension's SC baseline
  thinning_M256.txt        the M=256 unbiasedness proof
  figures/                 context sweeps, paired windows, and the two case diagrams (svg + png)
INDEX.md files            results_substack/ and logs_substack/ each map every file to its producing script
logs_substack/            de-fragmented per-group run logs
```

## SearchCast model grid (cells per dataset)
SearchCast is a grid of small ridge models, one per `(series group, horizon slice)` cell.
The count is `ceil(C / gs) × (720 / gh)`, where `gs` is each dataset's series grouping and
`gh` is the horizon-slice width.

**What we swept at `gh = 48`, and what SearchCast selected** (the `gs` with the lowest test
MSE, its own test-set selection). The paper does not tabulate a best `gs` per dataset: its
Figure 4 (§5.2) sweeps only four datasets and reports MSE degradation relative to each one's
best size, concluding qualitatively that fully shared is best on the ETT benchmarks and fully
per-series is best on Weather. We recover each `gs` by running the sweep and test-selecting,
as the released code does.

| dataset | C | `gs` values swept | best `gs` (lowest test MSE) | what the paper documents |
|---|---|---|---|---|
| ETTh1 | 7 | 1, 3, 7 | 7 | Figure 4: fully shared (`gs = C`) is best |
| ETTh2 | 7 | 1, 3, 7 | 7 | not in Figure 4, no `gs` given |
| ETTm1 | 7 | 1, 3, 7 | 3 | not in Figure 4, no `gs` given |
| ETTm2 | 7 | 1, 3, 7 | 7 | Figure 4: fully shared (`gs = C`) is best |
| Exchange | 8 | 1, 2, 4, 8 | 4 | not covered |
| Weather | 21 | 1 | 1 | Figure 4: fully per-series (`gs = 1`) is best |

The paper uses `gh = 48` (15 slices). The released `optuna_ridge.py` default is `gh = 24`
(30 slices), which `run_gh24.sh` runs as a robustness check at each dataset's best `gs` from
`gh = 48`. Going from `gh = 48` to `gh = 24` only re-slices the horizon, so it doubles the cell
count and leaves `gs` (and the series-group count) unchanged.

| dataset | C | gh | gs | series groups = ceil(C/gs) | bins = 720/gh | cells |
|---|---|---|---|---|---|---|
| ETTh1 | 7 | 48 | 7 | 1 | 15 | 15 |
| ETTh1 | 7 | 24 | 7 | 1 | 30 | 30 |
| ETTh2 | 7 | 48 | 7 | 1 | 15 | 15 |
| ETTh2 | 7 | 24 | 7 | 1 | 30 | 30 |
| ETTm1 | 7 | 48 | 3 | 3 | 15 | 45 |
| ETTm1 | 7 | 24 | 3 | 3 | 30 | 90 |
| ETTm2 | 7 | 48 | 7 | 1 | 15 | 15 |
| ETTm2 | 7 | 24 | 7 | 1 | 30 | 30 |
| Exchange | 8 | 48 | 4 | 2 | 15 | 30 |
| Exchange | 8 | 24 | 4 | 2 | 30 | 60 |
| Weather | 21 | 48 | 1 | 21 | 15 | 315 |
| Weather | 21 | 24 | 1 | 21 | 30 | 630 |

ETTm1 is kept as a full `{1, 3, 7}` sweep example, and it selects `gs = 3` (the same as at
`gh = 48`). Electricity (C = 321) and Traffic (C = 862) are the large-channel extension, not part
of this 6-dataset table.

## Reproduce
```bash
# from the repo root (…/SearchCast), with the repo's .venv active
bash toto_M256_run/scripts_substack/run_ett.sh        # ETTh1/h2/m1/m2 : Toto (all sizes, both modes) + context sweep + paired SearchCast
bash toto_M256_run/scripts_substack/run_exchange.sh   # Exchange       : Toto (all sizes, both modes) + context sweep + paired SearchCast
bash toto_M256_run/scripts_substack/run_weather.sh    # Weather (21ch) : Toto (all sizes, both modes) + context sweep + paired SearchCast
python toto_M256_run/scripts_substack/validate_thinning.py   # M=256 == stride-1 proof (~minutes)
python toto_M256_run/scripts_substack/summary_total.py       # -> summaries_total.md
```
Full Toto compute is **days** on an M1 (2.5B-multivariate ≈ 14 h/dataset). All results here were
produced by exactly these scripts. The anonymized scripts reproduce cached numbers to 4 decimals.

## Environment and reproducibility
Python **3.12**. The environment is defined by `pyproject.toml` + **`uv.lock`** at the repo root, and
`uv.lock` is the source of truth: it pins every transitive package with hashes and per-platform wheels,
resolved universally across platforms. Reproduce it with one deterministic command:
```bash
uv sync     # builds .venv from uv.lock (exact, cross-platform, hash-verified)
```
For environments where `uv` is inconvenient (Colab, plain-pip CI), **`requirements-lock.txt`** is exported
from that same `uv.lock` (via `uv export --no-hashes --no-emit-project`), so the two can never drift:
```bash
pip install -r requirements-lock.txt
```
Key pins: `toto-2==2.0.0` (its only PyPI release), torch 2.13.0, optuna 4.9.0, numpy 2.4.6. torch is the
CPU/MPS build from PyPI, since every shipped result is CPU/float32. To run the larger Toto sizes on a GPU
(for example a Colab T4), install a CUDA build of the same version over the top:
```bash
uv pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu124
```

**Toto-2 weights are pinned by HuggingFace commit**, so a re-run gets the exact weights every result used:
```
Datadog/Toto-2.0-22m   685e4ae3e2be8d8998025e53dd98e7fdcb296a89
Datadog/Toto-2.0-1B    1604e1a5242884fb9848f88c4ced14f4dc62d9d3
Datadog/Toto-2.0-2.5B  51a2812bbe449437c01b79c0e425ed578f335f5b
```
`toto_eval.py` loads these revisions and, if a pinned commit is ever removed or re-uploaded, falls back
to `revision="main"` with a printed warning. Weights auto-download from HuggingFace on first use (set
`HF_TOKEN` for speed). All Toto runs are **CPU/float32** (the stock `from_pretrained().to("cpu")`).

## Headline
Best-of-config Toto vs SearchCast (paired M=256): **ETTh1 −0.029 · ETTm1 −0.015 · Exchange −0.013 ·
ETTm2 −0.003 · Weather −0.002 · ETTh2 tie**. A win counts as *clean* only when its margin clears that
dataset's own M=256 thinning noise, judged with the generous per-dataset **max** (not the mean) so the
bar stays strict: 3 clean wins (ETTh1, ETTm1, Exchange), 2 narrow wins inside that noise (ETTm2, Weather),
and 1 tie (ETTh2), so **5 wins, 1 tie, 0 losses** against SearchCast on the paired benchmark. Full tables
and verdict in `results_substack/summaries_total.md`.

## Large-channel extension (Electricity + Traffic)
`traffic_electricity.py` / `.ipynb` carry the same comparison to the paper's two biggest datasets. Even
the smallest Toto (22m, zero-shot) in its native multivariate mode ties or beats SearchCast there:
electricity **0.1578 vs 0.1593** (Toto ahead), traffic **0.4051 vs 0.404** (Toto level, within noise), on datasets
where SearchCast is strongest. Results are in `results_substack/toto_elec_traffic.jsonl`.

## In AI Projects (for subscribers)
✅ The subscriber files marked above live in the **AI Projects** folder (Project 37): `NOTEBOOK.ipynb`
(the full step-by-step notebook), `traffic_electricity.py` / `.ipynb` (the Electricity and Traffic
extension), and `SEARCHCAST_EXPLAINED.md` / `SEARCHCAST_EXPLAINED_SHORT.md` (the SearchCast explainer).
The notebook walks the whole comparison end to end, shows exactly how to run every experiment, and adds
practical notes on getting more out of Toto-2. Come join in: https://aihorizonforecast.substack.com/p/ai-projects

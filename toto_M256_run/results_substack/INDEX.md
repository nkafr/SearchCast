# results_substack/ — inverse index (which script writes each file)

Every result file, and the script that produces it. Scripts live in `../scripts_substack/`.
The `run_*.sh` orchestrators are the entry points; they invoke `toto_eval.py` and
`run_searchcast_paired.py` internally (noted as "via …").

| file / pattern | written by | how |
|---|---|---|
| `toto_ett.jsonl` | **run_ett.sh** | via `toto_eval.py --out` (one JSON row per size/mode/ctx/horizon/point) |
| `toto_exchange.jsonl` | **run_exchange.sh** | via `toto_eval.py --out` |
| `toto_weather.jsonl` | **run_weather.sh** | via `toto_eval.py --out` |
| `searchcast_ETTh1.json` `searchcast_ETTh2.json` `searchcast_ETTm1.json` `searchcast_ETTm2.json` | **run_ett.sh** | via `run_searchcast_paired.py` (gh=48, test-selected sgs) |
| `searchcast_exchange.json` | **run_exchange.sh** | via `run_searchcast_paired.py` |
| `searchcast_weather.json` | **run_weather.sh** | via `run_searchcast_paired.py` (sgs=1) |
| `gh24/searchcast_<ds>.json` (6) | **run_gh24.sh** | via `run_searchcast_paired.py` with `SC_OUT_DIR=…/gh24` (gh=24 robustness; does NOT overwrite the gh=48 files) |
| `summaries_total.md` | **summary_total.py** | reads all `toto_*.jsonl` + `searchcast_*.json` + `thinning_M256.txt` (per-dataset noise column) |
| `thinning_M256.txt` | **validate_thinning.py** | persistence + global-Ridge, full-stride-1 vs M=256 |
| `seed_variance_ETTh1.json` | **seed_variance.py** | ETTh1 sgs=7 search for seeds 0/1/2/3 (via `run_searchcast_paired.py` with `SC_SEED`); per-seed MSE + seed-to-seed spread |
| `figures/context_sweeps.png` | the guided notebook (for subscribers) | the context-length figure, from the guided notebook in AI Projects (Project 37) |
| `figures/paired_ETTh1_H96.png` `figures/paired_ETTh1_H720.png` | **plot_paired.py** | reads `preds/toto_*.npz` + `preds/searchcast_*.npz` |
| `figures/demo_window_ETTh1_H720.png` | **plot_paired.py** | Toto-only demo window |
| `figures/pred_case1.svg` | **draw_case1.py** (for subscribers) | how one ETTh1 forecast is tiled by 15 models, and why N shrinks with the horizon (theme-aware SVG) |
| `figures/pred_case2.svg` | **draw_case2.py** (for subscribers) | one model applied to all its windows in a single matmul; the blue block ties back to Case 1 (theme-aware SVG) |
| `figures/pred_case1.png` `figures/pred_case2.png` | (from the SVGs) | light-theme raster of the two Case diagrams for article/Substack upload; re-export from the `.svg` with `qlmanage -t -s 1600` + autocrop when the SVG changes |

Notes:
- `.jsonl` = the canonical **data** (parsed by `summary_total.py` and the notebook). `.log` files
  (in `../logs_substack/`) are human-readable run traces, not parsed for results.
- `preds/*.npz` are per-window forecast dumps used by `plot_paired.py`. They are **not shipped** (too
  large), so `preds/` is empty in this repo. Regenerate them by running `toto_eval.py --save_preds` and
  the runners' `SC_SAVE_PREDS`. Keys are documented in each producing script's `Outputs:` header.

# logs_substack/ inverse index (which script writes each log)

Human-readable run traces (progress and timings). These are **not** parsed for results. The canonical
data is in `../results_substack/*.jsonl` and `*.json`. Scripts are in `../scripts_substack/`.

| log file | written by | contents |
|---|---|---|
| `ett.log` | `run_ett.sh` | Toto `[toto]` per-config lines (MSE/MAE/timing) for the ETT runs. Toto only. |
| `exchange.log` | `run_exchange.sh` | Toto `[toto]` per-config lines for the Exchange runs. Toto only. |
| `weather.log` | `run_weather.sh` | Toto `[toto]` per-config lines for the Weather runs. Toto only. |
| `gh24.log` | `run_gh24.sh` | SearchCast gh=24 robustness stdout for all six datasets (`[gh24]` and `[sc]`, including `TEST SELECTION`). This is the only shipped SearchCast run trace. |

Notes:
- **There are no `sc_<ds>.log` files in this shipped package, and the three group logs above are
  Toto-only.** On a fresh orchestrator run, SearchCast stdout goes to a separate `sc_<ds>.log`
  (`run_ett.sh` line 52 and the exchange/weather equivalents), while Toto stdout goes to the group log.
  The group logs shipped here are a **consolidated Toto trace** of the actual multi-run history (the
  evaluation was run in parts), so they carry no `[ett]`/`[exc]`/`[wx]` orchestration echo lines. For the
  current, canonical SearchCast numbers use `../results_substack/searchcast_*.json` and
  `../results_substack/gh24/searchcast_*.json`, not these logs.
- The `[toto]` result lines in `ett.log` are **one-to-one** with the rows in
  `results_substack/toto_ett.jsonl`. The `[toto]` line omits `size`, so use the jsonl (it has an explicit
  `"size"` field) to tell 22m / 1B / 2.5B apart.
- Because these are a consolidated trace, the line ordering can differ from the jsonl. A single fresh
  `bash run_ett.sh` produces a Toto log in lock-step order with the jsonl and writes the per-dataset
  `sc_<ds>.log` SearchCast traces alongside it.
- `gh24.log`'s ETTm1 lines are the older single-sgs (sgs=7) run. ETTm1 was later re-run as the full
  `{7,3,1}` sweep, which is what `../results_substack/gh24/searchcast_ETTm1.json` holds (it selects
  sgs=3), and `run_gh24.sh` now passes that full grid. Logs are not parsed for results, so the older
  trace is left in place and the json is canonical.

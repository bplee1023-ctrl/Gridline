# Gridline

Grades every NFL player in the 2026 season on a 0–100 scale from advanced, context-adjusted
stats, ranks them within 14 position boards, and refreshes itself after every game window.

* **Grade** — per-snap quality (PFF-style scale; 90+ elite, 60 average).
* **Impact** — season value in expected points above a replacement-level player.
* Free data only (nflverse via `nflreadpy`), $0 to run, one user, one Mac.

Setup: see **[SETUP.md](SETUP.md)**.

## How it fits together

```
nflverse feeds ──► Grade workflow (GitHub Actions, 4×/day) ──► GitHub Pages snapshot ──► Mac app
                     ingest → ledger → adjust/shrink → grade → publish     manifest.json     polls every 30 min,
                     ▲                                                     rankings/ …       caches offline
                     └── baseline_stats.json (2015–2025, `baseline` release)
```

| Path | What it is |
| --- | --- |
| `pipeline/ingest.py` | Loads every feed with nflreadpy; schema contracts fail the run on any change |
| `pipeline/ledger.py`, `roles.py` | Joins pbp + FTN + PFR + NGS + snaps; assigns one position per player |
| `pipeline/models/` | Ridge opponent/situation adjustment, split-half shrinkage, FG and punt models |
| `pipeline/positions/` | Metric catalog and per-position metric code |
| `pipeline/grade.py` | Metrics → components → 0–100 grade, Impact, bootstrap band, minimums |
| `pipeline/publish.py` | Writes `site/2026/…` and `manifest.json` |
| `pipeline/baseline/build.py` | Builds the 2015–2025 baseline and fitted weights; runs the backtests |
| `pipeline/validate.py` | Stability / predictiveness backtests |
| `config/*.yaml` | Weights and priors, NFL minimums, position overrides — editable without code |
| `app/ui/index.html` | The whole interface, one self-contained file (also served as the web fallback) |
| `app/src-tauri/` | Tauri 2 shell: disk cache in `~/Library/Application Support/Gridline`, auto-updater |
| `.github/workflows/` | `grade.yml` (schedule), `release-app.yml` (.dmg), `ci.yml` (tests) |

## Running locally

```bash
pip install -r requirements.txt
python -m pipeline.baseline.build          # once; ~1–3 min after downloads
python -m pipeline.run --force             # grades 2026 into site/
python -m http.server -d site 8000         # open http://localhost:8000
python -m pytest tests                     # unit, golden-snapshot and UI tests
```

## Design-doc deviations (all deliberate, all shown in the app's Methodology page)

* **Dropped metrics** — QB "aggressiveness vs. outcome" (NGS has no play-level outcome to pair
  it with) and CB shadow assignments (not public). Their components were renormalized.
* **Penalties** count every accepted foul charged to the player rather than position-specific
  lists.
* **Bootstrap** resamples a player's games rather than individual plays, because PFR stats
  only exist per game. Shown as half the 5th–95th percentile range.
* **Weights** are fit by non-negative ridge toward the priors, then blended with the priors in
  proportion to how much future value the fit explains (at most 50 %), so a fit with no
  signal can't zero out a component.
* **Game grades** use heavier shrinkage (1.25×k) and their own 0–100 scale, so a 90 is an
  elite single game.
* **DuckDB** is pinned as specified, but v1 didn't need it — Polars handles every join.
* **Season-end participation re-grade** is not built yet: 2026 participation data won't exist
  until after the Super Bowl. Estimated components are labeled everywhere until then.

## Data and licensing

Play-by-play © nflverse/nflfastR (CC-BY 4.0). FTN Data via nflverse (CC-BY-SA 4.0).
Next Gen Stats, Pro-Football-Reference advanced stats and snap counts via nflverse.
Fine for personal use; revisit before any public release.

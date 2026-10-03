"""Scheduled entry point.

    python -m pipeline.run                 # grade if any nflverse feed changed since the live snapshot
    python -m pipeline.run --force         # always re-grade (Thursday stat-correction rebuild)

Exit status 0 with `changed=false` written to $GITHUB_OUTPUT means nothing new: the workflow
skips the deploy and the app keeps the current snapshot.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.request
import warnings
from pathlib import Path

from . import ingest
from .config import BASELINE_DIR, BOARDS, SEASON, SITE_DIR
from .publish import write_snapshot
from .season import compute

log = logging.getLogger("gridline")


def _fetch_json(url: str):
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.load(r)
    except Exception:
        return None


def _gh_output(**kv):
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as f:
            for k, v in kv.items():
                f.write(f"{k}={v}\n")


def drift_report(pages: str | None, out: Path, season: int) -> list[str]:
    """Flag any qualified player whose season grade moved more than 15 points since the last snapshot."""
    if not pages:
        return []
    flags = []
    for pos in BOARDS:
        old = _fetch_json(f"{pages}/{season}/rankings/ALL/{pos}.json")
        new_path = out / str(season) / "rankings" / "ALL" / f"{pos}.json"
        if not old or not new_path.exists():
            continue
        new = json.loads(new_path.read_text())
        prev = {r["gsis_id"]: r["grade"] for r in old.get("rows", [])}
        for r in new["rows"]:
            if r["gsis_id"] in prev and r["meets_minimum"] and abs(r["grade"] - prev[r["gsis_id"]]) > 15:
                flags.append(f"{pos} {r['name']}: {prev[r['gsis_id']]} -> {r['grade']}")
    return flags


def main(argv=None) -> int:
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--season", type=int, default=SEASON)
    ap.add_argument("--out", type=Path, default=SITE_DIR)
    ap.add_argument("--baseline", type=Path, default=BASELINE_DIR / "baseline_stats.json")
    ap.add_argument("--pages-url", default=os.environ.get("GRIDLINE_PAGES_URL"))
    ap.add_argument("--boot", type=int, default=None, help="bootstrap draws (default from weights.yaml)")
    a = ap.parse_args(argv)
    t0 = time.time()

    ts = ingest.feed_timestamps()
    log.info("feed timestamps: %s", ts)
    prev = _fetch_json(f"{a.pages_url}/{a.season}/manifest.json") if a.pages_url else None
    if prev and not a.force and prev.get("feeds") == ts and not any(v.startswith("error") for v in ts.values()):
        log.info("no feed changed since %s; nothing to do", prev.get("generated_at"))
        _gh_output(changed="false")
        return 0

    if not a.baseline.exists():
        log.error("baseline not found at %s — run `python -m pipeline.baseline.build` first", a.baseline)
        return 2
    bs = json.loads(a.baseline.read_text())
    feeds = ingest.load_season(a.season)
    sd = compute(feeds, bs.get("models"))
    manifest = write_snapshot(sd, bs, ts, a.out, n_boot=a.boot)
    for f in drift_report(a.pages_url, a.out, a.season):
        log.warning("DRIFT >15 pts: %s", f)
    log.info("published %d files through week %s in %.0fs", len(manifest["files"]), manifest["through_week"], time.time() - t0)
    _gh_output(changed="true")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Freeze 2025 week 1 inputs and the grades they produce (run once; commit the output).

    python tests/make_golden.py

test_golden.py reruns the pipeline on these frozen inputs and must reproduce the grades
exactly, so any change to the methodology shows up as a deliberate fixture update.
"""
import json
import shutil
import sys
import warnings
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import ingest  # noqa: E402
from pipeline.config import BASELINE_DIR  # noqa: E402

OUT = Path(__file__).parent / "fixtures" / "golden"
WEEK = 1


def main():
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    f = ingest.load_season(2025)
    games = f.schedules.filter((pl.col("season") == 2025) & (pl.col("week") == WEEK))
    gids = games["game_id"].to_list()
    keep = {
        "pbp": f.pbp.filter(pl.col("game_id").is_in(gids)),
        "ftn": f.ftn.filter(pl.col("nflverse_game_id").is_in(gids)),
        "pfr_def": f.pfr_def.filter(pl.col("game_id").is_in(gids)),
        "pfr_pass": f.pfr_pass.filter(pl.col("game_id").is_in(gids)),
        "pfr_rush": f.pfr_rush.filter(pl.col("game_id").is_in(gids)),
        "pfr_rec": f.pfr_rec.filter(pl.col("game_id").is_in(gids)),
        "snaps": f.snaps.filter(pl.col("game_id").is_in(gids)),
        "ngs_pass": f.ngs_pass.filter(pl.col("week") == WEEK),
        "ngs_rush": f.ngs_rush.filter(pl.col("week") == WEEK),
        "ngs_rec": f.ngs_rec.filter(pl.col("week") == WEEK),
        "schedules": games,
    }
    ids = set(keep["snaps"]["pfr_player_id"].to_list())
    pids = set()
    for c in ("passer_player_id", "rusher_player_id", "receiver_player_id", "kicker_player_id", "punter_player_id"):
        pids |= set(keep["pbp"][c].drop_nulls().to_list())
    keep["players"] = f.players.filter(pl.col("pfr_id").is_in(list(ids)) | pl.col("gsis_id").is_in(list(pids)))
    keep["rosters"] = f.rosters.filter(pl.col("pfr_id").is_in(list(ids)) | pl.col("gsis_id").is_in(list(pids)))
    d = f.depth
    first_day = games["gameday"].min()
    keep["depth"] = d.filter(pl.col("dt").str.slice(0, 10) == d.filter(pl.col("dt").str.slice(0, 10) <= first_day)["dt"].str.slice(0, 10).max())
    for k, v in keep.items():
        v.write_parquet(OUT / f"{k}.parquet", compression="zstd")
    shutil.copy(BASELINE_DIR / "baseline_stats.json", OUT / "baseline_stats.json")

    from tests.test_golden import grade_frozen
    expected = grade_frozen()
    (OUT / "expected.json").write_text(json.dumps(expected, indent=0, sort_keys=True))
    print("wrote", OUT, {k: v.height for k, v in keep.items()})


if __name__ == "__main__":
    main()

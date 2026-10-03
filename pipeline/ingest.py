"""Load nflverse feeds with nflreadpy and enforce schema contracts.

Every column the pipeline reads is listed in CONTRACTS. If nflverse renames or drops a
column, or a feed comes back empty, the run fails loudly instead of publishing bad grades;
the last good snapshot stays live.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from .config import DATA_DIR

RELEASE = "https://github.com/nflverse/nflverse-data/releases/download/{tag}/timestamp.json"

# feed key -> nflverse release tag (used for change detection)
FEED_TAGS = {
    "pbp": "pbp",
    "ftn": "ftn_charting",
    "ngs": "nextgen_stats",
    "pfr": "pfr_advstats",
    "snaps": "snap_counts",
    "players": "players",
    "rosters": "rosters",
    "depth": "depth_charts",
}

PBP_COLS = [
    "play_id", "game_id", "season", "season_type", "week", "posteam", "defteam", "home_team",
    "away_team", "down", "ydstogo", "yardline_100", "score_differential", "wp", "qtr",
    "game_seconds_remaining", "play_type", "pass", "rush", "qb_dropback", "qb_scramble", "sack",
    "epa", "success", "air_epa", "yac_epa", "cpoe", "cp", "xyac_mean_yardage", "air_yards",
    "yards_after_catch", "complete_pass", "incomplete_pass", "interception", "fumble",
    "fumble_lost", "fumbled_1_player_id", "passer_player_id", "rusher_player_id",
    "receiver_player_id", "run_location", "run_gap", "field_goal_attempt", "field_goal_result",
    "kick_distance", "extra_point_result", "punt_attempt", "kickoff_attempt",
    "punt_returner_player_id", "kickoff_returner_player_id", "kicker_player_id",
    "punter_player_id", "return_yards", "touchback", "punt_inside_twenty", "punt_fair_catch",
    "penalty", "penalty_player_id", "penalty_type", "penalty_yards", "penalty_team",
    "solo_tackle_1_player_id", "solo_tackle_2_player_id", "assist_tackle_1_player_id",
    "assist_tackle_2_player_id", "tackle_with_assist_1_player_id",
    "tackle_with_assist_2_player_id", "tackle_for_loss_1_player_id",
    "tackle_for_loss_2_player_id", "pass_defense_1_player_id", "pass_defense_2_player_id",
    "interception_player_id", "sack_player_id", "half_sack_1_player_id", "half_sack_2_player_id",
    "roof", "surface", "temp", "wind", "two_point_attempt", "qb_kneel", "qb_spike", "yards_gained",
    "touchdown", "first_down", "aborted_play", "end_yard_line", "game_date",
]

CONTRACTS: dict[str, dict[str, str]] = {
    "pbp": {c: "any" for c in PBP_COLS} | {"epa": "num", "wp": "num", "play_id": "num"},
    "ftn": {
        "nflverse_game_id": "str", "nflverse_play_id": "num", "n_defense_box": "num",
        "is_play_action": "bool", "is_qb_out_of_pocket": "bool", "is_interception_worthy": "bool",
        "is_throw_away": "bool", "read_thrown": "str", "is_catchable_ball": "bool",
        "is_contested_ball": "bool", "is_created_reception": "bool", "is_drop": "bool",
        "is_qb_sneak": "bool", "n_blitzers": "num", "n_pass_rushers": "num",
        "is_qb_fault_sack": "bool",
    },
    "pfr_def": {
        "game_id": "str", "pfr_player_id": "str", "def_ints": "num", "def_targets": "num",
        "def_completions_allowed": "num", "def_yards_allowed": "num",
        "def_receiving_td_allowed": "num", "def_yards_after_catch": "num",
        "def_times_blitzed": "num", "def_times_hurried": "num", "def_times_hitqb": "num",
        "def_sacks": "num", "def_pressures": "num", "def_tackles_combined": "num",
        "def_missed_tackles": "num",
    },
    "pfr_pass": {"game_id": "str", "pfr_player_id": "str", "passing_bad_throws": "num",
                 "times_pressured": "num"},
    "pfr_rush": {"game_id": "str", "pfr_player_id": "str", "carries": "num",
                 "rushing_yards_after_contact": "num", "rushing_broken_tackles": "num",
                 "receiving_broken_tackles": "num"},
    "pfr_rec": {"game_id": "str", "pfr_player_id": "str", "rushing_broken_tackles": "num",
                "receiving_broken_tackles": "num"},
    "snaps": {"game_id": "str", "pfr_player_id": "str", "position": "str", "team": "str",
              "offense_snaps": "num", "defense_snaps": "num", "st_snaps": "num"},
    "ngs_pass": {"season": "num", "week": "num", "season_type": "str", "player_gsis_id": "str",
                 "team_abbr": "str", "avg_time_to_throw": "num", "attempts": "num"},
    "ngs_rush": {"season": "num", "week": "num", "player_gsis_id": "str", "team_abbr": "str",
                 "rush_attempts": "num", "rush_yards_over_expected": "num"},
    "ngs_rec": {"season": "num", "week": "num", "player_gsis_id": "str", "team_abbr": "str",
                "avg_separation": "num", "targets": "num", "receptions": "num",
                "avg_yac_above_expectation": "num"},
    "players": {"gsis_id": "str", "pfr_id": "str", "display_name": "str", "position": "str",
                "headshot": "str", "latest_team": "str"},
    "rosters": {"gsis_id": "str", "pfr_id": "str", "team": "str", "position": "str",
                "full_name": "str"},
    "schedules": {"game_id": "str", "season": "num", "game_type": "str", "week": "num",
                  "gameday": "str", "home_team": "str", "away_team": "str"},
}


class ContractError(RuntimeError):
    pass


def _kind_ok(dtype: pl.DataType, kind: str) -> bool:
    if kind == "any" or dtype == pl.Null:
        return True
    if kind == "num":
        return dtype.is_numeric() or dtype == pl.Boolean
    if kind == "str":
        return dtype in (pl.String, pl.Categorical)
    if kind == "bool":
        return dtype == pl.Boolean or dtype.is_numeric()
    return True


def check_contract(name: str, df: pl.DataFrame, require_rows: bool = True) -> None:
    spec = CONTRACTS.get(name, {})
    missing = [c for c in spec if c not in df.columns]
    if missing:
        raise ContractError(f"{name}: missing columns {missing}")
    bad = [f"{c} ({df.schema[c]})" for c, k in spec.items() if not _kind_ok(df.schema[c], k)]
    if bad:
        raise ContractError(f"{name}: unexpected column types {bad}")
    if require_rows and df.height == 0:
        raise ContractError(f"{name}: feed returned zero rows")


def feed_timestamps() -> dict[str, str]:
    out = {}
    for key, tag in FEED_TAGS.items():
        try:
            with urllib.request.urlopen(RELEASE.format(tag=tag), timeout=30) as r:
                out[key] = json.load(r).get("last_updated", "")
        except Exception as e:  # network hiccup: treat as unknown (forces a run)
            out[key] = f"error:{type(e).__name__}"
    return out


@dataclass
class Feeds:
    season: int
    pbp: pl.DataFrame
    ftn: pl.DataFrame
    pfr_def: pl.DataFrame
    pfr_pass: pl.DataFrame
    pfr_rush: pl.DataFrame
    pfr_rec: pl.DataFrame
    snaps: pl.DataFrame
    ngs_pass: pl.DataFrame
    ngs_rush: pl.DataFrame
    ngs_rec: pl.DataFrame
    players: pl.DataFrame
    rosters: pl.DataFrame
    depth: pl.DataFrame
    schedules: pl.DataFrame
    extra: dict = field(default_factory=dict)


FRAMES = ["pbp", "ftn", "pfr_def", "pfr_pass", "pfr_rush", "pfr_rec", "snaps", "ngs_pass",
          "ngs_rush", "ngs_rec", "players", "rosters", "depth", "schedules"]


def _empty_like(name: str) -> pl.DataFrame:
    return pl.DataFrame({c: [] for c in CONTRACTS.get(name, {})})


def load_season(season: int, frozen_dir: Path | None = None, strict: bool = True) -> Feeds:
    """Load every feed for one season. `frozen_dir` reads parquet snapshots instead (tests)."""
    if frozen_dir is not None:
        frames = {n: pl.read_parquet(Path(frozen_dir) / f"{n}.parquet") for n in FRAMES}
        return Feeds(season=season, **frames)

    import nflreadpy as nfl

    cache = DATA_DIR / "nflreadpy-cache"
    cache.mkdir(parents=True, exist_ok=True)
    nfl.config.update_config(cache_mode="filesystem", cache_dir=cache, cache_duration=3600,
                             verbose=False)

    def safe(name, fn, required=True):
        try:
            df = fn()
        except Exception as e:
            if required and strict:
                raise ContractError(f"{name}: failed to load ({e})") from e
            return _empty_like(name)
        check_contract(name, df, require_rows=required and strict)
        return df

    pbp = safe("pbp", lambda: nfl.load_pbp(season))
    pbp = pbp.select([c for c in PBP_COLS if c in pbp.columns])
    ngs = {}
    for st, key in (("passing", "ngs_pass"), ("rushing", "ngs_rush"), ("receiving", "ngs_rec")):
        df = safe(key, lambda st=st: nfl.load_nextgen_stats(season, st), required=False)
        if df.height and "week" in df.columns:
            df = df.filter((pl.col("season") == season) & (pl.col("week") > 0))
        ngs[key] = df
    depth = safe("depth", lambda: nfl.load_depth_charts(season), required=False)
    return Feeds(
        season=season,
        pbp=pbp,
        ftn=safe("ftn", lambda: nfl.load_ftn_charting(season), required=False),
        pfr_def=safe("pfr_def", lambda: nfl.load_pfr_advstats(season, "def", "week"), required=False),
        pfr_pass=safe("pfr_pass", lambda: nfl.load_pfr_advstats(season, "pass", "week"), required=False),
        pfr_rush=safe("pfr_rush", lambda: nfl.load_pfr_advstats(season, "rush", "week"), required=False),
        pfr_rec=safe("pfr_rec", lambda: nfl.load_pfr_advstats(season, "rec", "week"), required=False),
        snaps=safe("snaps", lambda: nfl.load_snap_counts(season)),
        players=safe("players", lambda: nfl.load_players()),
        rosters=safe("rosters", lambda: nfl.load_rosters(season)),
        depth=depth,
        schedules=safe("schedules", lambda: nfl.load_schedules(season).filter(pl.col("season") == season)),
        **ngs,
    )


def load_pbp_only(seasons: list[int]) -> pl.DataFrame:
    import nflreadpy as nfl
    cache = DATA_DIR / "nflreadpy-cache"
    cache.mkdir(parents=True, exist_ok=True)
    nfl.config.update_config(cache_mode="filesystem", cache_dir=cache, cache_duration=86400 * 30,
                             verbose=False)
    frames = []
    for s in seasons:
        df = nfl.load_pbp(s)
        frames.append(df.select([c for c in PBP_COLS if c in df.columns]))
    return pl.concat(frames, how="diagonal_relaxed")

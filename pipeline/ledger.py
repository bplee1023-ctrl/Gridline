"""Join every feed into one play ledger plus per-player-game tables.

Everything downstream reads from a `Ctx`. IDs are normalized to nflverse `gsis_id`; games to
nflverse `game_id` (e.g. 2026_05_KC_JAX).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import polars as pl

from .config import weights_cfg
from .ingest import Feeds

log = logging.getLogger("gridline")

POST_TYPES = {"WC", "DIV", "CON", "SB"}


@dataclass
class Ctx:
    season: int
    games: pl.DataFrame          # game_id, week, season_type, gameday, home_team, away_team
    plays: pl.DataFrame          # pbp + FTN, with weight `w` and `charted`
    players: pl.DataFrame        # gsis_id, name, headshot, pfr_id, latest_team, nfl_position
    snaps: pl.DataFrame          # gsis_id, game_id, team, off_snaps, def_snaps, st_snaps, pfr_pos
    pfr_def: pl.DataFrame
    pfr_pass: pl.DataFrame
    pfr_rush: pl.DataFrame
    pfr_rec: pl.DataFrame
    ngs_pass: pl.DataFrame
    ngs_rush: pl.DataFrame
    ngs_rec: pl.DataFrame
    team_games: pl.DataFrame     # game_id, team, opp, dropbacks, runs, plays, snaps
    depth: pl.DataFrame
    rosters: pl.DataFrame
    pending_games: list[str]
    models: dict = field(default_factory=dict)
    positions: pl.DataFrame | None = None   # gsis_id, position
    cache: dict = field(default_factory=dict)


def _games(sched: pl.DataFrame, pbp: pl.DataFrame) -> pl.DataFrame:
    played = pbp.select("game_id").unique()
    return (
        sched.join(played, on="game_id", how="inner")
        .with_columns(
            pl.when(pl.col("game_type") == "REG").then(pl.lit("REG")).otherwise(pl.lit("POST"))
            .alias("season_type"),
            pl.col("week").cast(pl.Int32),
        )
        .select("game_id", "week", "season_type", "gameday", "home_team", "away_team",
                *[c for c in ("roof", "surface", "temp", "wind") if c in sched.columns])
        .sort("week", "game_id")
    )


def _pfr_to_gsis(players: pl.DataFrame, rosters: pl.DataFrame) -> pl.DataFrame:
    a = players.select("gsis_id", pl.col("pfr_id").alias("pfr_player_id"))
    b = rosters.select("gsis_id", pl.col("pfr_id").alias("pfr_player_id"))
    return pl.concat([a, b]).drop_nulls().unique(subset="pfr_player_id", keep="first", maintain_order=True)


def _map_pfr(df: pl.DataFrame, idmap: pl.DataFrame) -> pl.DataFrame:
    if df.height == 0:
        return df.with_columns(pl.lit(None, pl.String).alias("gsis_id"))
    return df.join(idmap, on="pfr_player_id", how="left").filter(pl.col("gsis_id").is_not_null())


def _map_ngs(df: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    if df.height == 0:
        return df.with_columns(pl.lit(None, pl.String).alias("game_id"))
    teams = pl.concat([
        games.select("game_id", "week", pl.col("home_team").alias("team")),
        games.select("game_id", "week", pl.col("away_team").alias("team")),
    ])
    return (df.with_columns(pl.col("week").cast(pl.Int32))
            .join(teams, left_on=["week", "team_abbr"], right_on=["week", "team"], how="inner")
            .rename({"player_gsis_id": "gsis_id"}))


def build(feeds: Feeds, models: dict | None = None) -> Ctx:
    cfg = weights_cfg()["garbage_time"]
    pbp = feeds.pbp.filter(pl.col("season_type").is_in(["REG", "POST"]) & pl.col("game_id").is_not_null())
    games = _games(feeds.schedules, pbp)

    ftn = feeds.ftn
    if ftn.height:
        ftn = ftn.select(
            pl.col("nflverse_game_id").alias("game_id"),
            pl.col("nflverse_play_id").cast(pl.Float64).alias("play_id"),
            *[pl.col(c) for c in ("n_defense_box", "is_play_action", "is_qb_out_of_pocket",
                                  "is_interception_worthy", "is_throw_away", "read_thrown",
                                  "is_catchable_ball", "is_contested_ball", "is_created_reception",
                                  "is_drop", "is_qb_sneak", "n_blitzers", "n_pass_rushers",
                                  "is_qb_fault_sack")],
        ).unique(subset=["game_id", "play_id"], keep="first", maintain_order=True).with_columns(pl.lit(True).alias("charted"))
        plays = pbp.join(ftn, on=["game_id", "play_id"], how="left")
    else:
        plays = pbp.with_columns(pl.lit(None, pl.Boolean).alias("charted"))
    bool_cols = [c for c in ("is_play_action", "is_qb_out_of_pocket", "is_interception_worthy",
                             "is_throw_away", "is_catchable_ball", "is_contested_ball",
                             "is_created_reception", "is_drop", "is_qb_sneak", "is_qb_fault_sack")
                 if c in plays.columns]
    plays = plays.with_columns(
        pl.col("charted").fill_null(False),
        *[pl.col(c).cast(pl.Float64) for c in bool_cols],
        pl.when((pl.col("wp") < cfg["wp_low"]) | (pl.col("wp") > cfg["wp_high"]))
        .then(cfg["weight"]).otherwise(1.0).alias("w"),
        (pl.col("posteam") == pl.col("home_team")).cast(pl.Float64).alias("is_home"),
        pl.col("roof").is_in(["dome", "closed"]).cast(pl.Float64).alias("is_dome"),
    ).join(games.select("game_id", pl.col("season_type").alias("_st")), on="game_id", how="left")
    plays = plays.with_columns(pl.coalesce("_st", "season_type").alias("season_type")).drop("_st")

    charted_games = plays.group_by("game_id").agg(pl.col("charted").any().alias("c"))
    pending = sorted(charted_games.filter(~pl.col("c"))["game_id"].to_list())

    p = feeds.players
    players = p.select(
        "gsis_id", pl.col("display_name").alias("name"), "headshot", pl.col("pfr_id"),
        "latest_team", pl.col("position").alias("nfl_position"),
    ).drop_nulls("gsis_id").unique("gsis_id", keep="first", maintain_order=True)
    idmap = _pfr_to_gsis(p, feeds.rosters)

    snaps = _map_pfr(feeds.snaps, idmap)
    snaps = snaps.join(games.select("game_id"), on="game_id", how="inner").select(
        "gsis_id", "game_id", "team", pl.col("position").alias("pfr_pos"),
        pl.col("offense_snaps").fill_null(0).alias("off_snaps"),
        pl.col("defense_snaps").fill_null(0).alias("def_snaps"),
        pl.col("st_snaps").fill_null(0).alias("st_snaps"),
    ).sort("gsis_id", "game_id", "off_snaps", "def_snaps").unique(subset=["gsis_id", "game_id"], keep="last", maintain_order=True)

    # team-game context: offensive dropbacks / designed runs and snap totals
    real = plays.filter(pl.col("play_type").is_in(["pass", "run"]) & (pl.col("qb_kneel") != 1)
                        & (pl.col("qb_spike") != 1))
    tg = real.group_by("game_id", "posteam", "defteam").agg(
        pl.col("qb_dropback").sum().alias("dropbacks"),
        ((pl.col("rush") == 1) & (pl.col("qb_scramble") != 1)).sum().alias("runs"),
    ).rename({"posteam": "team", "defteam": "opp"})
    team_snaps = snaps.group_by("game_id", "team").agg(
        pl.col("off_snaps").max().alias("team_off_snaps"),
        pl.col("def_snaps").max().alias("team_def_snaps"),
    )
    tg = tg.join(team_snaps, on=["game_id", "team"], how="left").with_columns(
        (pl.col("dropbacks") + pl.col("runs")).alias("plays"),
        (pl.col("dropbacks") / (pl.col("dropbacks") + pl.col("runs"))).alias("pass_share"),
    )
    # opponent's view, for defenders: share of snaps facing dropbacks
    tg = tg.join(
        tg.select(pl.col("game_id"), pl.col("team").alias("opp"),
                  pl.col("dropbacks").alias("opp_dropbacks"), pl.col("runs").alias("opp_runs"),
                  pl.col("pass_share").alias("opp_pass_share")),
        on=["game_id", "opp"], how="left",
    )

    return Ctx(
        season=feeds.season, games=games, plays=plays, players=players, snaps=snaps,
        pfr_def=_map_pfr(feeds.pfr_def, idmap), pfr_pass=_map_pfr(feeds.pfr_pass, idmap),
        pfr_rush=_map_pfr(feeds.pfr_rush, idmap), pfr_rec=_map_pfr(feeds.pfr_rec, idmap),
        ngs_pass=_map_ngs(feeds.ngs_pass, games), ngs_rush=_map_ngs(feeds.ngs_rush, games),
        ngs_rec=_map_ngs(feeds.ngs_rec, games), team_games=tg, depth=feeds.depth,
        rosters=feeds.rosters, pending_games=pending, models=models or {},
    )


def player_game_frame(ctx: Ctx) -> pl.DataFrame:
    """One row per player-game with snaps and the team context needed for denominators."""
    return ctx.snaps.join(ctx.team_games, on=["game_id", "team"], how="left").with_columns(
        (pl.col("off_snaps") * pl.col("pass_share").fill_null(0.6)).alias("pass_snaps"),
        (pl.col("off_snaps") * (1 - pl.col("pass_share").fill_null(0.6))).alias("run_snaps"),
        (pl.col("def_snaps") * pl.col("opp_pass_share").fill_null(0.6)).alias("def_pass_snaps"),
        (pl.col("def_snaps") * (1 - pl.col("opp_pass_share").fill_null(0.6))).alias("def_run_snaps"),
    )

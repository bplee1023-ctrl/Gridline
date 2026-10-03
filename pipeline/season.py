"""Compute every metric for one season as a long per-player-game table."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import polars as pl

from . import ledger, roles
from .ingest import Feeds
from .positions import db, dl, k, lb, ol, p, qb, rb, ret, wr_te
from .positions._util import concat, emit, scrimmage

log = logging.getLogger("gridline")


@dataclass
class SeasonData:
    season: int
    long: pl.DataFrame        # gsis_id, game_id, metric, num, den, week, season_type
    counts: pl.DataFrame      # per player-game qualifier counts and snaps
    positions: pl.DataFrame   # gsis_id, position
    ctx: ledger.Ctx


def _team_value_targets(ctx: ledger.Ctx) -> pl.DataFrame:
    """Pseudo-metrics used only as regression targets (weight fit, Impact scale):
    the team's offensive EPA per play while the player was on the field, and the negative of
    the EPA the defense allowed, each scaled by the player's snap share."""
    s = scrimmage(ctx.plays)
    off = s.group_by("game_id", pl.col("posteam").alias("team")).agg(pl.col("epa").sum().alias("o_epa"), pl.len().alias("o_n"))
    de = s.group_by("game_id", pl.col("defteam").alias("team")).agg((-pl.col("epa")).sum().alias("d_epa"), pl.len().alias("d_n"))
    pg = ledger.player_game_frame(ctx).join(off, on=["game_id", "team"], how="left").join(de, on=["game_id", "team"], how="left")
    pg = pg.with_columns((pl.col("off_snaps") / pl.col("team_off_snaps")).clip(0, 1).fill_null(0).alias("osh"),
                         (pl.col("def_snaps") / pl.col("team_def_snaps")).clip(0, 1).fill_null(0).alias("dsh"))
    return concat([
        emit(pg.filter(pl.col("osh") > 0), "gsis_id", "_team_off_epa", pl.col("o_epa") * pl.col("osh"), pl.col("o_n") * pl.col("osh")),
        emit(pg.filter(pl.col("dsh") > 0), "gsis_id", "_team_def_epa", pl.col("d_epa") * pl.col("dsh"), pl.col("d_n") * pl.col("dsh")),
    ])


def counts(ctx: ledger.Ctx) -> pl.DataFrame:
    pl_ = ctx.plays.filter(pl.col("two_point_attempt").fill_null(0) != 1)
    att = pl_.filter((pl.col("sack") != 1) & ((pl.col("complete_pass") == 1) | (pl.col("incomplete_pass") == 1)
                                               | (pl.col("interception") == 1)) & (pl.col("play_type") == "pass"))
    frames = {
        "pass_attempts": att.group_by(pl.col("passer_player_id").alias("gsis_id"), "game_id").len(),
        "rush_attempts": pl_.filter((pl.col("play_type") == "run") & pl.col("rusher_player_id").is_not_null())
            .group_by(pl.col("rusher_player_id").alias("gsis_id"), "game_id").len(),
        "receptions": att.filter(pl.col("complete_pass") == 1)
            .group_by(pl.col("receiver_player_id").alias("gsis_id"), "game_id").len(),
        "targets": att.group_by(pl.col("receiver_player_id").alias("gsis_id"), "game_id").len(),
        "fg_attempts": pl_.filter(pl.col("field_goal_attempt") == 1)
            .group_by(pl.col("kicker_player_id").alias("gsis_id"), "game_id").len(),
        "xp_attempts": pl_.filter(pl.col("extra_point_result").is_not_null())
            .group_by(pl.col("kicker_player_id").alias("gsis_id"), "game_id").len(),
        "kickoffs": pl_.filter(pl.col("kickoff_attempt") == 1)
            .group_by(pl.col("kicker_player_id").alias("gsis_id"), "game_id").len(),
        "punts": pl_.filter(pl.col("punt_attempt") == 1)
            .group_by(pl.col("punter_player_id").alias("gsis_id"), "game_id").len(),
        "punt_returns": pl_.filter((pl.col("punt_attempt") == 1) & (pl.col("punt_fair_catch").fill_null(0) != 1))
            .group_by(pl.col("punt_returner_player_id").alias("gsis_id"), "game_id").len(),
        "kick_returns": pl_.filter((pl.col("kickoff_attempt") == 1) & (pl.col("touchback").fill_null(0) != 1))
            .group_by(pl.col("kickoff_returner_player_id").alias("gsis_id"), "game_id").len(),
    }
    base = ledger.player_game_frame(ctx).select("gsis_id", "game_id", "team", "off_snaps", "def_snaps", "st_snaps",
                                                "team_off_snaps", "team_def_snaps")
    keys = pl.concat([base.select("gsis_id", "game_id")] + [f.select("gsis_id", "game_id") for f in frames.values()]) \
        .drop_nulls().unique()
    out = keys.join(base, on=["gsis_id", "game_id"], how="left")
    for name, f in frames.items():
        out = out.join(f.drop_nulls("gsis_id").rename({"len": name}), on=["gsis_id", "game_id"], how="left")
    out = out.with_columns(pl.col(c).fill_null(0).cast(pl.Float64) for c in list(frames) + ["off_snaps", "def_snaps", "st_snaps"])
    # team for players without a snap row (rare): from pbp possession
    return out.join(ctx.games.select("game_id", "week", "season_type"), on="game_id", how="inner")


def compute(feeds: Feeds, models: dict | None = None) -> SeasonData:
    ctx = ledger.build(feeds, models)
    ctx.positions = roles.assign(ctx)
    families = [("qb", qb), ("rb", rb), ("wr_te", wr_te), ("ol", ol), ("dl", dl), ("lb", lb), ("db", db),
                ("k", k), ("p", p), ("ret", ret)]
    frames = []
    for name, mod in families:
        try:
            frames.append(mod.metrics(ctx))
        except Exception:
            log.exception("metric family %s failed for %s", name, feeds.season)
            raise
    frames.append(_team_value_targets(ctx))
    long = concat(frames).join(ctx.games.select("game_id", "week", "season_type"), on="game_id", how="inner")
    return SeasonData(season=feeds.season, long=long, counts=counts(ctx), positions=ctx.positions, ctx=ctx)

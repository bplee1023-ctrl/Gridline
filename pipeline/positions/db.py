"""Coverage metrics (CB, S, LB).

Coverage snaps are estimated as defensive snaps on opponent dropbacks minus the player's
blitzes until participation data arrives. Shadow assignments are unknown, so yards allowed
are opponent-adjusted at the team receiving-corps level: each target is charged the
difference between that offense's yards per target and the league's.
"""
from __future__ import annotations

import polars as pl

from ..ledger import Ctx, player_game_frame
from ._util import concat, emit, scrimmage


def metrics(ctx: Ctx) -> pl.DataFrame:
    if not ctx.pfr_def.height:
        return concat([])
    pg = player_game_frame(ctx).filter(pl.col("def_snaps") > 0)
    d = ctx.pfr_def.unique(keep="first", maintain_order=True, subset=["gsis_id", "game_id"]).join(
        pg.select("gsis_id", "game_id", "def_pass_snaps"), on=["gsis_id", "game_id"], how="inner"
    ).with_columns(
        (pl.col("def_pass_snaps") - pl.col("def_times_blitzed").fill_null(0)).clip(0, None).alias("cov_snaps"),
        pl.col("def_targets").fill_null(0), pl.col("def_completions_allowed").fill_null(0),
        pl.col("def_yards_allowed").fill_null(0),
    )
    t = scrimmage(ctx.plays).filter((pl.col("pass") == 1) & (pl.col("sack") != 1) & pl.col("receiver_player_id").is_not_null())
    ypt = t.group_by(pl.col("posteam").alias("opponent")).agg(pl.col("yards_gained").mean().alias("opp_ypt"),
                                                               pl.len().alias("n"))
    lg = float(t["yards_gained"].mean() or 0)
    k = 100.0
    ypt = ypt.with_columns(((pl.col("opp_ypt") * pl.col("n") + lg * k) / (pl.col("n") + k) - lg).alias("opp_delta"))
    d = d.join(ypt.select("opponent", "opp_delta"), on="opponent", how="left").with_columns(pl.col("opp_delta").fill_null(0))

    cov = d.filter(pl.col("cov_snaps") > 0)
    tg = d.filter(pl.col("def_targets") > 0)
    out = [
        emit(cov, "gsis_id", "cov_yds_snap", pl.col("def_yards_allowed") - pl.col("def_targets") * pl.col("opp_delta"), pl.col("cov_snaps")),
        emit(cov, "gsis_id", "cov_tgt_rate", pl.col("def_targets"), pl.col("cov_snaps")),
        emit(tg, "gsis_id", "forced_inc_rate", pl.col("def_targets") - pl.col("def_completions_allowed"), pl.col("def_targets")),
        emit(tg, "gsis_id", "int_rate", pl.col("def_ints"), pl.col("def_targets")),
        emit(tg, "gsis_id", "td_allowed", pl.col("def_receiving_td_allowed"), pl.col("def_targets")),
        emit(d.filter(pl.col("def_completions_allowed") > 0), "gsis_id", "yac_allowed", pl.col("def_yards_after_catch"),
             pl.col("def_completions_allowed")),
    ]
    # EPA swing of pass breakups and interceptions (pbp), split among credited defenders
    cols = ["pass_defense_1_player_id", "pass_defense_2_player_id", "interception_player_id"]
    pd_ = pl.concat([t.select("game_id", "play_id", "epa", pl.col(c).alias("gsis_id")) for c in cols]).drop_nulls("gsis_id") \
        .unique(keep="first", maintain_order=True, subset=["game_id", "play_id", "gsis_id"])
    pd_ = pd_.with_columns((-pl.col("epa") / pl.len().over("game_id", "play_id")).alias("swing")) \
        .group_by("gsis_id", "game_id").agg(pl.col("swing").sum())
    pm = cov.select("gsis_id", "game_id", "cov_snaps").join(pd_, on=["gsis_id", "game_id"], how="left").fill_null(0)
    out.append(emit(pm, "gsis_id", "playmaking_epa", pl.col("swing"), pl.col("cov_snaps")))
    return concat(out)

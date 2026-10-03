"""Receiving metrics (WR, TE and RB receiving)."""
from __future__ import annotations

import polars as pl

from ..ledger import Ctx, player_game_frame
from ._util import concat, emit
from .qb import targets


def metrics(ctx: Ctx) -> pl.DataFrame:
    t = targets(ctx)
    rid = "receiver_player_id"
    w = pl.col("w")
    out = [
        emit(t, rid, "rec_epa_tgt", w * pl.col("epa_adj"), w),
        emit(t, rid, "rec_success_tgt", w * pl.col("success_adj"), w),
        emit(t.filter(pl.col("cp").is_not_null()), rid, "cr_oe",
             pl.col("complete_pass").cast(pl.Float64) - pl.col("cp"), pl.lit(1.0)),
    ]
    pg = player_game_frame(ctx).select("gsis_id", "game_id", "pass_snaps")
    per_game = t.group_by(pl.col(rid).alias("gsis_id"), "game_id").agg(
        (w * pl.col("epa_adj")).sum().alias("epa"), pl.len().cast(pl.Float64).alias("tgts"),
        ((pl.col("read_thrown") == "1") & pl.col("charted")).sum().cast(pl.Float64).alias("first_read"),
        pl.col("charted").any().alias("charted"),
    )
    routes = pg.join(per_game, on=["gsis_id", "game_id"], how="left").fill_null(0) \
        .filter(pl.col("pass_snaps") > 0)
    # only players who were targeted at least once in the season get route-based metrics
    has_tgt = per_game.select("gsis_id").unique()
    routes = routes.join(has_tgt, on="gsis_id", how="semi")
    out.append(emit(routes, "gsis_id", "rec_epa_route", pl.col("epa"), pl.col("pass_snaps")))
    out.append(emit(routes, "gsis_id", "targets_route", pl.col("tgts"), pl.col("pass_snaps")))
    game_charted = ctx.plays.group_by("game_id").agg(pl.col("charted").any().alias("gc"))
    out.append(emit(routes.join(game_charted, on="game_id").filter(pl.col("gc")), "gsis_id", "first_read_share",
                    pl.col("first_read"), pl.col("pass_snaps")))

    ch = t.filter(pl.col("charted"))
    out.append(emit(ch.filter(pl.col("is_catchable_ball") == 1), rid, "drop_rate", pl.col("is_drop").fill_null(0), pl.lit(1.0)))
    out.append(emit(ch.filter(pl.col("is_contested_ball") == 1), rid, "contested_win",
                    pl.col("complete_pass").cast(pl.Float64), pl.lit(1.0)))
    out.append(emit(ch, rid, "created_rec", pl.col("is_created_reception").fill_null(0), pl.lit(1.0)))

    # YAC over expected: NGS weekly when available, otherwise nflfastR's xYAC model
    catches = t.filter((pl.col("complete_pass") == 1) & pl.col("xyac_mean_yardage").is_not_null())
    yac = catches.group_by(pl.col(rid).alias("gsis_id"), "game_id").agg(
        (pl.col("yards_after_catch").fill_null(0) - pl.col("xyac_mean_yardage")).sum().alias("num"),
        pl.len().cast(pl.Float64).alias("den"))
    if ctx.ngs_rec.height:
        ngs = ctx.ngs_rec.select(
            "gsis_id", "game_id",
            (pl.col("avg_yac_above_expectation") * pl.col("receptions")).cast(pl.Float64).alias("n_num"),
            pl.col("receptions").cast(pl.Float64).alias("n_den"),
            (pl.col("avg_separation") * pl.col("targets")).cast(pl.Float64).alias("sep_num"),
            pl.col("targets").cast(pl.Float64).alias("sep_den"),
        ).unique(keep="first", maintain_order=True, subset=["gsis_id", "game_id"])
        yac = yac.join(ngs.select("gsis_id", "game_id", "n_num", "n_den"), on=["gsis_id", "game_id"], how="left").with_columns(
            pl.when(pl.col("n_num").is_not_null() & (pl.col("n_den") > 0)).then("n_num").otherwise("num").alias("num"),
            pl.when(pl.col("n_num").is_not_null() & (pl.col("n_den") > 0)).then("n_den").otherwise("den").alias("den"))
        out.append(emit(ngs.drop_nulls(["sep_num"]).filter(pl.col("sep_den") > 0), "gsis_id", "separation",
                        pl.col("sep_num"), pl.col("sep_den")))
    out.append(emit(yac, "gsis_id", "yacoe", pl.col("num"), pl.col("den")))

    if ctx.pfr_rec.height:
        rec = catches.group_by(pl.col(rid).alias("gsis_id"), "game_id").len("rec").with_columns(pl.col("rec").cast(pl.Float64))
        bt = ctx.pfr_rec.select("gsis_id", "game_id", pl.col("receiving_broken_tackles").fill_null(0).cast(pl.Float64).alias("bt")) \
            .unique(keep="first", maintain_order=True, subset=["gsis_id", "game_id"]).join(rec, on=["gsis_id", "game_id"], how="inner")
        out.append(emit(bt, "gsis_id", "broken_tackles_rec", pl.col("bt"), pl.col("rec")))
    return concat(out)

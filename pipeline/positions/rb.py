"""Running back (and general rushing / ball-security) metrics."""
from __future__ import annotations

import polars as pl

from ..ledger import Ctx
from ._util import concat, emit, scrimmage
from .qb import rushes, targets


def metrics(ctx: Ctx) -> pl.DataFrame:
    r = rushes(ctx).filter(pl.col("qb_scramble") != 1)
    rid = "rusher_player_id"
    w = pl.col("w")
    out = [
        emit(r, rid, "rush_epa", w * pl.col("epa_adj"), w),
        emit(r, rid, "rush_success", w * pl.col("success_adj"), w),
    ]
    # RYOE: NGS where the player qualified that week, else pbp-derived over-expected yards
    lg_yds = float(r["yds"].mean() or 0)
    pbp_ryoe = r.group_by(pl.col(rid).alias("gsis_id"), "game_id").agg(
        (pl.col("yds_adj") - lg_yds).sum().alias("num"), pl.len().cast(pl.Float64).alias("den"))
    if ctx.ngs_rush.height:
        ngs = ctx.ngs_rush.select("gsis_id", "game_id",
                                  pl.col("rush_yards_over_expected").cast(pl.Float64).alias("n_num"),
                                  pl.col("rush_attempts").cast(pl.Float64).alias("n_den")) \
            .drop_nulls().unique(keep="first", maintain_order=True, subset=["gsis_id", "game_id"])
        pbp_ryoe = pbp_ryoe.join(ngs, on=["gsis_id", "game_id"], how="left").with_columns(
            pl.when(pl.col("n_num").is_not_null() & (pl.col("n_den") > 0)).then("n_num").otherwise("num").alias("num"),
            pl.when(pl.col("n_num").is_not_null() & (pl.col("n_den") > 0)).then("n_den").otherwise("den").alias("den"))
    out.append(emit(pbp_ryoe, "gsis_id", "ryoe", pl.col("num"), pl.col("den")))

    # contact & elusiveness (PFR)
    t = targets(ctx)
    rec = t.filter(pl.col("complete_pass") == 1).group_by(pl.col("receiver_player_id").alias("gsis_id"), "game_id") \
        .len("rec").with_columns(pl.col("rec").cast(pl.Float64))
    car = r.group_by(pl.col(rid).alias("gsis_id"), "game_id").len("car").with_columns(pl.col("car").cast(pl.Float64))
    touches = car.join(rec, on=["gsis_id", "game_id"], how="full", coalesce=True).fill_null(0) \
        .with_columns((pl.col("car") + pl.col("rec")).alias("touches"))
    if ctx.pfr_rush.height:
        pr = ctx.pfr_rush.select("gsis_id", "game_id", pl.col("carries").cast(pl.Float64).alias("pfr_car"),
                                 pl.col("rushing_yards_after_contact").cast(pl.Float64).alias("yaco"),
                                 (pl.col("rushing_broken_tackles").fill_null(0) + pl.col("receiving_broken_tackles").fill_null(0)).alias("btk"))
        out.append(emit(pr.filter(pl.col("pfr_car") > 0), "gsis_id", "yac_carry", pl.col("yaco"), pl.col("pfr_car")))
        bt = touches.join(pr, on=["gsis_id", "game_id"], how="inner")
        out.append(emit(bt, "gsis_id", "broken_tackles_touch", pl.col("btk"), pl.col("touches")))

    # fumbles per touch (lost = 1, recovered = 1/2)
    sc = scrimmage(ctx.plays).filter(
        pl.col("fumbled_1_player_id").is_not_null()
        & (((pl.col("fumbled_1_player_id") == pl.col("rusher_player_id")) & (pl.col("qb_dropback").fill_null(0) == 0))
           | (pl.col("fumbled_1_player_id") == pl.col("receiver_player_id"))))
    fum = sc.group_by(pl.col("fumbled_1_player_id").alias("gsis_id"), "game_id").agg(
        pl.when(pl.col("fumble_lost") == 1).then(1.0).otherwise(0.5).sum().alias("fum"))
    ft = touches.join(fum, on=["gsis_id", "game_id"], how="left").with_columns(pl.col("fum").fill_null(0))
    out.append(emit(ft.filter(pl.col("touches") > 0), "gsis_id", "fumbles_touch", pl.col("fum"), pl.col("touches")))

    # short yardage & goal line: 3rd/4th and <=2, or snaps inside the 5
    sy = r.filter((pl.col("down").is_in([3, 4]) & (pl.col("ydstogo") <= 2)) | (pl.col("yardline_100") <= 5)) \
        .with_columns(
            pl.when(pl.col("yardline_100") <= 5).then(pl.col("touchdown") == 1)
            .otherwise((pl.col("first_down") == 1) | (pl.col("touchdown") == 1)).cast(pl.Float64).alias("conv"),
            pl.when(pl.col("yardline_100") <= 5).then(pl.lit("gl")).when(pl.col("ydstogo") <= 1).then(pl.lit("1"))
            .otherwise(pl.lit("2")).alias("bucket"))
    if sy.height:
        sy = sy.join(sy.group_by("bucket").agg(pl.col("conv").mean().alias("exp")), on="bucket")
        out.append(emit(sy, rid, "short_yardage_oe", pl.col("conv") - pl.col("exp"), pl.lit(1.0)))

    ctx.cache["touches"] = touches
    return concat(out)

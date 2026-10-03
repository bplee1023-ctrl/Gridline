"""Quarterback metrics, plus the shared opponent-adjusted play frames other positions reuse."""
from __future__ import annotations

import numpy as np
import polars as pl

from ..ledger import Ctx, player_game_frame
from ..models.adjust import adjust
from ._util import concat, emit, penalties, scrimmage


def dropbacks(ctx: Ctx) -> pl.DataFrame:
    """Passing dropbacks (scrambles excluded: they count as QB rushing). Sacks are charged to
    the QB only when FTN marks them QB-fault; uncharted games keep every sack until charted."""
    if "dropbacks" in ctx.cache:
        return ctx.cache["dropbacks"]
    d = scrimmage(ctx.plays).filter(
        (pl.col("qb_dropback") == 1) & (pl.col("qb_scramble") != 1) & pl.col("passer_player_id").is_not_null()
        & ((pl.col("sack") != 1) | ~pl.col("charted") | (pl.col("is_qb_fault_sack").fill_null(0) == 1))
    ).with_columns(pl.col("success").cast(pl.Float64))
    d = adjust(d, "epa", "passer_player_id", "defteam")
    d = adjust(d, "success", "passer_player_id", "defteam")
    ctx.cache["dropbacks"] = d
    return d


def rushes(ctx: Ctx) -> pl.DataFrame:
    """Every rush (designed runs and scrambles), adjusted for opponent, situation and box count."""
    if "rushes" in ctx.cache:
        return ctx.cache["rushes"]
    r = scrimmage(ctx.plays).filter(((pl.col("rush") == 1) | (pl.col("qb_scramble") == 1))
                                    & pl.col("rusher_player_id").is_not_null()) \
        .with_columns(pl.col("success").cast(pl.Float64),
                      pl.col("n_defense_box").cast(pl.Float64).alias("box"),
                      pl.col("yards_gained").cast(pl.Float64).clip(-15, 50).alias("yds"),
                      pl.col("qb_scramble").fill_null(0).alias("qb_scramble"))
    # adjusted line yards: losses x1.2, 0-4 yds full credit, 5-10 half credit, 11+ none
    y = pl.col("yards_gained").cast(pl.Float64).fill_null(0)
    r = r.with_columns(
        pl.when(y < 0).then(1.2 * y).when(y <= 4).then(y).when(y <= 10).then(4 + 0.5 * (y - 4))
        .otherwise(7.0).alias("aly"))
    for v in ("epa", "success", "yds", "aly"):
        r = adjust(r, v, "rusher_player_id", "defteam", extra=["box"])
    ctx.cache["rushes"] = r
    return r


def targets(ctx: Ctx) -> pl.DataFrame:
    if "targets" in ctx.cache:
        return ctx.cache["targets"]
    t = scrimmage(ctx.plays).filter(
        (pl.col("pass") == 1) & (pl.col("sack") != 1) & pl.col("receiver_player_id").is_not_null()
    ).with_columns(pl.col("success").cast(pl.Float64))
    t = adjust(t, "epa", "receiver_player_id", "defteam")
    t = adjust(t, "success", "receiver_player_id", "defteam")
    ctx.cache["targets"] = t
    return t


def metrics(ctx: Ctx) -> pl.DataFrame:
    d = dropbacks(ctx)
    pid = "passer_player_id"
    w = pl.col("w")
    out = [
        emit(d, pid, "qb_epa_db", w * pl.col("epa_adj"), w),
        emit(d, pid, "qb_success", w * pl.col("success_adj"), w),
    ]
    att = d.filter((pl.col("sack") != 1) & ((pl.col("complete_pass") == 1) | (pl.col("incomplete_pass") == 1)
                                             | (pl.col("interception") == 1)))
    out.append(emit(att.filter(pl.col("air_epa").is_not_null()), pid, "qb_air_epa_att", w * pl.col("air_epa"), w))
    out.append(emit(att.filter(pl.col("cpoe").is_not_null()), pid, "qb_cpoe", pl.col("cpoe"), pl.lit(1.0)))
    charted_att = att.filter(pl.col("charted") & (pl.col("is_throw_away").fill_null(0) != 1))
    out.append(emit(charted_att, pid, "qb_catchable", pl.col("is_catchable_ball").fill_null(0), pl.lit(1.0)))
    out.append(emit(att.filter(pl.col("charted")), pid, "qb_int_worthy",
                    pl.col("is_interception_worthy").fill_null(0), pl.lit(1.0)))
    out.append(emit(charted_att.filter(pl.col("read_thrown") == "2"), pid, "qb_epa_progression",
                    w * pl.col("epa_adj"), w))

    # bad throws (PFR, per game) over pbp attempts that weren't throwaways
    if ctx.pfr_pass.height:
        a = att.filter(pl.col("is_throw_away").fill_null(0) != 1).group_by(pid, "game_id").len("att") \
            .rename({pid: "gsis_id"})
        bt = ctx.pfr_pass.select("gsis_id", "game_id", pl.col("passing_bad_throws").fill_null(0).alias("bt")) \
            .join(a, on=["gsis_id", "game_id"], how="inner")
        out.append(emit(bt, "gsis_id", "qb_bad_throw", pl.col("bt"), pl.col("att")))

    # fumbles per dropback (all dropbacks incl. every sack and scrambles)
    alldb = scrimmage(ctx.plays).filter((pl.col("qb_dropback") == 1) & pl.col("passer_player_id").is_not_null())
    alldb = alldb.with_columns(pl.coalesce("passer_player_id", "rusher_player_id").alias("qb"))
    out.append(emit(alldb, "qb", "qb_fumble_db",
                    (pl.col("fumbled_1_player_id") == pl.col("qb")).cast(pl.Float64).fill_null(0)
                    * pl.when(pl.col("fumble_lost") == 1).then(1.0).otherwise(0.5), pl.lit(1.0)))
    charted_db = alldb.filter(pl.col("charted") & (pl.col("qb_scramble") != 1))
    out.append(emit(charted_db, "qb", "qb_fault_sack", pl.col("is_qb_fault_sack").fill_null(0), pl.lit(1.0)))
    out.append(emit(d.filter((pl.col("n_blitzers").fill_null(0) > 0) | (pl.col("n_pass_rushers").fill_null(0) >= 5)),
                    pid, "qb_epa_pressure_look", w * pl.col("epa_adj"), w))
    out.append(emit(d.filter(pl.col("is_qb_out_of_pocket").fill_null(0) == 1), pid, "qb_oop_epa",
                    w * pl.col("epa_adj"), w))

    # sack rate over expected given time to throw (QB-game level, NGS weekly)
    if ctx.ngs_pass.height:
        sg = alldb.filter(pl.col("qb_scramble") != 1).group_by("qb", "game_id").agg(
            pl.col("sack").sum().cast(pl.Float64).alias("sacks"), pl.len().cast(pl.Float64).alias("db")
        ).rename({"qb": "gsis_id"}).join(
            ctx.ngs_pass.select("gsis_id", "game_id", "avg_time_to_throw"), on=["gsis_id", "game_id"], how="inner"
        ).drop_nulls("avg_time_to_throw").filter(pl.col("db") >= 5)
        if sg.height >= 20:
            x, y, wt = sg["avg_time_to_throw"].to_numpy(), (sg["sacks"] / sg["db"]).to_numpy(), sg["db"].to_numpy()
            b1, b0 = np.polyfit(x, y, 1, w=np.sqrt(wt))
            sg = sg.with_columns((pl.col("sacks") - pl.col("db") * (b0 + b1 * pl.col("avg_time_to_throw"))).alias("soe"))
            out.append(emit(sg, "gsis_id", "qb_sack_oe_ttt", pl.col("soe"), pl.col("db")))

    # rushing
    r = rushes(ctx)
    rid = "rusher_player_id"
    designed = r.filter(pl.col("qb_scramble") != 1)
    scr = r.filter(pl.col("qb_scramble") == 1)
    out.append(emit(designed, rid, "qb_designed_run_epa", w * pl.col("epa_adj"), w))
    out.append(emit(scr, rid, "qb_scramble_epa", w * pl.col("epa_adj"), w))
    out.append(emit(scr, rid, "qb_scramble_success", w * pl.col("success"), w))
    sneaks = r.filter(pl.col("is_qb_sneak").fill_null(0) == 1).with_columns(
        ((pl.col("first_down") == 1) | (pl.col("touchdown") == 1)).cast(pl.Float64).alias("conv"),
        pl.when(pl.col("ydstogo") <= 1).then(pl.lit("1")).otherwise(pl.lit("2+")).alias("bucket"))
    if sneaks.height:
        exp = sneaks.group_by("bucket").agg(pl.col("conv").mean().alias("exp"))
        sneaks = sneaks.join(exp, on="bucket")
        out.append(emit(sneaks, rid, "qb_sneak_oe", pl.col("conv") - pl.col("exp"), pl.lit(1.0)))

    out.append(penalty_metrics(ctx))
    return concat(out)


def penalty_metrics(ctx: Ctx) -> pl.DataFrame:
    """Penalty yards per 100 offensive / defensive snaps (shared by every position)."""
    if "pen" in ctx.cache:
        return ctx.cache["pen"]
    pen = penalties(ctx.plays).group_by("gsis_id", "game_id").agg(pl.col("pen_yds").sum())
    pg = player_game_frame(ctx).select("gsis_id", "game_id", "off_snaps", "def_snaps") \
        .join(pen, on=["gsis_id", "game_id"], how="left").with_columns(pl.col("pen_yds").fill_null(0))
    res = concat([
        emit(pg.filter(pl.col("off_snaps") > 0), "gsis_id", "pen_off", 100 * pl.col("pen_yds"), pl.col("off_snaps")),
        emit(pg.filter(pl.col("def_snaps") > 0), "gsis_id", "pen_def", 100 * pl.col("pen_yds"), pl.col("def_snaps")),
    ])
    ctx.cache["pen"] = res
    return res

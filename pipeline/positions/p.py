"""Punter metrics against expectation models on line of scrimmage and conditions."""
from __future__ import annotations

import polars as pl

from ..ledger import Ctx
from ..models import punt_model
from ._util import concat, emit

TARGETS = {"punt_epa": "epa", "punt_in20": "punt_inside_twenty", "punt_tb": "touchback",
           "punt_fc": "punt_fair_catch", "punt_ret": "return_yards", "pr_value": "pr_value"}


def punts(plays: pl.DataFrame) -> pl.DataFrame:
    return plays.filter((pl.col("punt_attempt") == 1) & pl.col("epa").is_not_null()).with_columns(
        pl.col("punt_inside_twenty").cast(pl.Float64).fill_null(0), pl.col("touchback").cast(pl.Float64).fill_null(0),
        pl.col("punt_fair_catch").cast(pl.Float64).fill_null(0), pl.col("return_yards").cast(pl.Float64).fill_null(0),
        (-pl.col("epa")).alias("pr_value"))


def models_from(plays: pl.DataFrame) -> dict:
    pu = punts(plays)
    m = {f"m_{k}": punt_model.fit(pu, v) for k, v in TARGETS.items() if k != "pr_value"}
    m["m_pr_value"] = punt_model.fit(pu.filter(pl.col("punt_returner_player_id").is_not_null()
                                               & (pl.col("punt_fair_catch") != 1)), "pr_value")
    return m


def metrics(ctx: Ctx) -> pl.DataFrame:
    pu = punts(ctx.plays).filter(pl.col("punter_player_id").is_not_null())
    if not pu.height:
        return concat([])
    models = ctx.models if "m_punt_epa" in ctx.models else models_from(ctx.plays)
    pu = pu.with_columns(*[pl.Series(f"x_{k}", punt_model.predict(pu, models[f"m_{k}"]))
                           for k in ("punt_epa", "punt_in20", "punt_tb", "punt_fc", "punt_ret")])
    pid = "punter_player_id"
    one = pl.lit(1.0)
    return concat([
        emit(pu, pid, "punt_ep_oe", pl.col("epa") - pl.col("x_punt_epa"), one),
        emit(pu, pid, "punt_in20_oe", pl.col("punt_inside_twenty") - pl.col("x_punt_in20"), one),
        emit(pu, pid, "punt_tb_oe", pl.col("touchback") - pl.col("x_punt_tb"), one),
        emit(pu, pid, "punt_fc_oe", pl.col("punt_fair_catch") - pl.col("x_punt_fc"), one),
        # half the return yards are credited to the coverage unit
        emit(pu, pid, "punt_ret_yds_oe", -0.5 * (pl.col("return_yards") - pl.col("x_punt_ret")), one),
    ])

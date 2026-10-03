"""Kick and punt returners (one combined board)."""
from __future__ import annotations

import polars as pl

from ..config import weights_cfg
from ..ledger import Ctx
from ..models import punt_model
from ._util import concat, emit
from .p import models_from, punts


def returns(ctx: Ctx) -> pl.DataFrame:
    if "returns" in ctx.cache:
        return ctx.cache["returns"]
    thr = weights_cfg()["explosive_return_yards"]
    models = ctx.models if "m_pr_value" in ctx.models else models_from(ctx.plays)
    pr = punts(ctx.plays).filter(pl.col("punt_returner_player_id").is_not_null() & (pl.col("punt_fair_catch") != 1))
    pr = pr.with_columns(pl.Series("x", punt_model.predict(pr, models["m_pr_value"])) if pr.height else pl.lit(0.0).alias("x")) \
        .select("game_id", "play_id", pl.col("punt_returner_player_id").alias("gsis_id"),
                (pl.col("pr_value") - pl.col("x")).alias("v"),
                (pl.col("return_yards") >= thr["punt"]).cast(pl.Float64).alias("expl"),
                "fumbled_1_player_id", "fumble_lost", pl.lit("PR").alias("kind"))
    kr = ctx.plays.filter((pl.col("kickoff_attempt") == 1) & pl.col("kickoff_returner_player_id").is_not_null()
                          & pl.col("epa").is_not_null() & (pl.col("touchback").fill_null(0) != 1))
    mean = float(kr["epa"].mean() or 0) if kr.height else 0.0
    kr = kr.select("game_id", "play_id", pl.col("kickoff_returner_player_id").alias("gsis_id"),
                   (pl.col("epa") - mean).alias("v"),
                   (pl.col("return_yards").fill_null(0) >= thr["kick"]).cast(pl.Float64).alias("expl"),
                   "fumbled_1_player_id", "fumble_lost", pl.lit("KR").alias("kind"))
    r = pl.concat([pr, kr]).with_columns(
        pl.when(pl.col("fumbled_1_player_id") == pl.col("gsis_id"))
        .then(pl.when(pl.col("fumble_lost") == 1).then(1.0).otherwise(0.5)).otherwise(0.0).alias("fum"))
    ctx.cache["returns"] = r
    return r


def metrics(ctx: Ctx) -> pl.DataFrame:
    r = returns(ctx)
    one = pl.lit(1.0)
    return concat([
        emit(r, "gsis_id", "ret_epa_oe", pl.col("v"), one),
        emit(r, "gsis_id", "explosive_ret", pl.col("expl"), one),
        emit(r, "gsis_id", "ret_fumble", pl.col("fum"), one),
    ])

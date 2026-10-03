"""Kicker metrics: field goals and extra points over expected, kickoffs vs expectation."""
from __future__ import annotations

import polars as pl

from ..ledger import Ctx
from ..models import fg_model
from ._util import concat, emit

KO_ERRORS = ["Kickoff Out of Bounds", "Kickoff Short of Landing Zone"]


def models_from(plays: pl.DataFrame) -> dict:
    fg = plays.filter((pl.col("field_goal_attempt") == 1) & pl.col("field_goal_result").is_not_null()) \
        .with_columns((pl.col("field_goal_result") == "made").cast(pl.Int32).alias("made"))
    xp = plays.filter(pl.col("extra_point_result").is_in(["good", "failed", "blocked"])) \
        .with_columns((pl.col("extra_point_result") == "good").cast(pl.Int32).alias("made"))
    return {"fg": fg_model.fit(fg, "made"), "xp": fg_model.fit(xp, "made")}


def metrics(ctx: Ctx) -> pl.DataFrame:
    p = ctx.plays
    models = ctx.models if "fg" in ctx.models else models_from(p)
    out = []
    fg = p.filter((pl.col("field_goal_attempt") == 1) & pl.col("field_goal_result").is_not_null()
                  & pl.col("kicker_player_id").is_not_null() & pl.col("kick_distance").is_not_null())
    if fg.height:
        fg = fg.with_columns(fg_model.predict(fg, models["fg"]).alias("p"),
                             (pl.col("field_goal_result") == "made").cast(pl.Float64).alias("made"))
        out.append(emit(fg, "kicker_player_id", "fg_poe", 3 * (pl.col("made") - pl.col("p")), pl.lit(1.0)))
    xp = p.filter(pl.col("extra_point_result").is_in(["good", "failed", "blocked"])
                  & pl.col("kicker_player_id").is_not_null() & pl.col("kick_distance").is_not_null())
    if xp.height:
        xp = xp.with_columns(fg_model.predict(xp, models["xp"]).alias("p"),
                             (pl.col("extra_point_result") == "good").cast(pl.Float64).alias("made"))
        out.append(emit(xp, "kicker_player_id", "xp_oe", pl.col("made") - pl.col("p"), pl.lit(1.0)))
    ko = p.filter((pl.col("kickoff_attempt") == 1) & pl.col("kicker_player_id").is_not_null() & pl.col("epa").is_not_null())
    if ko.height:
        # expectation = league kickoff result this season (kickoff rules changed in 2024 and 2025)
        mean = float(ko["epa"].mean())
        out.append(emit(ko, "kicker_player_id", "ko_epa_oe", -(pl.col("epa") - mean), pl.lit(1.0)))
        out.append(emit(ko, "kicker_player_id", "ko_oob", pl.col("penalty_type").is_in(KO_ERRORS).cast(pl.Float64).fill_null(0),
                        pl.lit(1.0)))
    return concat(out)

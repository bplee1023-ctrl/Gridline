"""Linebacker-specific metric: pressure rate per blitz."""
from __future__ import annotations

import polars as pl

from ..ledger import Ctx
from ._util import concat, emit


def metrics(ctx: Ctx) -> pl.DataFrame:
    if not ctx.pfr_def.height:
        return concat([])
    d = ctx.pfr_def.unique(keep="first", maintain_order=True, subset=["gsis_id", "game_id"]).filter(pl.col("def_times_blitzed").fill_null(0) > 0)
    return concat([emit(d, "gsis_id", "blitz_pressure", pl.col("def_pressures"), pl.col("def_times_blitzed"))])

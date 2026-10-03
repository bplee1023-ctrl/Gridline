"""Defender metrics shared by EDGE, IDL, LB, CB and S: pass rush, run defense, tackling."""
from __future__ import annotations

import polars as pl

from ..ledger import Ctx, player_game_frame
from ._util import concat, emit, scrimmage, tacklers


def metrics(ctx: Ctx) -> pl.DataFrame:
    pg = player_game_frame(ctx).filter(pl.col("def_snaps") > 0)
    out = []

    # ---- run defense (pbp tackles on designed runs)
    runs = scrimmage(ctx.plays).filter((pl.col("rush") == 1) & (pl.col("qb_scramble") != 1))
    t = tacklers(runs).join(runs.select("game_id", "play_id", "epa", "yards_gained", "w"), on=["game_id", "play_id"])
    per = t.group_by(pl.col("tackler").alias("gsis_id"), "game_id").agg(
        (pl.col("credit") * (pl.col("epa") < 0)).sum().alias("stops"),
        pl.col("tfl").sum().alias("tfl"),
        (pl.col("credit") * pl.col("yards_gained")).sum().alias("depth_sum"),
        pl.col("credit").sum().alias("tackles"))
    rd = pg.join(per, on=["gsis_id", "game_id"], how="left").fill_null(0)
    out.append(emit(rd, "gsis_id", "run_stops", pl.col("stops"), pl.col("def_run_snaps")))
    out.append(emit(rd, "gsis_id", "tfl", pl.col("tfl"), pl.col("def_run_snaps")))
    out.append(emit(rd.filter(pl.col("tackles") > 0), "gsis_id", "tackle_depth", pl.col("depth_sum"), pl.col("tackles")))

    if not ctx.pfr_def.height:
        return concat(out)
    d = ctx.pfr_def.unique(keep="first", maintain_order=True, subset=["gsis_id", "game_id"]).join(
        pg.select("gsis_id", "game_id", "def_pass_snaps"), on=["gsis_id", "game_id"], how="inner")

    # opponent adjustment for pass rush: scale by how easily this opponent's line allows pressure
    team_allowed = ctx.pfr_def.group_by(pl.col("opponent").alias("offense")).agg(pl.col("def_pressures").sum().alias("p")) \
        .join(ctx.team_games.group_by(pl.col("team").alias("offense")).agg(pl.col("dropbacks").sum().alias("db")), on="offense")
    lg = float(team_allowed["p"].sum() / max(team_allowed["db"].sum(), 1))
    kk = 150.0
    team_allowed = team_allowed.with_columns(
        (lg / ((pl.col("p") + kk * lg) / (pl.col("db") + kk))).alias("opp_factor"))
    d = d.join(team_allowed.select(pl.col("offense").alias("opponent"), "opp_factor"), on="opponent", how="left") \
        .with_columns(pl.col("opp_factor").fill_null(1.0))
    # pass-rush snaps: pass snaps, minus estimated coverage drops for off-ball players (handled in db.py)
    rush = d.filter(pl.col("def_pass_snaps") > 0)
    out.append(emit(rush, "gsis_id", "pressure_rate", pl.col("def_pressures") * pl.col("opp_factor"), pl.col("def_pass_snaps")))
    out.append(emit(rush, "gsis_id", "hurry_rate", pl.col("def_times_hurried"), pl.col("def_pass_snaps")))
    out.append(emit(rush, "gsis_id", "knockdown_rate", pl.col("def_times_hitqb"), pl.col("def_pass_snaps")))
    out.append(emit(rush, "gsis_id", "sack_rate", pl.col("def_sacks"), pl.col("def_pass_snaps")))
    out.append(emit(d.filter(pl.col("def_pressures") > 0), "gsis_id", "pressure_to_sack", pl.col("def_sacks"), pl.col("def_pressures")))
    tk = d.with_columns((pl.col("def_missed_tackles").fill_null(0) + pl.col("def_tackles_combined").fill_null(0)).alias("att"))
    out.append(emit(tk.filter(pl.col("att") > 0), "gsis_id", "missed_tackle_rate", pl.col("def_missed_tackles"), pl.col("att")))
    return concat(out)
